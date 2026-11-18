// SPDX-FileCopyrightText: 2026 Hualet Wang
// SPDX-License-Identifier: GPL-3.0-or-later

#include "CrispLiveVoicePipeline.h"

#include "AudioLevelMeter.h"
#include "StreamingDenoiser.h"
#include "LiveStreamTranscriber.h"

#include "CrispAsrEngine.h"
#include "CrispSession.h"
#include "LiveDebugRecorder.h"
#include "LiveSegmenterFactory.h"
#include "Recorder.h"
#include "TextJoiner.h"
#include "log.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstring>
#include <optional>
#include <spawn.h>
#include <stdexcept>
#include <sys/wait.h>
#include <unistd.h>

extern char** environ;

namespace echoflow {

using Clock = std::chrono::steady_clock;

namespace {

double elapsedSeconds(Clock::time_point started)
{
    return std::chrono::duration<double>(Clock::now() - started).count();
}

bool waitForChild(pid_t child, int attempts)
{
    int status = 0;
    for (int i = 0; i < attempts; ++i) {
        pid_t r = waitpid(child, &status, WNOHANG);
        if (r == child) return true;
        if (r < 0 && errno != EINTR) return true;
        usleep(100000);
    }
    return false;
}

std::vector<float> toFloat32(const std::vector<int16_t>& s16)
{
    std::vector<float> out;
    out.reserve(s16.size());
    for (auto s : s16) out.push_back(static_cast<float>(s) / 32768.0f);
    return out;
}

std::string joinText(const std::vector<std::string>& results)
{
    std::string joined;
    for (const auto& text : results) {
        if (!text.empty()) joined = joinOverlappingText(joined, text);
    }
    return joined;
}

}  // namespace

CrispLiveVoicePipeline::CrispLiveVoicePipeline(Config cfg)
    : cfg_(std::move(cfg)), previewScheduler_(cfg_.streamStepMs)
{
}

CrispLiveVoicePipeline::~CrispLiveVoicePipeline()
{
    cancel();
}

void CrispLiveVoicePipeline::setPartialTextCallback(
    std::function<void(const std::string&)> callback)
{
    std::lock_guard<std::mutex> lock(callbackMutex_);
    partialTextCallback_ = std::move(callback);
}

void CrispLiveVoicePipeline::setAudioLevelCallback(std::function<void(const AudioLevelFrame&)> callback)
{
    std::lock_guard<std::mutex> lock(callbackMutex_);
    audioLevelCallback_ = std::move(callback);
}

void CrispLiveVoicePipeline::setConfirmedTextCallback(
    std::function<void(const std::string&)> callback)
{
    std::lock_guard<std::mutex> lock(callbackMutex_);
    confirmedTextCallback_ = std::move(callback);
}

void CrispLiveVoicePipeline::setRecognitionBusyCallback(std::function<void(bool)> callback)
{
    std::lock_guard<std::mutex> lock(callbackMutex_);
    recognitionBusyCallback_ = std::move(callback);
}

void CrispLiveVoicePipeline::emitRecognitionBusy(bool busy)
{
    std::lock_guard<std::mutex> stateLock(recognitionStateMutex_);
    if (!busy && finishing_ && active_) return;
    if (recognitionBusy_.exchange(busy) == busy) return;
    std::function<void(bool)> callback;
    {
        std::lock_guard<std::mutex> lock(callbackMutex_);
        callback = recognitionBusyCallback_;
    }
    if (callback) callback(busy);
}

void CrispLiveVoicePipeline::start()
{
    if (active_) return;

    try {
        startImpl(false);
    } catch (...) {
        cancel();
        throw;
    }
}

void CrispLiveVoicePipeline::startReplay()
{
    if (active_ || !cfg_.streamTranscription) {
        throw std::logic_error("cannot start streaming PCM replay");
    }
    try {
        startImpl(true);
    } catch (...) {
        cancel();
        throw;
    }
}

void CrispLiveVoicePipeline::appendReplayAudio(std::vector<int16_t> samples)
{
    if (!active_ || !replaying_) throw std::logic_error("PCM replay not started");
    if (!cancelled_) processCapturedAudio(std::move(samples));
}

void CrispLiveVoicePipeline::startImpl(bool replay)
{
    replaying_ = replay;
    denoiser_ = cfg_.noiseSuppression ? std::make_unique<StreamingDenoiser>() : nullptr;
    queueHighWaterSamples_ = 0;
    queueMaxWaitMs_ = 0;
    finishing_ = false;
    previewScheduler_ = LivePreviewScheduler(cfg_.streamStepMs);
    const auto loadStarted = Clock::now();
    const bool reused = static_cast<bool>(session_);
    if (!session_) {
        session_ = std::make_unique<CrispSession>(cfg_.crispModelPath, cfg_.crispBackend,
                                                 cfg_.crispThreads);
    }
    if (!session_->isLoaded()) {
        session_.reset();
        throw std::runtime_error("failed to load crisp model: " + cfg_.crispModelPath);
    }
    log(std::string("live ASR model: ") + (reused ? "reused" : "loaded")
        + ", ready_ms=" + std::to_string(std::chrono::duration_cast<
            std::chrono::milliseconds>(Clock::now() - loadStarted).count()));
    auto lang = CrispAsrEngine::languageCode(cfg_.language.value_or(""));
    if (!lang.empty()) session_->setLanguage(lang);
    log("live ASR configuration: language=" + (lang.empty() ? std::string("auto") : lang)
        + ", threads=" + std::to_string(cfg_.crispThreads)
        + ", step_ms=" + std::to_string(cfg_.streamStepMs)
        + ", noise_suppression=" + (cfg_.noiseSuppression ? "rnnoise" : "off"));
    if (cfg_.crispMaxNewTokens > 0) session_->setMaxNewTokens(cfg_.crispMaxNewTokens);

    streamPreviewEnabled_ = true;
    nativeStreaming_ = cfg_.streamTranscription
        && session_->openStream(cfg_.streamStepMs, cfg_.streamStepMs, 800);
    if (nativeStreaming_) {
        session_->closeStream();
        const auto streamConfig = makeLiveStreamConfig(cfg_);
        LiveStreamDecoder decoder;
        decoder.open = [this] {
            if (!session_->openStream(cfg_.streamStepMs, cfg_.streamStepMs, 800)) {
                throw std::runtime_error("failed to reopen native ASR stream");
            }
            session_->setStreamFullSegmentDecode(true);
            segmentFinalizing_ = false;
            streamSilence_ = false;
            previewScheduler_.beginSegment();
            session_->setStreamLiveDecode(false);
        };
        decoder.feed = [this](const int16_t* samples, size_t count) -> std::optional<std::string> {
            std::vector<float> pcm(count);
            for (size_t i = 0; i < count; ++i) pcm[i] = samples[i] / 32768.0f;
            const auto started = Clock::now();
            streamPreviewEnabled_ = previewScheduler_.admit(
                count, streamBacklog_, finishing_ || segmentFinalizing_ || streamSilence_, started);
            // Disable between admissions so the vendor buffers/coalesces audio.
            session_->setStreamLiveDecode(streamPreviewEnabled_);
            if (streamPreviewEnabled_) emitRecognitionBusy(true);
            const auto before = session_->streamInferenceCount();
            const bool changed = session_->feedStream(pcm.data(), static_cast<int>(count));
            const bool decoded = session_->streamInferenceCount() > before;
            const auto decodeMs = std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - started).count();
            if (decoded) {
                previewScheduler_.decoded(std::chrono::milliseconds(decodeMs), Clock::now());
            }
            if (changed || decodeMs >= 20) {
                log("live native partial: chars=" + std::to_string(session_->streamText().size())
                    + ", changed=" + (changed ? "true" : "false")
                    + ", decode_ms=" + std::to_string(decodeMs));
            }
            if (!streamBacklog_) emitRecognitionBusy(false);
            if (decoded || changed) return session_->streamText();
            return std::nullopt;
        };
        decoder.flush = [this] {
            emitRecognitionBusy(true);
            const auto started = Clock::now();
            auto text = session_->flushStream();
            previewScheduler_.decoded(std::chrono::duration_cast<std::chrono::milliseconds>(
                Clock::now() - started), Clock::now());
            log("live native segment confirmed: chars=" + std::to_string(text.size())
                + ", decode_ms=" + std::to_string(std::chrono::duration_cast<
                    std::chrono::milliseconds>(Clock::now() - started).count()));
            if (!streamBacklog_) emitRecognitionBusy(false);
            return text;
        };
        decoder.close = [this] { session_->closeStream(); };
        decoder.deferPreview = [this] { segmentFinalizing_ = true; };
        decoder.setSilence = [this](bool quiet) { streamSilence_ = quiet; };
        if (session_->streamKind() == 3) {
            decoder.checkpoint = [this]() -> std::optional<std::string> {
                if (finishing_ || cancelled_ || streamBacklog_) return std::nullopt;
                emitRecognitionBusy(true);
                const auto started = Clock::now();
                auto text = session_->refreshStream();
                const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - started);
                previewScheduler_.decoded(duration, Clock::now());
                log("live native short pause: chars=" + std::to_string(text.size())
                    + ", decode_ms=" + std::to_string(duration.count()));
                if (!streamBacklog_) emitRecognitionBusy(false);
                return text;
            };
        }
        auto endpoint = makeLiveStreamEndpoint(cfg_, streamConfig.maxSegmentMs);
        log("live native VAD backend: " + endpoint->name());
        streamTranscriber_ = std::make_unique<LiveStreamTranscriber>(
            streamConfig, std::move(decoder),
            [this](const std::string& text) { emitText(text); },
            [this](const std::string& text) { emitConfirmedText(text); }, std::move(endpoint));
        log("crisp live transcription uses bounded native streams, max_segment_ms="
            + std::to_string(streamConfig.maxSegmentMs) + ", kind=" + std::to_string(session_->streamKind())
            + ", step_ms=" + std::to_string(cfg_.streamStepMs));
    } else {
        auto segmenter = makeLiveSegmenter(cfg_);
        coordinator_ = std::make_unique<LiveSegmentCoordinator>(
            std::move(segmenter),
            [this](const AudioSegment& segment) {
                emitRecognitionBusy(true);
                try {
                    auto f32 = toFloat32(segment.samples);
                    auto text = session_->transcribe(f32.data(), static_cast<int>(f32.size()));
                    emitRecognitionBusy(false);
                    return text;
                } catch (...) {
                    sessionFailed_ = true;
                    emitRecognitionBusy(false);
                    throw;
                }
            },
            [this](const std::vector<std::string>& results) {
                const std::string text = joinText(results);
                log("live partial emitted: segments=" + std::to_string(results.size())
                    + ", chars=" + std::to_string(text.size()));
                emitConfirmedText(text);
                emitText(text);
            });
        coordinator_->start();
        log(cfg_.streamTranscription
            ? "crisp native stream unavailable; live transcription uses VAD segments"
            : "crisp live transcription uses VAD segments (streaming disabled)");
    }
    if (replay && !nativeStreaming_) throw std::runtime_error("native streaming unavailable for PCM replay");
    if (!replay && cfg_.saveLiveDebugAudio) {
        debugRecorder_ = std::make_unique<LiveDebugRecorder>(cfg_.recordingsDir);
        debugRecorder_->start();
    }

    if (!replay) startRecording();
    cancelled_ = false;
    {
        std::lock_guard<std::mutex> lock(readerErrorMutex_);
        readerError_ = nullptr;
    }
    inputSamples_ = 0;
    startedAt_ = Clock::now();
    active_ = true;

    if (nativeStreaming_) {
        {
            std::lock_guard<std::mutex> lock(streamQueueMutex_);
            streamAudioQueue_.clear();
            streamInputClosed_ = false;
        }
        streamThread_ = std::thread(&CrispLiveVoicePipeline::nativeStreamLoop, this);
    }
    if (!replay) {
        readerThread_ = std::thread(&CrispLiveVoicePipeline::readerLoop, this);
        log("crisp live pipeline started: source=" +
            (cfg_.pwRecord.source.empty() ? std::string("default") : cfg_.pwRecord.source));
    }
}

void CrispLiveVoicePipeline::startRecording()
{
    int pipeFds[2] = {-1, -1};
    if (pipe(pipeFds) != 0) {
        const std::string error = std::string("pipe failed: ") + std::strerror(errno);
        cancel();
        throw std::runtime_error(error);
    }
    int readFd = pipeFds[0];
    if (readFd == STDOUT_FILENO) {
        readFd = dup(pipeFds[0]); close(pipeFds[0]); pipeFds[0] = readFd;
    }

    posix_spawn_file_actions_t a;
    if (posix_spawn_file_actions_init(&a) != 0) {
        close(readFd); close(pipeFds[1]);
        cancel();
        throw std::runtime_error("pw-record spawn init failed");
    }
    posix_spawn_file_actions_adddup2(&a, pipeFds[1], STDOUT_FILENO);
    posix_spawn_file_actions_addclose(&a, readFd);
    posix_spawn_file_actions_addclose(&a, pipeFds[1]);

    auto args = buildPipeWireLiveRecordArgs(cfg_);
    std::vector<char*> argv; argv.reserve(args.size() + 1);
    for (auto& s : args) argv.push_back(s.data());
    argv.push_back(nullptr);

    pid_t pid = -1;
    int rc = posix_spawnp(&pid, "pw-record", &a, nullptr, argv.data(), environ);
    posix_spawn_file_actions_destroy(&a);
    if (rc != 0) {
        close(readFd); close(pipeFds[1]);
        cancel();
        throw std::runtime_error(std::string("posix_spawnp pw-record failed: ") + std::strerror(rc));
    }
    close(pipeFds[1]);

    recorderChild_ = pid;
    readFd_ = readFd;
}


void CrispLiveVoicePipeline::readerLoop()
{
    try {
        AudioLevelMeter levelMeter(16000, 1, [this](const AudioLevelFrame& level) { emitAudioLevel(level); }, true);
        std::array<unsigned char, 6400> buf{};
        bool hasCarry = false;
        unsigned char carry = 0;
        while (readFd_ != -1 && !cancelled_) {
            ssize_t n = read(readFd_, buf.data(), buf.size());
            if (n > 0) {
                size_t byteCount = static_cast<size_t>(n);
                std::vector<int16_t> samples;
                samples.reserve((byteCount + (hasCarry ? 1 : 0)) / 2);
                size_t offset = 0;
                if (hasCarry && byteCount > 0) {
                    unsigned int raw = static_cast<unsigned int>(carry)
                        | (static_cast<unsigned int>(buf[0]) << 8);
                    samples.push_back(raw >= 0x8000U ? static_cast<int>(raw) - 0x10000
                                                     : static_cast<int>(raw));
                    hasCarry = false; offset = 1;
                }
                for (; offset + 1 < byteCount; offset += 2) {
                    unsigned int raw = static_cast<unsigned int>(buf[offset])
                        | (static_cast<unsigned int>(buf[offset + 1]) << 8);
                    samples.push_back(raw >= 0x8000U ? static_cast<int>(raw) - 0x10000
                                                     : static_cast<int>(raw));
                }
                if (offset < byteCount) { carry = buf[offset]; hasCarry = true; }

                levelMeter.append(samples.data(), samples.size());
                processCapturedAudio(std::move(samples));
                continue;
            }
            if (n == 0) break;
            if (errno == EINTR) continue;
            throw std::runtime_error(std::string("audio capture read failed: ") + std::strerror(errno));
        }
        if (denoiser_ && !cancelled_) enqueueAudio(denoiser_->finishS16());
    } catch (const std::exception& e) {
        {
            std::lock_guard<std::mutex> lock(readerErrorMutex_);
            if (!readerError_) readerError_ = std::current_exception();
        }
        log(std::string("crisp live reader failed: ") + e.what());
    } catch (...) {
        {
            std::lock_guard<std::mutex> lock(readerErrorMutex_);
            if (!readerError_) readerError_ = std::current_exception();
        }
        log("crisp live reader failed");
    }
}

void CrispLiveVoicePipeline::processCapturedAudio(std::vector<int16_t> samples)
{
    inputSamples_ += samples.size();
    if (debugRecorder_) debugRecorder_->append(samples.data(), samples.size());
    if (denoiser_) samples = denoiser_->append(samples.data(), samples.size());
    enqueueAudio(std::move(samples));
}

void CrispLiveVoicePipeline::enqueueAudio(std::vector<int16_t> samples)
{
    if (samples.empty()) return;
    if (nativeStreaming_) {
        std::lock_guard<std::mutex> lock(streamQueueMutex_);
        const bool wasSpilling = streamAudioQueue_.spilling();
        streamAudioQueue_.append(std::move(samples));
        queueHighWaterSamples_ = std::max(queueHighWaterSamples_, streamAudioQueue_.size());
        if (!wasSpilling && streamAudioQueue_.spilling()) {
            log("live ASR backlog spooled to temporary storage");
        }
        streamQueueReady_.notify_one();
    } else if (!coordinator_ || !coordinator_->append(samples.data(), samples.size())) {
        throw std::runtime_error("live segment queue stopped accepting audio");
    }
}

void CrispLiveVoicePipeline::nativeStreamLoop()
{
    try {
        auto nextDiagnostic = Clock::now();
        while (true) {
            std::vector<int16_t> samples;
            {
                std::unique_lock<std::mutex> lock(streamQueueMutex_);
                streamQueueReady_.wait(lock, [this] {
                    return cancelled_ || !streamAudioQueue_.empty() || streamInputClosed_;
                });
                if (cancelled_) return;
                if (streamAudioQueue_.empty()) {
                    if (streamInputClosed_) return;
                    continue;
                }
                std::chrono::milliseconds queueWait;
                samples = streamAudioQueue_.take(&queueWait);
                queueMaxWaitMs_ = std::max(queueMaxWaitMs_, static_cast<long long>(queueWait.count()));
                streamBacklog_ = streamAudioQueue_.size() >= 16000 / 2;
                if (Clock::now() >= nextDiagnostic) {
                    log("live ASR progress: queued_ms=" + std::to_string(streamAudioQueue_.size() / 16)
                        + ", queue_wait_ms=" + std::to_string(queueWait.count())
                        + ", preview_interval_ms=" + std::to_string(previewScheduler_.intervalMs()));
                    nextDiagnostic = Clock::now() + std::chrono::seconds(1);
                }
            }

            if (streamBacklog_ || finishing_) emitRecognitionBusy(true);
            streamTranscriber_->append(samples.data(), samples.size());
            bool idle;
            {
                std::lock_guard<std::mutex> lock(streamQueueMutex_);
                idle = streamAudioQueue_.empty() && !finishing_;
            }
            if (idle) emitRecognitionBusy(false);
        }
    } catch (const std::exception& e) {
        {
            std::lock_guard<std::mutex> lock(readerErrorMutex_);
            if (!readerError_) readerError_ = std::current_exception();
        }
        sessionFailed_ = true;
        log(std::string("crisp native stream worker failed: ") + e.what());
        cancelled_ = true;
        streamQueueReady_.notify_all();
    } catch (...) {
        {
            std::lock_guard<std::mutex> lock(readerErrorMutex_);
            if (!readerError_) readerError_ = std::current_exception();
        }
        sessionFailed_ = true;
        log("crisp native stream worker failed");
        cancelled_ = true;
        streamQueueReady_.notify_all();
    }
}

void CrispLiveVoicePipeline::stopNativeStreamInput(bool discard)
{
    {
        std::lock_guard<std::mutex> lock(streamQueueMutex_);
        streamInputClosed_ = true;
        if (discard) {
            streamAudioQueue_.clear();
        }
    }
    streamQueueReady_.notify_all();
    if (streamThread_.joinable()) {
        streamThread_.join();
    }
}

std::string CrispLiveVoicePipeline::finish()
{
    if (!active_) return {};
    finishing_ = true;
    emitRecognitionBusy(true);
    auto finishStarted = Clock::now();
    log("crisp live pipeline stop after " + std::to_string(elapsedSeconds(startedAt_)) + "s");

    stopRecorder();
    reapChild(recorderChild_);
    if (readerThread_.joinable()) readerThread_.join();
    if (readFd_ != -1) { close(readFd_); readFd_ = -1; }
    if (replaying_ && denoiser_ && !cancelled_) enqueueAudio(denoiser_->finishS16());
    if (nativeStreaming_) stopNativeStreamInput(false);

    if (readerError_) {
        if (coordinator_) coordinator_->cancel();
        cleanupAfterFinish();
        std::rethrow_exception(readerError_);
    }

    std::string finalText;
    try {
        std::vector<std::string> debugResults;
        LivePipelineMetrics debugMetrics;
        if (!cancelled_) {
            if (nativeStreaming_) {
                finalText = streamTranscriber_->finish();
                debugResults.push_back(finalText);
                debugMetrics.inputSamples = inputSamples_;
                debugMetrics.enqueuedSegments = streamTranscriber_->confirmedSegments();
            } else if (coordinator_) {
                debugResults = coordinator_->finish();
                finalText = joinText(debugResults);
                debugMetrics = coordinator_->metrics();
            }
            if (debugRecorder_) {
                debugRecorder_->finish(debugMetrics, debugResults);
                log("live debug audio saved: " + debugRecorder_->wavPath().string());
            }
        }
    } catch (...) {
        sessionFailed_ = true;
        cleanupAfterFinish();
        throw;
    }
    cleanupAfterFinish();
    log("crisp live finish in " + std::to_string(elapsedSeconds(finishStarted)) +
        "s, chars=" + std::to_string(finalText.size()));
    return finalText;
}

void CrispLiveVoicePipeline::cleanupAfterFinish()
{
    active_ = false;
    nativeStreaming_ = false;
    inputSamples_ = 0;
    {
        std::lock_guard<std::mutex> lock(streamQueueMutex_);
        streamAudioQueue_.clear();
        streamInputClosed_ = false;
    }
    log("live capture metrics: queue_high_water_ms=" + std::to_string(queueHighWaterSamples_ / 16)
        + ", queue_max_wait_ms=" + std::to_string(queueMaxWaitMs_)
        + ", denoise_ms=" + std::to_string(denoiser_ ? denoiser_->processingMs() : 0));
    denoiser_.reset();
    debugRecorder_.reset();
    coordinator_.reset();
    if (streamTranscriber_) streamTranscriber_->cancel();
    streamTranscriber_.reset();
    if (session_) session_->closeStream();
    if (sessionFailed_.exchange(false)) session_.reset();
    emitRecognitionBusy(false);
}

void CrispLiveVoicePipeline::cancel()
{
    try {
        cancelled_ = true;
        stopRecorder();
        reapChild(recorderChild_);
        if (readFd_ != -1) { close(readFd_); readFd_ = -1; }
        if (readerThread_.joinable()) readerThread_.join();
        if (nativeStreaming_) stopNativeStreamInput(true);
        active_ = false;
        if (coordinator_) coordinator_->cancel();
        coordinator_.reset();
        nativeStreaming_ = false;
        denoiser_.reset();
        debugRecorder_.reset();
        if (streamTranscriber_) streamTranscriber_->cancel();
        streamTranscriber_.reset();
        if (session_) session_->closeStream();
        if (sessionFailed_.exchange(false)) session_.reset();
        emitRecognitionBusy(false);
    } catch (const std::exception& e) {
        log(std::string("crisp pipeline cancel failed: ") + e.what());
    } catch (...) {
        log("crisp pipeline cancel failed");
    }
}

void CrispLiveVoicePipeline::stopRecorder()
{
    if (recorderChild_ != -1) kill(recorderChild_, SIGINT);
}

void CrispLiveVoicePipeline::reapChild(pid_t& child)
{
    if (child == -1) return;
    pid_t c = child;
    if (!waitForChild(c, 50)) {
        kill(c, SIGTERM);
        if (!waitForChild(c, 20)) {
            kill(c, SIGKILL);
            waitForChild(c, 20);
        }
    }
    child = -1;
}

void CrispLiveVoicePipeline::emitAudioLevel(const AudioLevelFrame& level)
{
    std::function<void(const AudioLevelFrame&)> callback;
    {
        std::lock_guard<std::mutex> lock(callbackMutex_);
        callback = audioLevelCallback_;
    }
    if (callback) callback(level);
}

void CrispLiveVoicePipeline::emitText(const std::string& text)
{
    std::function<void(const std::string&)> cb;
    {
        std::lock_guard<std::mutex> lock(callbackMutex_);
        cb = partialTextCallback_;
    }
    if (cb) cb(text);
}

void CrispLiveVoicePipeline::emitConfirmedText(const std::string& text)
{
    std::function<void(const std::string&)> callback;
    {
        std::lock_guard<std::mutex> lock(callbackMutex_);
        callback = confirmedTextCallback_;
    }
    if (callback) callback(text);
}

}  // namespace echoflow
