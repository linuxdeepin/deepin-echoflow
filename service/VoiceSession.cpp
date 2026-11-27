// SPDX-FileCopyrightText: 2026 Hualet Wang
// SPDX-License-Identifier: GPL-3.0-or-later

#include "VoiceSession.h"

#include "log.h"
#include "StreamingTextPolisher.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <exception>
#include <filesystem>
#include <sstream>

namespace echoflow {

namespace {

std::string trim(const std::string& value)
{
    auto first = value.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) {
        return {};
    }
    auto last = value.find_last_not_of(" \t\r\n");
    return value.substr(first, last - first + 1);
}

std::string upper(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(),
        [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
    return value;
}

// Normalizes "x y w h" caret geometry. Malformed geometry is dropped so the UI
// host treats the message as focus without a known caret.
std::string focusMessage(const std::string& verb, const std::string& argument)
{
    if (verb == "TYPED") {
        return verb;
    }
    std::istringstream stream(argument);
    long x = 0;
    long y = 0;
    long width = 0;
    long height = 0;
    std::string rest;
    if (!(stream >> x >> y >> width >> height) || (stream >> rest) || width < 0
        || height <= 0) {
        return verb;
    }
    return verb + " " + std::to_string(x) + " " + std::to_string(y) + " "
        + std::to_string(width) + " " + std::to_string(height);
}

VoiceSession::ThreadLauncher defaultThreadLauncher()
{
    return [](std::function<void()> task) {
        return std::thread(std::move(task));
    };
}

} // namespace

VoiceSession::VoiceSession(Config cfg, IRecorder& recorder, IAsrEngine& asr,
    ICommitter& committer, IUiNotifier& ui, ThreadLauncher threadLauncher)
    : cfg_(std::move(cfg))
    , recorder_(&recorder)
    , asr_(&asr)
    , committer_(committer)
    , ui_(ui)
    , threadLauncher_(threadLauncher ? std::move(threadLauncher) : defaultThreadLauncher())
{
    nextGeneration_ = static_cast<std::uint64_t>(
        std::chrono::steady_clock::now().time_since_epoch().count());
    setupAudioLevelCallback();
}

VoiceSession::VoiceSession(Config cfg, ILiveVoicePipeline& livePipeline,
    ICommitter& committer, IUiNotifier& ui, ThreadLauncher threadLauncher)
    : cfg_(std::move(cfg))
    , livePipeline_(&livePipeline)
    , committer_(committer)
    , ui_(ui)
    , threadLauncher_(threadLauncher ? std::move(threadLauncher) : defaultThreadLauncher())
    , partialCallbackState_(std::make_shared<PartialCallbackState>())
{
    nextGeneration_ = static_cast<std::uint64_t>(
        std::chrono::steady_clock::now().time_since_epoch().count());
    setupAudioLevelCallback();
    partialCallbackState_->ui = &ui_;
    partialCallbackState_->committer = &committer_;
    livePipeline_->setPartialTextCallback(
        [state = partialCallbackState_](const std::string& text) {
            {
                std::lock_guard<std::mutex> lock(state->mutex);
                if (!state->accepting && !state->preeditAccepting) return;
                state->rawText = text;
                if (!state->pendingConfirmedRaw.empty()
                    && text.rfind(state->pendingConfirmedRaw, 0) == 0) {
                    state->confirmedRaw = std::move(state->pendingConfirmedRaw);
                    state->polishedConfirmed = std::move(state->pendingPolished);
                    state->pendingConfirmedRaw.clear();
                    state->pendingPolished.clear();
                }
            }
            state->condition.notify_all();
            publishPartial(state);
        });
}

void VoiceSession::setupAudioLevelCallback()
{
    audioLevelCallbackState_ = std::make_shared<AudioLevelCallbackState>();
    audioLevelCallbackState_->ui = &ui_;
    auto callback = [state = audioLevelCallbackState_](const AudioLevelFrame& frame) {
        IUiNotifier* notifier;
        {
            std::lock_guard<std::mutex> lock(state->mutex);
            if (!state->accepting || !state->ui || !std::isfinite(frame.level)) return;
            ++state->active;
            notifier = state->ui;
        }
        // Independent of preedit delivery: a busy ASR/commit socket must not
        // delay metering on the capture thread. UI sends are nonblocking.
        try {
            const auto encode = [](double level) {
                return std::to_string(static_cast<int>(std::lround(
                    (std::isfinite(level) ? std::clamp(level, 0.0, 1.0) : 0.0) * 1000)));
            };
            std::string message = "AUDIO_LEVEL " + encode(frame.level);
            for (const double bar : frame.bars) message += " " + encode(bar);
            notifier->send(message);
        } catch (...) {
            log("voice audio level notification failed");
        }
        {
            std::lock_guard<std::mutex> lock(state->mutex);
            --state->active;
        }
        state->condition.notify_all();
    };
    if (livePipeline_) livePipeline_->setAudioLevelCallback(std::move(callback));
    else recorder_->setAudioLevelCallback(std::move(callback));
}

void VoiceSession::setAudioLevelAccepting(bool accepting)
{
    const auto state = audioLevelCallbackState_;
    std::unique_lock<std::mutex> lock(state->mutex);
    state->accepting = accepting;
    if (!accepting) state->condition.wait(lock, [state] { return state->active == 0; });
}

void VoiceSession::shutdownAudioLevelCallback()
{
    setAudioLevelAccepting(false);
    if (livePipeline_) livePipeline_->setAudioLevelCallback({});
    else recorder_->setAudioLevelCallback({});
    std::lock_guard<std::mutex> lock(audioLevelCallbackState_->mutex);
    audioLevelCallbackState_->ui = nullptr;
}

void VoiceSession::publishPartial(const std::shared_ptr<PartialCallbackState>& state)
{
    // Serialize delivery, including retries, so old previews cannot overtake
    // a newer ASR/polish result. Render again after taking this lock.
    std::lock_guard<std::mutex> delivery(state->deliveryMutex);
    const auto render = [&] {
        auto text = state->rawText;
        if (!state->confirmedRaw.empty() && text.rfind(state->confirmedRaw, 0) == 0) {
            text = state->polishedConfirmed + text.substr(state->confirmedRaw.size());
        }
        return text;
    };
    IUiNotifier* notifier = nullptr;
    ICommitter* committer = nullptr;
    std::uint64_t generation = 0;
    std::uint64_t revision = 0;
    std::string text;
    {
        std::lock_guard<std::mutex> lock(state->mutex);
        if (!state->accepting && !state->preeditAccepting) return;
        text = render();
        if (state->published && text == state->lastPublished) return;
        // A failed/uncertain send must not suppress a later identical result.
        state->published = false;
        ++state->active;
        notifier = state->ui;
        if (state->preeditAccepting) committer = state->committer;
        generation = state->generation;
        revision = ++state->revision;
    }
    const auto finishNotification = [&] {
        {
            std::lock_guard<std::mutex> lock(state->mutex);
            --state->active;
        }
        state->condition.notify_all();
    };
    try {
        constexpr int kMaxAttempts = 3;
        for (int attempt = 0; attempt < kMaxAttempts; ++attempt) {
            bool ok = false;
            bool retryable = false;
            try {
                const auto result = committer
                    ? committer->setPreedit(text, generation, revision)
                    : std::pair<bool, std::string>{true, "disabled"};
                ok = result.first;
                const auto& detail = result.second;
                retryable = detail == "socket" || detail == "bind"
                    || detail == "sendto" || detail == "timeout";
                if (!ok && detail != "preedit unsupported") {
                    log("voice preedit update failed: " + detail);
                }
            } catch (const std::exception& error) {
                retryable = true;
                log(std::string("voice preedit update failed: ") + error.what());
            }
            std::unique_lock<std::mutex> lock(state->mutex);
            if (ok) {
                state->lastPublished = text;
                state->published = true;
                break;
            }
            if (!retryable || attempt + 1 == kMaxAttempts) break;
            // Wake immediately when cancellation or newer text supersedes this
            // attempt; no later ASR callback is needed to retry stable text.
            const bool superseded = state->condition.wait_for(lock, std::chrono::milliseconds(40), [&] {
                return !state->preeditAccepting || state->generation != generation || render() != text;
            });
            if (superseded) break;
            revision = ++state->revision;
        }
        if (notifier) notifier->send("STREAM_TEXT " + text);
    } catch (...) {
        finishNotification();
        throw;
    }
    finishNotification();
}

VoiceSession::~VoiceSession()
{
    bool cancelRecording = false;
    std::uint64_t shutdownGeneration = 0;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        shuttingDown_ = true;
        if (state_ == SessionState::Recording) {
            cancelRecording = true;
            shutdownGeneration = runningGeneration_;
            setStateLocked(SessionState::Idle);
            runningGeneration_ = 0;
        } else if (state_ == SessionState::Transcribing || state_ == SessionState::Cancelling) {
            shutdownGeneration = runningGeneration_;
            cancelRequested_ = true;
            setStateLocked(SessionState::Cancelling);
        }
    }
    condition_.notify_all();
    if (streamingPolisher_) streamingPolisher_->cancel();

    // Close the finalization/commit gate before callback teardown can block.
    // No external collaborator is called while the session mutex is held.
    shutdownAudioLevelCallback();
    shutdownPartialTextCallback();
    if (shutdownGeneration != 0) {
        try {
            committer_.cancelVoiceSession(shutdownGeneration);
        } catch (const std::exception& e) {
            log(std::string("voice session cancel during shutdown failed: ") + e.what());
        }
    }

    if (cancelRecording) {
        try {
            if (livePipeline_) {
                livePipeline_->cancel();
            } else {
                recorder_->stop();
            }
        } catch (const std::exception& e) {
            log(std::string("voice cancel failed: ") + e.what());
        }
    }

    if (finalizationWorker_.joinable()) {
        finalizationWorker_.join();
    }
    if (livePipeline_) {
        livePipeline_->setConfirmedTextCallback({});
        livePipeline_->setRecognitionBusyCallback({});
    }
    streamingPolisher_.reset();
}

SessionState VoiceSession::state() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return state_;
}

void VoiceSession::setTextPolisher(ITextPolisher* polisher)
{
    streamingPolisher_.reset();
    if (polisher) {
        const auto state = partialCallbackState_;
        streamingPolisher_ = std::make_shared<StreamingTextPolisher>(*polisher,
            [state](std::uint64_t session, const std::string& raw, const std::string& polished) {
                if (!state) return;
                {
                    std::lock_guard<std::mutex> lock(state->mutex);
                    if (session != state->polishSession || state->polishSession == 0
                        || raw.size() < std::max(state->confirmedRaw.size(),
                                                 state->pendingConfirmedRaw.size())) return;
                    if (state->rawText.rfind(raw, 0) == 0) {
                        state->confirmedRaw = raw;
                        state->polishedConfirmed = polished;
                        state->pendingConfirmedRaw.clear();
                        state->pendingPolished.clear();
                    } else {
                        // Confirmation can precede the corresponding ASR preview.
                        // Keep the already-rendered prefix until ASR catches up.
                        state->pendingConfirmedRaw = raw;
                        state->pendingPolished = polished;
                    }
                }
                state->condition.notify_all();
                publishPartial(state);
            });
    }
    if (livePipeline_) {
        const std::weak_ptr<StreamingTextPolisher> weak = streamingPolisher_;
        livePipeline_->setRecognitionBusyCallback([weak](bool busy) {
            if (const auto worker = weak.lock()) worker->setRecognitionBusy(busy);
        });
        livePipeline_->setConfirmedTextCallback([weak](const std::string& text) {
            if (const auto worker = weak.lock()) worker->appendConfirmed(text);
        });
    }
}

std::string VoiceSession::handleCommand(const std::string& command)
{
    std::string cmd = trim(command);
    auto space = cmd.find(' ');
    std::string verb = upper(space == std::string::npos ? cmd : cmd.substr(0, space));
    std::string argument = trim(space == std::string::npos ? std::string() : cmd.substr(space + 1));

    // Keep accepting focus reports from older addons. Current addons notify
    // the UI directly so focus tracking also works while this service is idle.
    // These compatibility messages never change the session state.
    if (verb == "FOCUS" || verb == "CARET" || verb == "TYPED") {
        ui_.send(focusMessage(verb, argument));
        return "OK";
    }

    if (verb == "BLUR") {
        // Focus left the input context: discard any live recording the same way
        // the X button does, then hide the capsule.
        // New addons already sent BLUR on the UI focus channel. Relaying it
        // again here could overwrite a newer FOCUS while ASR was busy.
        if (argument != "ui-notified") {
            ui_.send("BLUR");
        }
        cancelActiveSession();
        return "TOOLTIP hide";
    }

    // START/STOP back the press-and-hold trigger. Unlike CTRL_DOWN they never
    // toggle, so a hold cannot stop a session it did not start from idle or
    // start a new one while finishing.
    if (verb == "START") {
        const auto currentState = state();
        if (currentState == SessionState::Idle) {
            return startRecording();
        }
        return phaseName(currentState);
    }

    if (verb == "STOP") {
        if (state() == SessionState::Recording) {
            return startFinalization();
        }
        return phaseName(state());
    }

    if (verb == "CANCEL") {
        // The capsule's X button: discard the recording and hide.
        return cancelActiveSession();
    }

    if (verb == "STATUS") {
        const auto [phase, version] = phaseSnapshot();
        publishPhase(phase, version, true);
        return phaseName(phase);
    }

    if (verb == "CTRL_DOWN") {
        const auto currentState = state();
        if (currentState == SessionState::Idle) {
            return startRecording();
        }
        if (currentState == SessionState::Recording) {
            return startFinalization();
        }
        if (currentState == SessionState::Cancelling) {
            return "CANCELLING";
        }
        return "TRANSCRIBING";
    }

    return "ERR unknown-command";
}

std::string VoiceSession::startRecording()
{
    std::error_code modelError;
    if (!cfg_.crispModelPath.empty()
        && !std::filesystem::exists(cfg_.crispModelPath, modelError) && !modelError) {
        ui_.send("MODEL_REQUIRED");
        return "ERR model-not-installed";
    }
    joinCompletedWorker();

    std::uint64_t generation;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        generation = ++nextGeneration_;
    }
    const auto failStart = [&](const std::string& error) {
        // The plugin may have accepted the generation even if its reply timed out.
        try {
            const auto [ok, detail] = committer_.cancelVoiceSession(generation);
            if (!ok) log("voice begin rollback cancel failed: " + detail);
        } catch (const std::exception& e) {
            log(std::string("voice begin rollback cancel failed: ") + e.what());
        }
        const auto [phase, version] = phaseSnapshot();
        publishPhase(phase, version, true);
        return "ERR " + error;
    };

    // Focus notifications can be stale by the time a mouse hold elapses.
    // Bind the current eligible input context before opening the microphone.
    std::string beginError;
    try {
        const auto [ok, detail] = committer_.beginVoiceSession(generation);
        if (!ok) beginError = detail.empty() ? "voice session begin rejected" : detail;
    } catch (const std::exception& e) {
        beginError = e.what();
    }
    if (!beginError.empty()) {
        log("voice session begin failed: " + beginError);
        return failStart(beginError);
    }

    try {
        const auto polishSession = streamingPolisher_ ? streamingPolisher_->start() : 0;
        if (partialCallbackState_) {
            std::lock_guard<std::mutex> lock(partialCallbackState_->mutex);
            partialCallbackState_->polishSession = polishSession;
            partialCallbackState_->rawText.clear();
            partialCallbackState_->confirmedRaw.clear();
            partialCallbackState_->polishedConfirmed.clear();
            partialCallbackState_->pendingConfirmedRaw.clear();
            partialCallbackState_->pendingPolished.clear();
            partialCallbackState_->lastPublished.clear();
            partialCallbackState_->published = false;
            partialCallbackState_->generation = generation;
            partialCallbackState_->revision = 0;
        }
        if (livePipeline_) {
            livePipeline_->start();
        } else {
            recorder_->start();
        }
    } catch (const std::exception& e) {
        log(std::string("voice start failed: ") + e.what());
        if (streamingPolisher_) streamingPolisher_->cancel();
        if (livePipeline_) {
            try {
                livePipeline_->cancel();
            } catch (const std::exception& cancelError) {
                log(std::string("voice start rollback failed: ") + cancelError.what());
            }
        }
        return failStart(e.what());
    }

    std::uint64_t phaseVersion;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        runningGeneration_ = generation;
        cancelRequested_ = false;
        commitClaimed_ = false;
        finalizationStartPublished_ = false;
        phaseVersion = setStateLocked(SessionState::Recording);
    }
    publishPhase(SessionState::Recording, phaseVersion);
    setAudioLevelAccepting(true);
    setPartialTextAccepting(true);
    return "RECORDING";
}

std::string VoiceSession::startFinalization()
{
    std::uint64_t generation = 0;
    std::uint64_t phaseVersion = 0;
    std::string launchError;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (state_ != SessionState::Recording) {
            return phaseName(state_);
        }

        // The pipeline is about to be finalized; stop accepting streaming
        // partials. The worker waits for TRANSCRIBING to be published before
        // touching capture/ASR, so notifier calls stay outside this mutex.
        phaseVersion = setStateLocked(SessionState::Transcribing);
        finalizationRunning_ = true;
        finalizationStartPublished_ = false;
        commitClaimed_ = false;
        generation = runningGeneration_;
        try {
            finalizationWorker_
                = threadLauncher_([this, generation] { finalize(generation); });
            if (livePipeline_) livePipeline_->beginFinalization();
        } catch (const std::system_error& e) {
            launchError = e.what();
            phaseVersion = setStateLocked(SessionState::Recording);
            finalizationRunning_ = false;
            finalizationStartPublished_ = false;
        }
    }

    if (!launchError.empty()) {
        log("voice finalization launch failed: " + launchError);
        publishPhase(SessionState::Recording, phaseVersion);
        return "ERR " + launchError;
    }

    setAudioLevelAccepting(false);
    setPartialTextAccepting(false, true);
    publishPhase(SessionState::Transcribing, phaseVersion);
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (finalizationRunning_ && runningGeneration_ == generation) {
            finalizationStartPublished_ = true;
        }
    }
    condition_.notify_all();
    return "TRANSCRIBING";
}

void VoiceSession::finalize(std::uint64_t generation)
{
    {
        std::unique_lock<std::mutex> lock(mutex_);
        condition_.wait(lock, [this, generation] {
            return shuttingDown_ || (finalizationRunning_
                && runningGeneration_ == generation && finalizationStartPublished_);
        });
        if (shuttingDown_ && !finalizationStartPublished_) {
            return;
        }
    }

    std::string text;
    bool recognitionFailed = false;
    if (livePipeline_) {
        try {
            text = livePipeline_->finish();
        } catch (const std::exception& e) {
            recognitionFailed = true;
            log(std::string("asr transcribe failed: ") + e.what());
        }
    } else {
        try {
            const auto audio = recorder_->stop();
            if (!audio.empty()) {
                text = asr_->transcribe(audio);
            }
        } catch (const std::exception& e) {
            log(std::string("asr transcribe failed: ") + e.what());
        }
    }
    setPartialTextAccepting(false, true);
    if (recognitionFailed && partialCallbackState_) {
        std::lock_guard<std::mutex> lock(partialCallbackState_->mutex);
        text = partialCallbackState_->rawText;
        log("ASR finalization failed; retaining recognized preview, bytes=" + std::to_string(text.size()));
    }
    if (cfg_.logTranscripts) {
        log("transcript: " + text);
    }
    if (streamingPolisher_) {
        if (!text.empty()) text = polishText(text);
        else streamingPolisher_->cancel();
    }
    setPartialTextAccepting(false);
    if (cfg_.stripTrailingPunctuation) {
        text = stripPunctuation(text);
    }
    if (cfg_.logTranscripts) {
        log("commit text: " + text);
    }

    // Keep the composition visible until the addon handles COMMIT. Clearing
    // it in a separate request creates a blank gap and loses the preview if
    // the commit has not reached the addon yet.
    bool commitClaimed = false;
    {
        std::unique_lock<std::mutex> lock(mutex_);
        const bool current = finalizationRunning_ && runningGeneration_ == generation;

        // This is the cancellation/commit claim gate. CANCEL taking it first
        // suppresses the external commit. A worker taking it first records an
        // irrevocable claim, then releases the mutex before commitText().
        if (current && !cancelRequested_ && !shuttingDown_ && !text.empty()) {
            commitClaimed_ = true;
            commitClaimed = true;
        }
    }

    if (commitClaimed) {
        try {
            const auto [ok, detail] = committer_.commitText(text, generation);
            if (!ok) {
                log("voice commit failed: " + detail);
                committer_.cancelVoiceSession(generation);
            }
        } catch (const std::exception& e) {
            log(std::string("voice commit failed: ") + e.what());
            try { committer_.cancelVoiceSession(generation); } catch (...) {}
        }
    } else {
        try {
            committer_.cancelVoiceSession(generation);
        } catch (const std::exception& e) {
            log(std::string("voice session cleanup failed: ") + e.what());
        }
    }

    bool notifyIdle = false;
    std::uint64_t idleVersion = 0;
    {
        std::unique_lock<std::mutex> lock(mutex_);
        const bool current = finalizationRunning_ && runningGeneration_ == generation;
        if (current) {
            // An accepted CANCEL increments the pending count before its
            // notifier call. Waiting here releases mutex_, guaranteeing
            // CANCELLING is published before the terminal IDLE.
            condition_.wait(lock,
                [this] { return pendingCancellationNotifications_ == 0; });
            idleVersion = setStateLocked(SessionState::Idle);
            runningGeneration_ = 0;
            finalizationRunning_ = false;
            finalizationStartPublished_ = false;
            commitClaimed_ = false;
            cancelRequested_ = false;
            notifyIdle = !shuttingDown_;
        }
    }
    if (notifyIdle) {
        publishPhase(SessionState::Idle, idleVersion);
    }
}

std::string VoiceSession::polishText(const std::string& text)
{
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (cancelRequested_ || shuttingDown_) return text;
    }
    return streamingPolisher_->finish(text);
}

void VoiceSession::joinCompletedWorker()
{
    std::thread completedWorker;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (finalizationWorker_.joinable() && !finalizationRunning_) {
            completedWorker = std::move(finalizationWorker_);
        }
    }
    if (completedWorker.joinable()) {
        completedWorker.join();
    }
}

void VoiceSession::setPartialTextAccepting(bool accepting, bool keepPreedit)
{
    if (!partialCallbackState_) {
        return;
    }

    const auto state = partialCallbackState_;
    std::unique_lock<std::mutex> lock(state->mutex);
    state->accepting = accepting;
    state->preeditAccepting = accepting || keepPreedit;
    if (!accepting && !keepPreedit) state->polishSession = 0;
    state->condition.notify_all();
    if (!accepting) {
        state->condition.wait(lock, [state] { return state->active == 0; });
    }
}

void VoiceSession::shutdownPartialTextCallback()
{
    if (!partialCallbackState_) {
        return;
    }

    const auto state = partialCallbackState_;
    {
        std::lock_guard<std::mutex> lock(state->mutex);
        state->accepting = false;
        state->preeditAccepting = false;
        state->polishSession = 0;
    }
    state->condition.notify_all();
    livePipeline_->setPartialTextCallback({});
    {
        std::unique_lock<std::mutex> lock(state->mutex);
        state->condition.wait(lock, [state] { return state->active == 0; });
        state->ui = nullptr;
    }
}

std::string VoiceSession::cancelActiveSession()
{
    bool cancelRecording = false;
    bool publishCancellation = false;
    std::uint64_t cancellationGeneration = 0;
    std::string result = "CANCELLED";
    SessionState notification = SessionState::Idle;
    std::uint64_t notificationVersion = 0;
    bool hasNotification = false;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (state_ == SessionState::Recording) {
            cancelRecording = true;
            cancellationGeneration = runningGeneration_;
            notificationVersion = setStateLocked(SessionState::Idle);
            runningGeneration_ = 0;
            cancelRequested_ = false;
            hasNotification = true;
        } else if (state_ == SessionState::Transcribing && commitClaimed_) {
            // The external commit has already won the claim gate. Report the
            // in-flight phase honestly; cancellation can no longer suppress it.
            result = "TRANSCRIBING";
        } else if (state_ == SessionState::Transcribing || state_ == SessionState::Cancelling) {
            cancellationGeneration = runningGeneration_;
            cancelRequested_ = true;
            notification = SessionState::Cancelling;
            notificationVersion = setStateLocked(SessionState::Cancelling);
            ++pendingCancellationNotifications_;
            publishCancellation = true;
        } else {
            notificationVersion = stateVersion_;
            hasNotification = true;
        }
    }

    if (cancelRecording || publishCancellation) {
        setAudioLevelAccepting(false);
        if (streamingPolisher_) streamingPolisher_->cancel();
        // No phase after Recording may overtake an already-admitted partial.
        // This wait uses only the callback-state mutex and releases it while
        // the external notifier completes.
        setPartialTextAccepting(false);
    }

    if ((cancelRecording || publishCancellation) && cancellationGeneration != 0) {
        try {
            const auto [ok, detail] = committer_.cancelVoiceSession(cancellationGeneration);
            if (!ok && detail != "preedit unsupported") {
                log("voice session cancel failed: " + detail);
            }
        } catch (const std::exception& e) {
            log(std::string("voice session cancel failed: ") + e.what());
        }
    }

    if (publishCancellation) {
        const auto finishNotification = [this] {
            {
                std::lock_guard<std::mutex> lock(mutex_);
                --pendingCancellationNotifications_;
            }
            condition_.notify_all();
        };
        try {
            publishPhase(SessionState::Cancelling, notificationVersion);
        } catch (...) {
            finishNotification();
            throw;
        }
        finishNotification();
    }

    if (cancelRecording) {
        if (livePipeline_) {
            try {
                livePipeline_->cancel();
            } catch (const std::exception& e) {
                log(std::string("voice cancel failed: ") + e.what());
            }
        } else {
            try {
                recorder_->stop();
            } catch (const std::exception& e) {
                log(std::string("voice cancel failed: ") + e.what());
            }
        }
    }
    if (hasNotification) {
        publishPhase(notification, notificationVersion);
    }
    return result;
}

std::uint64_t VoiceSession::setStateLocked(SessionState state)
{
    if (state_ != state) {
        state_ = state;
        ++stateVersion_;
    }
    return stateVersion_;
}

std::pair<SessionState, std::uint64_t> VoiceSession::phaseSnapshot() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return { state_, stateVersion_ };
}

void VoiceSession::publishPhase(
    SessionState state, std::uint64_t version, bool suppressIfStale)
{
    bool drain = false;
    {
        std::lock_guard<std::mutex> lock(phaseNotificationMutex_);
        phaseNotifications_.push_back(
            PhaseNotification { state, version, suppressIfStale });
        if (!phaseNotificationDraining_) {
            phaseNotificationDraining_ = true;
            drain = true;
        }
    }

    if (drain) {
        while (true) {
            PhaseNotification next;
            {
                std::lock_guard<std::mutex> lock(phaseNotificationMutex_);
                if (phaseNotifications_.empty()) {
                    phaseNotificationDraining_ = false;
                    break;
                }
                next = phaseNotifications_.front();
                phaseNotifications_.pop_front();
            }

            bool publish = true;
            if (next.suppressIfStale) {
                std::lock_guard<std::mutex> lock(mutex_);
                // A delayed STATUS snapshot must not overwrite a newer phase.
                publish = state_ == next.state && stateVersion_ == next.version;
            }

            if (publish) {
                try {
                    ui_.send(phaseName(next.state));
                } catch (const std::exception& e) {
                    log(std::string("voice phase notification failed: ") + e.what());
                } catch (...) {
                    log("voice phase notification failed: unknown exception");
                }
            }
        }
    }
}

std::string VoiceSession::phaseName(SessionState state)
{
    switch (state) {
    case SessionState::Idle:
        return "IDLE";
    case SessionState::Recording:
        return "RECORDING";
    case SessionState::Transcribing:
        return "TRANSCRIBING";
    case SessionState::Cancelling:
        return "CANCELLING";
    }
    return "IDLE";
}

} // namespace echoflow
