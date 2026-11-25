// SPDX-FileCopyrightText: 2026 Hualet Wang
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef ECHOFLOW_CRISP_LIVE_VOICE_PIPELINE_H
#define ECHOFLOW_CRISP_LIVE_VOICE_PIPELINE_H

#include "Config.h"
#include "Interfaces.h"
#include "LiveSegmentCoordinator.h"
#include "LiveAudioQueue.h"
#include "LivePreviewScheduler.h"

#include <atomic>
#include <chrono>
#include <functional>
#include <condition_variable>
#include <deque>
#include <exception>
#include <memory>
#include <mutex>
#include <string>
#include <sys/types.h>
#include <thread>
#include <cstdint>

namespace echoflow {

class CrispSession;
class LiveStreamTranscriber;
class LiveDebugRecorder;
class StreamingDenoiser;

class CrispLiveVoicePipeline : public ILiveVoicePipeline {
public:
    explicit CrispLiveVoicePipeline(Config cfg);
    ~CrispLiveVoicePipeline() override;

    CrispLiveVoicePipeline(const CrispLiveVoicePipeline&) = delete;
    CrispLiveVoicePipeline& operator=(const CrispLiveVoicePipeline&) = delete;

    void setRecognitionBusyCallback(std::function<void(bool)> callback) override;
    void start() override;
    void beginFinalization() noexcept override;
    std::string finish() override;
    void cancel() override;
    void setPartialTextCallback(std::function<void(const std::string&)> callback) override;
    void setAudioLevelCallback(std::function<void(const AudioLevelFrame&)> callback) override;
    void setConfirmedTextCallback(std::function<void(const std::string&)> callback) override;

    // PCM replay uses the same recognition worker without a capture process.
    void startReplay();
    void appendReplayAudio(std::vector<int16_t> samples);

private:
    void startImpl(bool replay);
    void startRecording();
    void processCapturedAudio(std::vector<int16_t> samples);
    void readerLoop();
    void enqueueAudio(std::vector<int16_t> samples);
    void nativeStreamLoop();
    void stopNativeStreamInput(bool discard);
    void stopRecorder();
    void reapChild(pid_t& child);
    void emitAudioLevel(const AudioLevelFrame& level);
    void emitText(const std::string& text);
    void emitConfirmedText(const std::string& text);
    void cleanupAfterFinish();
    void emitRecognitionBusy(bool busy);

    Config cfg_;
    LivePreviewScheduler previewScheduler_;
    std::unique_ptr<CrispSession> session_;
    pid_t recorderChild_ = -1;
    int readFd_ = -1;
    std::thread readerThread_;
    std::exception_ptr readerError_;
    std::mutex readerErrorMutex_;
    std::function<void(const AudioLevelFrame&)> audioLevelCallback_;
    std::function<void(const std::string&)> partialTextCallback_;
    std::function<void(const std::string&)> confirmedTextCallback_;
    mutable std::mutex callbackMutex_;
    std::function<void(bool)> recognitionBusyCallback_;
    std::atomic<bool> recognitionBusy_{false};
    std::mutex recognitionStateMutex_;

    std::unique_ptr<LiveSegmentCoordinator> coordinator_;
    std::unique_ptr<LiveDebugRecorder> debugRecorder_;
    std::unique_ptr<StreamingDenoiser> denoiser_;
    std::unique_ptr<LiveStreamTranscriber> streamTranscriber_;
    bool nativeStreaming_ = false;
    bool replaying_ = false;
    uint64_t inputSamples_ = 0;
    std::mutex streamQueueMutex_;
    std::condition_variable streamQueueReady_;
    LiveAudioQueue streamAudioQueue_;
    bool streamPreviewEnabled_ = true;
    bool streamBacklog_ = false;
    bool segmentFinalizing_ = false;
    bool streamSilence_ = false;
    std::atomic<bool> sessionFailed_{false};
    std::atomic<bool> finishing_{false};
    size_t queueHighWaterSamples_ = 0;
    long long queueMaxWaitMs_ = 0;
    bool streamInputClosed_ = false;
    std::thread streamThread_;

    std::atomic<bool> active_{false};
    std::atomic<bool> cancelled_{false};
    std::chrono::steady_clock::time_point startedAt_{};
};

}  // namespace echoflow

#endif
