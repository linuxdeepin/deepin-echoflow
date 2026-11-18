// SPDX-FileCopyrightText: 2026 Hualet Wang
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef ECHOFLOW_VOICE_SESSION_H
#define ECHOFLOW_VOICE_SESSION_H

#include "Config.h"
#include "Interfaces.h"

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>

namespace echoflow {

class StreamingTextPolisher;

enum class SessionState {
    Idle,
    Recording,
    Transcribing,
    Cancelling,
};

class VoiceSession {
public:
    using ThreadLauncher = std::function<std::thread(std::function<void()>)>;

    VoiceSession(Config cfg, IRecorder& recorder, IAsrEngine& asr,
        ICommitter& committer, IUiNotifier& ui, ThreadLauncher threadLauncher = {});
    VoiceSession(Config cfg, ILiveVoicePipeline& livePipeline,
        ICommitter& committer, IUiNotifier& ui, ThreadLauncher threadLauncher = {});
    ~VoiceSession();

    // Optional; must be set before the first command and outlive this session.
    // Confirmed segments are polished in the background; pending work has a
    // bounded wait at finish, then falls back to its raw text.
    void setTextPolisher(ITextPolisher* polisher);

    std::string handleCommand(const std::string& command);
    SessionState state() const;

private:
    struct PartialCallbackState {
        std::mutex mutex;
        std::mutex deliveryMutex;
        std::condition_variable condition;
        IUiNotifier* ui = nullptr;
        ICommitter* committer = nullptr;
        std::uint64_t generation = 0;
        std::uint64_t revision = 0;
        std::uint64_t polishSession = 0;
        std::string rawText;
        std::string confirmedRaw;
        std::string polishedConfirmed;
        std::string pendingConfirmedRaw;
        std::string pendingPolished;
        std::string lastPublished;
        bool published = false;
        bool accepting = false;
        bool preeditAccepting = false;
        std::size_t active = 0;
    };

    struct AudioLevelCallbackState {
        std::mutex mutex;
        std::condition_variable condition;
        IUiNotifier* ui = nullptr;
        bool accepting = false;
        std::size_t active = 0;
    };

    struct PhaseNotification {
        SessionState state;
        std::uint64_t version;
        bool suppressIfStale;
    };

    void setupAudioLevelCallback();
    void setAudioLevelAccepting(bool accepting);
    void shutdownAudioLevelCallback();
    static void publishPartial(const std::shared_ptr<PartialCallbackState>& state);
    std::string startRecording();
    std::string startFinalization();
    void finalize(std::uint64_t generation);
    std::string polishText(const std::string& text);
    void joinCompletedWorker();
    void setPartialTextAccepting(bool accepting, bool keepPreedit = false);
    void shutdownPartialTextCallback();
    // Discard any in-flight recording and return to Idle, emitting IDLE so the
    // UI hides the capsule. Shared by BLUR (focus lost) and CANCEL (X button).
    std::string cancelActiveSession();
    std::uint64_t setStateLocked(SessionState state);
    std::pair<SessionState, std::uint64_t> phaseSnapshot() const;
    void publishPhase(
        SessionState state, std::uint64_t version, bool suppressIfStale = false);
    static std::string phaseName(SessionState state);

    Config cfg_;
    IRecorder* recorder_ = nullptr;
    IAsrEngine* asr_ = nullptr;
    ILiveVoicePipeline* livePipeline_ = nullptr;
    std::shared_ptr<StreamingTextPolisher> streamingPolisher_;
    ICommitter& committer_;
    IUiNotifier& ui_;
    ThreadLauncher threadLauncher_;
    std::shared_ptr<PartialCallbackState> partialCallbackState_;
    std::shared_ptr<AudioLevelCallbackState> audioLevelCallbackState_;

    mutable std::mutex mutex_;
    std::condition_variable condition_;
    SessionState state_ = SessionState::Idle;
    std::uint64_t stateVersion_ = 0;
    std::thread finalizationWorker_;
    std::uint64_t nextGeneration_ = 0;
    std::uint64_t runningGeneration_ = 0;
    bool finalizationRunning_ = false;
    bool finalizationStartPublished_ = false;
    bool commitClaimed_ = false;
    std::size_t pendingCancellationNotifications_ = 0;
    bool cancelRequested_ = false;
    bool shuttingDown_ = false;

    std::mutex phaseNotificationMutex_;
    std::deque<PhaseNotification> phaseNotifications_;
    bool phaseNotificationDraining_ = false;
};

} // namespace echoflow

#endif
