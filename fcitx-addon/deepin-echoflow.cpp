// SPDX-FileCopyrightText: 2026 Hualet Wang
// SPDX-License-Identifier: GPL-3.0-or-later

#include "CommandClient.h"
#include "VoiceInputPolicy.h"
#include "VoicePreedit.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <vector>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <unistd.h>

#include <fcitx-utils/capabilityflags.h>
#include <fcitx-utils/event.h>
#include <fcitx-utils/eventloopinterface.h>
#include <fcitx-utils/handlertable.h>
#include <fcitx-utils/key.h>
#include <fcitx-utils/log.h>
#include <fcitx-utils/rect.h>
#include <fcitx-utils/utf8.h>
#include <fcitx/action.h>
#include <fcitx/addonfactory.h>
#include <fcitx/addonmanager.h>
#include <fcitx/inputcontext.h>
#include <fcitx/inputcontextmanager.h>
#include <fcitx/instance.h>
#include <fcitx/userinterfacemanager.h>

namespace {

constexpr const char *kCommitSocketName = "deepin-echoflow-fcitx.sock";
constexpr size_t kMaxMessageSize = 65536;
// Chromium blurs and refocuses its input context several times right after a
// click; focus reports within this window of a program's first focus still
// count as that first focus, so the one-per-program hint is not lost to churn.
constexpr int64_t kFocusChurnWindowMs = 500;
// A FocusOut whose FocusIn lands on the same program within this window is
// the same churn, not a real switch: the BLUR that would discard a live
// recording or a finished transcript is dropped. A different program taking
// focus fires the BLUR at once; focus going nowhere fires it on timeout.
constexpr int64_t kBlurDebounceMs = 500;

bool isUsableRuntimeDirectory(const std::string &path) {
    struct stat st {};
    return stat(path.c_str(), &st) == 0 && S_ISDIR(st.st_mode) &&
           access(path.c_str(), W_OK | X_OK) == 0;
}

std::string runtimeDir() {
    std::string runUserDir = "/run/user/" + std::to_string(getuid());
    if (isUsableRuntimeDirectory(runUserDir)) {
        return runUserDir;
    }
    if (const char *runtime = std::getenv("XDG_RUNTIME_DIR")) {
        if (runtime[0] != '\0' && isUsableRuntimeDirectory(runtime)) {
            return runtime;
        }
    }
    return "/tmp";
}

std::string commitSocketPath() { return runtimeDir() + "/" + kCommitSocketName; }

bool sendControlCommand(std::string_view command, bool wakeup = false) {
    return echoflow::sendAddonCommand(runtimeDir(), wakeup
        ? echoflow::CommandChannel::Wakeup : echoflow::CommandChannel::Control, command);
}

bool sendFocusCommand(std::string_view command) {
    return echoflow::sendAddonCommand(runtimeDir(), echoflow::CommandChannel::Ui, command);
}

struct Request {
    enum class Type { Commit, Preedit, ClearPreedit, BeginVoice, CancelVoice, Ping };
    Type type = Type::Commit;
    std::uint64_t generation = 0;
    std::uint64_t revision = 0;
    std::string text;
};

bool parseGeneration(const std::string& value, std::uint64_t* generation)
{
    if (value.empty()) return false;
    try {
        size_t consumed = 0;
        const auto parsed = std::stoull(value, &consumed);
        if (consumed != value.size() || parsed == 0) return false;
        *generation = parsed;
        return true;
    } catch (...) {
        return false;
    }
}

Request parseRequest(const std::string &message) {
    if (message.empty() || message == "PING" || message == "PING\n") {
        return {Request::Type::Ping, 0, 0, {}};
    }
    const auto newline = message.find('\n');
    const std::string header = message.substr(0, newline);
    const std::string payload = newline == std::string::npos
        ? std::string() : message.substr(newline + 1);
    const auto space = header.find(' ');
    const std::string verb = header.substr(0, space);
    if (verb == "COMMIT" || verb == "PREEDIT" || verb == "CLEAR_PREEDIT"
        || verb == "VOICE_BEGIN" || verb == "VOICE_CANCEL") {
        Request request;
        request.type = verb == "COMMIT" ? Request::Type::Commit
            : verb == "PREEDIT" ? Request::Type::Preedit
            : verb == "CLEAR_PREEDIT" ? Request::Type::ClearPreedit
            : verb == "VOICE_BEGIN" ? Request::Type::BeginVoice
            : Request::Type::CancelVoice;
        if (space != std::string::npos) {
            const std::string arguments = header.substr(space + 1);
            const auto secondSpace = arguments.find(' ');
            const std::string generation = arguments.substr(0, secondSpace);
            if (!parseGeneration(generation, &request.generation)) return {};
            if (secondSpace != std::string::npos) {
                if (verb != "PREEDIT" || !parseGeneration(
                        arguments.substr(secondSpace + 1), &request.revision)) return {};
            }
        }
        if (verb == "VOICE_BEGIN" || verb == "VOICE_CANCEL") {
            if (space == std::string::npos || newline == std::string::npos) return {};
        }
        request.text = payload;
        return request;
    }
    return {Request::Type::Commit, 0, 0, message};
}

class EchoFlow final : public fcitx::AddonInstance {
public:
    explicit EchoFlow(fcitx::AddonManager *manager)
        : instance_(manager->instance()) {
        blurTimer_ = instance_->eventLoop().addTimeEvent(
            CLOCK_MONOTONIC, fcitx::now(CLOCK_MONOTONIC) / 1000, 1,
            [this](fcitx::EventSourceTime *, uint64_t) {
                firePendingBlur();
                return true;
            });
        blurTimer_->setEnabled(false);
        setupCommitSocket();
        registerEventWatchers();
    }

    ~EchoFlow() override {
        ioEvent_.reset();
        clearVoicePreedit();
        unlinkOwnedCommitSocket();
        if (fd_ >= 0) {
            close(fd_);
            fd_ = -1;
        }
    }

private:
    void unlinkOwnedCommitSocket() {
        if (!socketBound_ || fd_ < 0) {
            return;
        }
        struct stat fdStat {};
        struct stat pathStat {};
        if (fstat(fd_, &fdStat) != 0 ||
            stat(commitSocketPath_.c_str(), &pathStat) != 0) {
            return;
        }
        if (fdStat.st_dev == pathStat.st_dev &&
            fdStat.st_ino == pathStat.st_ino) {
            unlink(commitSocketPath_.c_str());
        }
    }

    void setupCommitSocket() {
        commitSocketPath_ = commitSocketPath();
        sockaddr_un addr {};
        if (commitSocketPath_.size() >= sizeof(addr.sun_path)) {
            FCITX_ERROR() << "deepin-echoflow socket path too long: "
                          << commitSocketPath_;
            return;
        }

        fd_ = socket(AF_UNIX, SOCK_DGRAM | SOCK_NONBLOCK | SOCK_CLOEXEC, 0);
        if (fd_ < 0) {
            FCITX_ERROR() << "deepin-echoflow failed to create socket: "
                          << strerror(errno);
            return;
        }

        addr.sun_family = AF_UNIX;
        std::strncpy(addr.sun_path, commitSocketPath_.c_str(),
                     sizeof(addr.sun_path) - 1);
        unlink(commitSocketPath_.c_str());
        if (bind(fd_, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) < 0) {
            FCITX_ERROR() << "deepin-echoflow failed to bind " << commitSocketPath_
                          << ": " << strerror(errno);
            close(fd_);
            fd_ = -1;
            return;
        }
        socketBound_ = true;
        chmod(commitSocketPath_.c_str(), S_IRUSR | S_IWUSR);

        ioEvent_ = instance_->eventLoop().addIOEvent(
            fd_, fcitx::IOEventFlag::In,
            [this](fcitx::EventSourceIO *, int fd, fcitx::IOEventFlags flags) {
                return handleReadable(fd, flags);
            });
        FCITX_INFO() << "deepin-echoflow commit socket listening on "
                     << commitSocketPath_;
    }

    void registerEventWatchers() {
        eventWatchers_.push_back(instance_->watchEvent(
            fcitx::EventType::InputContextFocusIn,
            fcitx::EventWatcherPhase::Default,
            [this](fcitx::Event &event) { handleFocusEvent(event, true); }));
        eventWatchers_.push_back(instance_->watchEvent(
            fcitx::EventType::InputContextFocusOut,
            fcitx::EventWatcherPhase::Default,
            [this](fcitx::Event &event) { handleFocusEvent(event, false); }));
        eventWatchers_.push_back(instance_->watchEvent(
            fcitx::EventType::InputContextCreated,
            fcitx::EventWatcherPhase::Default,
            [this](fcitx::Event &event) {
                handleInputContextLifetime(event, true);
            }));
        eventWatchers_.push_back(instance_->watchEvent(
            fcitx::EventType::InputContextDestroyed,
            fcitx::EventWatcherPhase::Default,
            [this](fcitx::Event &event) {
                handleInputContextLifetime(event, false);
            }));
        eventWatchers_.push_back(instance_->watchEvent(
            fcitx::EventType::InputContextCursorRectChanged,
            fcitx::EventWatcherPhase::Default,
            [this](fcitx::Event &event) { handleCursorRectChanged(event); }));
        eventWatchers_.push_back(instance_->watchEvent(
            fcitx::EventType::InputContextKeyEvent,
            fcitx::EventWatcherPhase::Default,
            [this](fcitx::Event &event) { handleKeyEvent(event); }));
    }

    void handleFocusEvent(fcitx::Event &event, bool focused) {
        auto *icEvent = static_cast<fcitx::InputContextEvent *>(&event);
        fcitx::InputContext *ic = icEvent->inputContext();
        if (focused && ic) {
            if (voicePreeditInputContext_ && voicePreeditInputContext_ != ic) {
                cancelBoundVoiceSession();
            }
            focusedInputContext_ = ic;
            if (blurPending_ && ic->program() != pendingBlurProgram_) {
                // Another program took the focus: a real switch, so the
                // deferred BLUR fires now instead of after the timeout.
                firePendingBlur();
            } else {
                cancelPendingBlur();
            }
            sendGeometryCommand(focusVerb(ic->program()), ic);
            return;
        }
        focusedInputContext_ = nullptr;
        // BLUR is deferred, not sent on the spot: Chromium-family apps blur
        // and refocus around clicks and internal focus moves, and the
        // transient half of such a pair must not discard a live recording or
        // an already-finished transcript.
        blurPending_ = true;
        pendingBlurProgram_ = ic ? ic->program() : std::string();
        blurTimer_->setNextInterval(
            static_cast<uint64_t>(kBlurDebounceMs) * 1000);
        blurTimer_->setEnabled(true);
    }

    // A FocusOut fired the timer or hit a different program: send the BLUR
    // now. BLUR discards the recording, so a later right Ctrl release must
    // not send a STOP for it.
    void firePendingBlur() {
        if (!blurPending_) {
            return;
        }
        cancelPendingBlur();
        rightCtrlDown_ = false;
        cancelBoundVoiceSession();
        echoflow::sendAddonBlur(runtimeDir());
    }

    void cancelPendingBlur() {
        blurPending_ = false;
        pendingBlurProgram_.clear();
        blurTimer_->setEnabled(false);
    }

    // The focus hint shows once per program run: the first focus reports
    // FOCUS (hint-eligible) and later ones REFOCUS (geometry only). A
    // program counts as restarted once its last input context is gone, so
    // quitting and reopening allows one more hint.
    const char *focusVerb(const std::string &program) {
        const int64_t now = fcitx::now(CLOCK_MONOTONIC);
        const auto inserted = hintedPrograms_.try_emplace(program, now);
        if (inserted.second) {
            return "FOCUS";
        }
        if (now - inserted.first->second <= kFocusChurnWindowMs * 1000 * 1000) {
            return "FOCUS";
        }
        return "REFOCUS";
    }

    void handleInputContextLifetime(fcitx::Event &event, bool created) {
        auto *icEvent = static_cast<fcitx::InputContextEvent *>(&event);
        fcitx::InputContext *ic = icEvent->inputContext();
        const std::string program = ic->program();
        if (!created && voicePreeditInputContext_ == ic) {
            voicePreeditInputContext_ = nullptr;
            voicePreedit_.reset();
            const auto generation = activeVoiceGeneration_ != 0
                ? activeVoiceGeneration_ : clearedVoiceGeneration_;
            clearedVoiceGeneration_ = generation;
            activeVoiceGeneration_ = 0;
            if (generation != 0) sendControlCommand("CANCEL", true);
        }
        if (created) {
            ++liveInputContexts_[program];
            return;
        }
        const auto count = liveInputContexts_.find(program);
        if (count == liveInputContexts_.end() || --count->second > 0) {
            return;
        }
        liveInputContexts_.erase(count);
        hintedPrograms_.erase(program);
    }

    void handleCursorRectChanged(fcitx::Event &event) {
        auto *icEvent = static_cast<fcitx::InputContextEvent *>(&event);
        if (focusedInputContext_ && icEvent->inputContext() == focusedInputContext_) {
            sendGeometryCommand("CARET", focusedInputContext_);
        }
    }

    // Caret geometry lets the UI anchor its hint and confirm long presses.
    // Password fields never expose it, so neither feature engages there.
    void sendGeometryCommand(const std::string &verb, fcitx::InputContext *ic) {
        const fcitx::Rect &rect = ic->cursorRect();
        if (rect.height() <= 0 ||
            ic->capabilityFlags().test(fcitx::CapabilityFlag::Password)) {
            sendFocusCommand(verb);
            return;
        }
        const std::string command =
            verb + " " + std::to_string(rect.left()) + " " +
            std::to_string(rect.top()) + " " +
            std::to_string(std::max(rect.width(), 0)) + " " +
            std::to_string(rect.height());
        sendFocusCommand(command);
    }

    void handleKeyEvent(fcitx::Event &event) {
        auto *keyEvent = static_cast<fcitx::KeyEvent *>(&event);
        const fcitx::Key key = keyEvent->key();

        // Right Ctrl is push-to-talk, matching the mouse long press: press
        // starts recording and release stops it for transcription.
        if (isRightCtrl(key)) {
            if (keyEvent->isRelease()) {
                if (rightCtrlDown_) {
                    rightCtrlDown_ = false;
                    sendControlCommand("STOP", true);
                }
                return;
            }
            if (rightCtrlDown_) {
                return;
            }
            rightCtrlDown_ = true;
            sendControlCommand("START", true);
            return;
        }

        // Any other non-modifier key press means the user is typing/editing in
        // the focused input context, so tell the service to declutter the voice
        // capsule. Observer only: we never consume the event (no accept/filter).
        if (!keyEvent->isRelease() && !key.isModifier()) {
            sendFocusCommand("TYPED");
            if (hasBoundVoiceSession()) {
                cancelBoundVoiceSession();
            }
        }
    }

    static bool isRightCtrl(const fcitx::Key &key) {
        return key.sym() == FcitxKey_Control_R;
    }

    bool handleReadable(int fd, fcitx::IOEventFlags flags) {
        if (flags.testAny(fcitx::IOEventFlag::Err) ||
            flags.testAny(fcitx::IOEventFlag::Hup)) {
            FCITX_WARN() << "deepin-echoflow socket event error/hangup";
        }

        while (true) {
            sockaddr_un peer {};
            socklen_t peerLen = sizeof(peer);
            std::array<char, kMaxMessageSize> buffer {};
            ssize_t len =
                recvfrom(fd, buffer.data(), buffer.size(), 0,
                         reinterpret_cast<sockaddr *>(&peer), &peerLen);
            if (len < 0) {
                if (errno == EAGAIN || errno == EWOULDBLOCK) {
                    return true;
                }
                FCITX_WARN() << "deepin-echoflow recvfrom failed: "
                             << strerror(errno);
                return true;
            }
            handleMessage(std::string(buffer.data(), static_cast<size_t>(len)),
                          peer, peerLen);
        }
    }

    void handleMessage(const std::string &text, const sockaddr_un &peer,
                       socklen_t peerLen) {
        Request request = parseRequest(text);
        if (request.type == Request::Type::Ping) {
            reply(peer, peerLen, "PONG");
            return;
        }
        if (request.type == Request::Type::BeginVoice) {
            if (!voiceGenerations_.canBegin(request.generation)) {
                reply(peer, peerLen, "ERR stale-generation");
                return;
            }
            auto *ic = focusedInputContext_;
            if (!ic || !ic->hasFocus() ||
                !echoflow::isVoiceInputAllowed(ic->capabilityFlags())) {
                reply(peer, peerLen, "ERR no-eligible-input-context");
                return;
            }
            // One service owns the control socket. A newer BEGIN can replace
            // a binding left behind when that service died without cancelling.
            clearVoicePreedit();
            voiceGenerations_.begin(request.generation);
            activeVoiceGeneration_ = request.generation;
            clearedVoiceGeneration_ = 0;
            lastPreeditRevision_ = 0;
            voiceCommitAllowed_ = false;
            voicePreeditInputContext_ = ic;
            voicePreedit_.begin(ic->inputPanel());
            reply(peer, peerLen, "OK");
            return;
        }
        if (request.type == Request::Type::CancelVoice) {
            const bool activeMatch = request.generation == activeVoiceGeneration_;
            const bool justFinalizedMatch = request.generation == clearedVoiceGeneration_;
            if (!activeMatch && !justFinalizedMatch) {
                reply(peer, peerLen, "ERR stale-generation");
                return;
            }
            clearVoicePreedit();
            activeVoiceGeneration_ = 0;
            clearedVoiceGeneration_ = request.generation;
            voiceCommitAllowed_ = false;
            voicePreeditInputContext_ = nullptr;
            reply(peer, peerLen, "OK");
            return;
        }
        if (request.type != Request::Type::ClearPreedit
            && request.type != Request::Type::Preedit && request.text.empty()) {
            reply(peer, peerLen, "ERR empty-text");
            return;
        }
        if (request.text.find('\0') != std::string::npos ||
            !fcitx::utf8::validate(request.text)) {
            reply(peer, peerLen, "ERR invalid-text");
            return;
        }

        if (request.type == Request::Type::ClearPreedit) {
            if (request.generation != activeVoiceGeneration_) {
                reply(peer, peerLen, "ERR stale-generation");
                return;
            }
            clearVoicePreedit(false);
            activeVoiceGeneration_ = 0;
            clearedVoiceGeneration_ = request.generation;
            voiceCommitAllowed_ = true;
            reply(peer, peerLen, "OK");
            return;
        }

        if (request.type == Request::Type::Preedit) {
            fcitx::InputContext *ic = voicePreeditInputContext_;
            if (request.generation == 0 || request.generation != activeVoiceGeneration_
                || request.generation <= clearedVoiceGeneration_ || request.revision == 0
                || request.revision <= lastPreeditRevision_ || !ic) {
                reply(peer, peerLen, "ERR stale-generation");
                return;
            }
            if (ic != focusedInputContext_ || !ic->hasFocus() ||
                !echoflow::isVoiceInputAllowed(ic->capabilityFlags())) {
                reply(peer, peerLen, "ERR input-context-changed");
                return;
            }
            const bool inlinePreedit = ic->isPreeditEnabled()
                && ic->capabilityFlags().test(fcitx::CapabilityFlag::Preedit);
            voicePreedit_.update(ic->inputPanel(), request.text, inlinePreedit);
            lastPreeditRevision_ = request.revision;
            ic->updatePreedit();
            ic->updateUserInterface(fcitx::UserInterfaceComponent::InputPanel);
            reply(peer, peerLen, "OK");
            return;
        }

        fcitx::InputContext *ic = voicePreeditInputContext_;
        if (request.generation != 0) {
            const bool activeMatch = request.generation == activeVoiceGeneration_;
            // Accept legacy CLEAR_PREEDIT -> COMMIT clients too. New clients
            // commit directly so composition and final text change in one event.
            const bool clearedMatch = activeVoiceGeneration_ == 0 && voiceCommitAllowed_
                && request.generation == clearedVoiceGeneration_;
            if ((!activeMatch && !clearedMatch) || !ic || !ic->hasFocus()
                || ic != focusedInputContext_
                || !echoflow::isVoiceInputAllowed(ic->capabilityFlags())) {
                reply(peer, peerLen, "ERR stale-generation");
                return;
            }
        } else if (!ic) {
            ic = focusedInputContext_ ? focusedInputContext_
                                      : instance_->lastFocusedInputContext();
        }
        if (!ic) {
            reply(peer, peerLen, "ERR no-focused-input-context");
            return;
        }

        clearVoicePreedit();
        if (request.generation != 0) {
            activeVoiceGeneration_ = 0;
            clearedVoiceGeneration_ = request.generation;
            voiceCommitAllowed_ = false;
        }
        voicePreeditInputContext_ = nullptr;
        if (ic->capabilityFlags() &
            fcitx::CapabilityFlag::CommitStringWithCursor) {
            ic->commitStringWithCursor(request.text,
                                       fcitx::utf8::length(request.text));
        } else {
            ic->commitString(request.text);
        }
        reply(peer, peerLen, "OK");
    }

    bool hasBoundVoiceSession() const {
        return activeVoiceGeneration_ != 0
            || (voicePreeditInputContext_ && clearedVoiceGeneration_ != 0);
    }

    void cancelBoundVoiceSession() {
        const auto generation = activeVoiceGeneration_ != 0
            ? activeVoiceGeneration_ : clearedVoiceGeneration_;
        clearVoicePreedit();
        activeVoiceGeneration_ = 0;
        clearedVoiceGeneration_ = generation;
        voiceCommitAllowed_ = false;
        voicePreeditInputContext_ = nullptr;
        if (generation != 0) {
            sendControlCommand("CANCEL", true);
        }
    }

    void clearVoicePreedit(bool releaseTarget = true) {
        if (!voicePreeditInputContext_) {
            return;
        }
        auto *ic = voicePreeditInputContext_;
        if (voicePreedit_.restore(ic->inputPanel())) {
            ic->updatePreedit();
            ic->updateUserInterface(fcitx::UserInterfaceComponent::InputPanel);
        }
        if (releaseTarget) {
            voicePreeditInputContext_ = nullptr;
        }
    }

    void reply(const sockaddr_un &peer, socklen_t peerLen,
               const char *message) {
        if (peerLen == 0 || peer.sun_path[0] == '\0') {
            return;
        }
        if (sendto(fd_, message, strlen(message), 0,
                   reinterpret_cast<const sockaddr *>(&peer), peerLen) < 0) {
            FCITX_WARN() << "deepin-echoflow reply failed: " << strerror(errno);
        }
    }

    fcitx::Instance *instance_ = nullptr;
    int fd_ = -1;
    bool socketBound_ = false;
    bool rightCtrlDown_ = false;
    std::string commitSocketPath_;
    // program name -> monotonic timestamp of its first FOCUS in this run.
    std::map<std::string, int64_t> hintedPrograms_;
    std::map<std::string, int> liveInputContexts_;
    std::unique_ptr<fcitx::EventSourceIO> ioEvent_;
    std::unique_ptr<fcitx::EventSourceTime> blurTimer_;
    // Whether a FocusOut's BLUR is still within the debounce window, and the
    // program that lost focus (empty when the event carried no input context).
    bool blurPending_ = false;
    std::string pendingBlurProgram_;
    std::vector<std::unique_ptr<fcitx::HandlerTableEntry<fcitx::EventHandler>>>
        eventWatchers_;
    fcitx::InputContext *focusedInputContext_ = nullptr;
    fcitx::InputContext *voicePreeditInputContext_ = nullptr;
    echoflow::VoiceGenerations voiceGenerations_;
    std::uint64_t activeVoiceGeneration_ = 0;
    std::uint64_t clearedVoiceGeneration_ = 0;
    bool voiceCommitAllowed_ = false;
    std::uint64_t lastPreeditRevision_ = 0;
    echoflow::VoicePreedit voicePreedit_;
};

class EchoFlowFactory final : public fcitx::AddonFactory {
public:
    fcitx::AddonInstance *create(fcitx::AddonManager *manager) override {
        return new EchoFlow(manager);
    }
};

} // namespace

FCITX_ADDON_FACTORY(EchoFlowFactory)
