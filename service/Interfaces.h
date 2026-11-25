// SPDX-FileCopyrightText: 2026 Hualet Wang
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef ECHOFLOW_INTERFACES_H
#define ECHOFLOW_INTERFACES_H

#include "AudioLevelFrame.h"

#include <cstdint>
#include <filesystem>
#include <functional>
#include <string>
#include <utility>

namespace echoflow {

class IRecorder {
public:
    virtual ~IRecorder() = default;
    virtual void start() = 0;
    virtual std::filesystem::path stop() = 0;
    virtual void setAudioLevelCallback(std::function<void(const AudioLevelFrame&)>) {}
};

class IAsrEngine {
public:
    virtual ~IAsrEngine() = default;
    virtual std::string transcribe(const std::filesystem::path& audio) = 0;
};

class ILiveVoicePipeline {
public:
    virtual ~ILiveVoicePipeline() = default;
    virtual void start() = 0;
    // Called on STOP before phase notification. Do not block or invoke callbacks.
    virtual void beginFinalization() noexcept {}
    virtual std::string finish() = 0;
    virtual void cancel() = 0;
    virtual void setPartialTextCallback(std::function<void(const std::string&)> callback) = 0;
    virtual void setAudioLevelCallback(std::function<void(const AudioLevelFrame&)>) {}
    // Recognition work has priority over optional background polishing.
    virtual void setRecognitionBusyCallback(std::function<void(bool)>) {}
    // Cumulative confirmed raw text. Unsupported pipelines can leave this unset.
    virtual void setConfirmedTextCallback(std::function<void(const std::string&)>) {}
};

class ICommitter {
public:
    virtual ~ICommitter() = default;
    virtual std::pair<bool, std::string> commitText(const std::string& text) = 0;
    virtual std::pair<bool, std::string> beginVoiceSession(std::uint64_t)
    {
        return {true, "OK"};
    }
    virtual std::pair<bool, std::string> cancelVoiceSession(std::uint64_t generation)
    {
        return clearPreedit(generation);
    }
    virtual std::pair<bool, std::string> commitText(
        const std::string& text, std::uint64_t)
    {
        return commitText(text);
    }
    virtual std::pair<bool, std::string> setPreedit(const std::string& text)
    {
        return {false, "preedit unsupported"};
    }
    virtual std::pair<bool, std::string> setPreedit(
        const std::string& text, std::uint64_t)
    {
        return setPreedit(text);
    }
    virtual std::pair<bool, std::string> setPreedit(
        const std::string& text, std::uint64_t generation, std::uint64_t)
    {
        return setPreedit(text, generation);
    }
    virtual std::pair<bool, std::string> clearPreedit()
    {
        return {false, "preedit unsupported"};
    }
    virtual std::pair<bool, std::string> clearPreedit(std::uint64_t)
    {
        return clearPreedit();
    }
};

class ITextPolisher {
public:
    virtual ~ITextPolisher() = default;
    // Returns the rewritten text, or throws on failure. shouldAbort is polled
    // during generation so CANCEL does not wait for the model to finish.
    virtual std::string polish(const std::string& text,
        const std::function<bool()>& shouldAbort) = 0;
};

class IUiNotifier {
public:
    virtual ~IUiNotifier() = default;
    virtual void send(const std::string& message) = 0;
};

}  // namespace echoflow

#endif
