// SPDX-FileCopyrightText: 2026 Hualet Wang
// SPDX-License-Identifier: GPL-3.0-or-later

#include "CrispAsrEngine.h"

#include "CrispSession.h"
#include "StreamingDenoiser.h"
#include "log.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>

namespace echoflow {

std::string CrispAsrEngine::languageCode(const std::string& value)
{
    if (value.empty()) {
        const char* locale = nullptr;
        for (const char* variable : {"LC_ALL", "LC_MESSAGES", "LANG"}) {
            const char* candidate = std::getenv(variable);
            if (candidate && candidate[0] != '\0') {
                locale = candidate;
                break;
            }
        }

        std::string language = locale ? locale : "";
        const auto separator = language.find_first_of("_.@-");
        if (separator != std::string::npos) {
            language.resize(separator);
        }
        std::transform(language.begin(), language.end(), language.begin(),
                       [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
        if (language == "zh") return "zh";
        if (language == "es") return "es";
        if (language == "pt") return "pt";
        if (language == "fr") return "fr";
        if (language == "it") return "it";
        return "en";
    }
    if (value == "Chinese" || value == "chinese" || value == "zh" || value == "cmn") {
        return "zh";
    }
    if (value == "English" || value == "english" || value == "en") {
        return "en";
    }
    if (value == "Spanish" || value == "spanish") {
        return "es";
    }
    if (value == "Portuguese" || value == "portuguese") {
        return "pt";
    }
    if (value == "French" || value == "french") {
        return "fr";
    }
    if (value == "Italian" || value == "italian") {
        return "it";
    }
    return value;
}

CrispAsrEngine::CrispAsrEngine(Config cfg)
    : cfg_(std::move(cfg))
{
    loadSessionIfAvailable();
}

void CrispAsrEngine::loadSessionIfAvailable()
{
    std::error_code modelError;
    if (!std::filesystem::exists(cfg_.crispModelPath, modelError) && !modelError) {
        return;
    }
    session_ = std::make_unique<CrispSession>(cfg_.crispModelPath, cfg_.crispBackend,
                                              cfg_.crispThreads);
    if (session_->isLoaded()) {
        session_->setHotwords(cfg_.hotwords);
        auto lang = languageCode(cfg_.language.value_or(""));
        log("file ASR configuration: language=" + lang + ", threads=" + std::to_string(cfg_.crispThreads)
            + ", noise_suppression=" + (cfg_.noiseSuppression ? "rnnoise" : "off"));
        if (!lang.empty()) {
            session_->setLanguage(lang);
        }
        if (cfg_.crispMaxNewTokens > 0) {
            session_->setMaxNewTokens(cfg_.crispMaxNewTokens);
        }
    }
}

CrispAsrEngine::~CrispAsrEngine() = default;

std::string CrispAsrEngine::transcribe(const std::filesystem::path& audio)
{
    if (!session_) {
        loadSessionIfAvailable();
    }
    if (!session_ || !session_->isLoaded()) {
        log("crisp session not loaded, cannot transcribe");
        return {};
    }

    auto pcm = CrispSession::readWavF32(audio.string());
    if (pcm.empty()) {
        log("failed to read audio for transcription: " + audio.string());
        return {};
    }

    if (cfg_.noiseSuppression) {
        StreamingDenoiser denoiser;
        auto cleaned = denoiser.append(pcm.data(), pcm.size());
        const auto tail = denoiser.finish();
        cleaned.insert(cleaned.end(), tail.begin(), tail.end());
        log("file denoise: samples=" + std::to_string(denoiser.inputSamples())
            + ", processing_ms=" + std::to_string(denoiser.processingMs()));
        pcm = std::move(cleaned);
    }
    std::string text = session_->transcribe(pcm.data(), static_cast<int>(pcm.size()));
    return text;
}

}  // namespace echoflow
