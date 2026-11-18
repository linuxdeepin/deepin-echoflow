// SPDX-FileCopyrightText: 2026 Hualet Wang
// SPDX-License-Identifier: GPL-3.0-or-later

#include "TranscriptionQualityMetrics.h"

#include "Config.h"
#include "CrispAsrEngine.h"
#include "CrispSession.h"
#include "LiveStreamTranscriber.h"
#include "LocalLlmPolisher.h"
#include "PolishPrompt.h"
#include "StreamingDenoiser.h"

#include <QCommandLineParser>
#include <QCoreApplication>
#include <QCryptographicHash>
#include <QFile>
#include <QFileInfo>
#include <QJsonDocument>
#include <QJsonObject>
#include <QSaveFile>
#include <QTextStream>

#include <algorithm>
#include <chrono>
#include <optional>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;
using echoflow::benchmark::normalized;
using echoflow::benchmark::scored;

double elapsedMs(Clock::time_point started)
{
    return std::chrono::duration<double, std::milli>(Clock::now() - started).count();
}

QString sha256(const QString& path)
{
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly)) throw std::runtime_error("cannot open input for hashing");
    QCryptographicHash hash(QCryptographicHash::Sha256);
    if (!hash.addData(&file)) throw std::runtime_error("cannot hash input");
    return QString::fromLatin1(hash.result().toHex());
}


QJsonObject replay(echoflow::CrispSession& session, const echoflow::Config& cfg,
                   const std::vector<int16_t>& audio, bool fullSegment)
{
    auto config = echoflow::makeLiveStreamConfig(cfg);
    if (!fullSegment) {
        config.maxSegmentMs = std::max({3000, cfg.streamStepMs + 200,
                                       config.minSpeechMs + config.prePaddingMs});
    }
    bool deferred = false;
    bool quiet = false;
    qint64 calls = 0;
    int updates = 0;
    double flushMs = 0;
    double pauseMs = 0;
    echoflow::LiveStreamDecoder decoder;
    decoder.open = [&] {
        if (!session.openStream(cfg.streamStepMs, cfg.streamStepMs, 800)) {
            throw std::runtime_error("cannot open Qwen stream");
        }
        session.setStreamFullSegmentDecode(fullSegment);
        deferred = false;
    };
    decoder.feed = [&](const int16_t* data, size_t count) -> std::optional<std::string> {
        std::vector<float> pcm(count);
        for (size_t i = 0; i < count; ++i) pcm[i] = data[i] / 32768.0f;
        session.setStreamLiveDecode(!deferred && !quiet);
        const auto before = session.streamInferenceCount();
        const bool changed = session.feedStream(pcm.data(), static_cast<int>(count));
        if (changed || session.streamInferenceCount() > before) return session.streamText();
        return std::nullopt;
    };
    decoder.flush = [&] {
        const auto started = Clock::now();
        auto text = session.flushStream();
        flushMs += elapsedMs(started);
        return text;
    };
    decoder.close = [&] {
        calls += session.streamInferenceCount();
        session.closeStream();
    };
    decoder.deferPreview = [&] { deferred = true; };
    decoder.setSilence = [&](bool value) { quiet = value; };
    if (fullSegment) {
        decoder.checkpoint = [&] {
            const auto started = Clock::now();
            auto text = session.refreshStream();
            pauseMs += elapsedMs(started);
            return text;
        };
    }
    echoflow::LiveStreamTranscriber stream(config, decoder,
        [&](const std::string&) { ++updates; }, {},
        echoflow::makeLiveStreamEndpoint(cfg, config.maxSegmentMs));
    const auto started = Clock::now();
    stream.append(audio.data(), audio.size());
    const auto text = stream.finish();
    return {{"transcript", QString::fromStdString(text)}, {"elapsed_ms", elapsedMs(started)},
        {"final_decode_total_ms", flushMs}, {"pause_decode_total_ms", pauseMs}, {"inference_calls", calls}, {"updates", updates},
        {"confirmed_segments", static_cast<qint64>(stream.confirmedSegments())},
        {"max_segment_ms", config.maxSegmentMs}, {"short_pause_ms", config.shortPauseMs}, {"full_segment_correction", fullSegment}};
}

QJsonObject polishDiagnostic(const std::string& raw, const QString& model, int threads)
{
    echoflow::LocalLlmPolisher polisher(model.toStdString(), threads);
    const auto started = Clock::now();
    std::string text;
    int fallbacks = 0;
    for (const auto& piece : echoflow::splitTranscriptForPolish(raw, echoflow::kPolishChunkMaxChars)) {
        const auto first = piece.find_first_not_of(" \t\r\n");
        const auto last = piece.find_last_not_of(" \t\r\n");
        std::string result = piece;
        if (first != std::string::npos) {
            const auto deadline = Clock::now() + std::chrono::seconds(5);
            const auto abort = [&] { return Clock::now() >= deadline; };
            try {
                const auto polished = polisher.polish(piece.substr(first, last - first + 1), abort);
                if (!polished.empty() && !abort()) {
                    result = piece.substr(0, first) + polished + piece.substr(last + 1);
                } else {
                    ++fallbacks;
                }
            } catch (const std::exception&) {
                ++fallbacks;
            }
        }
        text += result;
    }
    return {{"transcript", QString::fromStdString(text)}, {"elapsed_ms", elapsedMs(started)},
        {"fallback_pieces", fallbacks}, {"scope", "offline_chunked_polish_diagnostic"}};
}
}

int main(int argc, char** argv)
{
    QCoreApplication app(argc, argv);
    QCommandLineParser parser;
    parser.setApplicationDescription("Compare identical local audio through whole-file ASR, prefix streaming, "
                                     "and full-segment correction. Replay is unpaced; no microphone or Fcitx.");
    parser.addHelpOption();
    parser.addOptions({
        {{"m", "model"}, "ASR GGUF; overrides the config model.", "path"},
        {{"c", "config"}, "Optional EchoFlow config (polish is only enabled by --polish-model).", "path"},
        {{"w", "wav"}, "Local speech WAV.", "path"},
        {{"l", "language"}, "Recognition language, e.g. Chinese or Korean.", "name"},
        {{"t", "threads"}, "Inference threads.", "count", "2"},
        {"step-ms", "Preview interval in milliseconds.", "count"},
        {"reference", "Optional reference transcript for CER.", "text"},
        {"reference-file", "UTF-8 reference transcript (alternative to --reference).", "path"},
        {"polish-model", "Optional local polish GGUF for a separate diagnostic.", "path"},
        {{"o", "output"}, "Write JSON report atomically; otherwise stdout.", "path"}});
    parser.process(app);
    try {
        bool valid = false;
        const int threads = parser.value("threads").toInt(&valid);
        if (!valid || threads < 1 || parser.value("wav").isEmpty()) {
            throw std::runtime_error("specify --wav and positive --threads");
        }
        if (parser.isSet("reference") && parser.isSet("reference-file")) {
            throw std::runtime_error("choose --reference or --reference-file");
        }
        QString reference = parser.value("reference");
        if (parser.isSet("reference-file")) {
            QFile input(parser.value("reference-file"));
            if (!input.open(QIODevice::ReadOnly)) throw std::runtime_error("cannot open reference");
            reference = QString::fromUtf8(input.readAll()).trimmed();
        }
        if ((parser.isSet("reference") || parser.isSet("reference-file")) && normalized(reference).isEmpty()) {
            throw std::runtime_error("reference contains no scored characters");
        }
        auto cfg = parser.isSet("config") ? echoflow::loadDtkConf(parser.value("config").toStdString())
                                         : echoflow::Config::defaultConfig();
        if (parser.isSet("model")) cfg.crispModelPath = parser.value("model").toStdString();
        if (parser.isSet("language")) cfg.language = parser.value("language").toStdString();
        if (parser.isSet("step-ms")) {
            cfg.streamStepMs = parser.value("step-ms").toInt(&valid);
            if (!valid) throw std::runtime_error("invalid preview interval");
        }
        echoflow::makeLiveStreamConfig(cfg);
        if (!QFileInfo::exists(QString::fromStdString(cfg.crispModelPath))) {
            throw std::runtime_error("specify an existing ASR model with --model or --config");
        }
        if (parser.isSet("polish-model") && !QFileInfo::exists(parser.value("polish-model"))) {
            throw std::runtime_error("polish model does not exist");
        }
        qputenv("OPENBLAS_NUM_THREADS", QByteArray::number(threads));
        auto pcm = echoflow::CrispSession::readWavF32(parser.value("wav").toStdString());
        if (pcm.empty()) throw std::runtime_error("empty or unsupported WAV");
        if (cfg.noiseSuppression) {
            echoflow::StreamingDenoiser denoiser;
            auto clean = denoiser.append(pcm.data(), pcm.size());
            const auto tail = denoiser.finish();
            clean.insert(clean.end(), tail.begin(), tail.end());
            pcm = std::move(clean);
        }
        std::vector<int16_t> audio(pcm.size());
        for (size_t i = 0; i < pcm.size(); ++i) {
            audio[i] = static_cast<int16_t>(std::lround(std::clamp(pcm[i] * 32768.0f, -32768.0f, 32767.0f)));
            pcm[i] = audio[i] / 32768.0f;
        }
        echoflow::CrispSession session(cfg.crispModelPath, cfg.crispBackend, threads);
        if (!session.isLoaded() || session.streamKind() != 3) throw std::runtime_error("requires a Qwen prefix-stream model");
        const auto language = echoflow::CrispAsrEngine::languageCode(cfg.language.value_or(""));
        session.setLanguage(language);
        if (cfg.crispMaxNewTokens > 0) session.setMaxNewTokens(cfg.crispMaxNewTokens);
        const auto started = Clock::now();
        const auto whole = session.transcribe(pcm.data(), static_cast<int>(pcm.size()));
        QJsonObject variants;
        variants.insert("whole_raw", scored({{"transcript", QString::fromStdString(whole)},
                                              {"elapsed_ms", elapsedMs(started)}}, reference));
        variants.insert("prefix_stream_raw", scored(replay(session, cfg, audio, false), reference));
        const auto corrected = replay(session, cfg, audio, true);
        variants.insert("stream_raw", scored(corrected, reference));
        if (parser.isSet("polish-model")) {
            variants.insert("stream_polished_diagnostic", scored(polishDiagnostic(
                corrected.value("transcript").toString().toStdString(), parser.value("polish-model"), threads), reference));
        }
        const QJsonObject report{{"schema_version", 1}, {"scope", "unpaced_quality_replay_not_latency_ab"},
            {"model", QString::fromStdString(cfg.crispModelPath)},
            {"model_sha256", sha256(QString::fromStdString(cfg.crispModelPath))},
            {"wav_sha256", sha256(parser.value("wav"))}, {"language", QString::fromStdString(language)},
            {"audio_seconds", static_cast<double>(audio.size()) / 16000}, {"threads", threads},
            {"step_ms", cfg.streamStepMs}, {"end_silence_ms", cfg.vadMinSilenceMs},
            {"noise_suppression", cfg.noiseSuppression}, {"reference", reference},
            {"cer_normalization", "NFKC, casefold, remove Unicode punctuation and whitespace"}, {"variants", variants}};
        const auto json = QJsonDocument(report).toJson(QJsonDocument::Indented);
        if (parser.isSet("output")) {
            QSaveFile output(parser.value("output"));
            if (!output.open(QIODevice::WriteOnly) || output.write(json) != json.size() || !output.commit()) {
                throw std::runtime_error("cannot write quality report");
            }
        } else {
            QTextStream(stdout) << json;
        }
        return 0;
    } catch (const std::exception& error) {
        QTextStream(stderr) << error.what() << '\n';
        return 1;
    }
}
