// SPDX-FileCopyrightText: 2026 Hualet Wang
// SPDX-License-Identifier: GPL-3.0-or-later

#include "CrispSession.h"
#include "core/qwen3_stream_model.h"

#include <QCommandLineParser>
#include <QCoreApplication>
#include <QCryptographicHash>
#include <QDateTime>
#include <QFile>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QSaveFile>
#include <QSysInfo>
#include <QTextStream>

#include <algorithm>
#include <array>
#include <memory>
#include <stdexcept>
#include <vector>

using core_qwen3_stream_model::DecodeMetrics;
using core_qwen3_stream_model::PrefixKvCache;

namespace {
QJsonObject metricsJson(const DecodeMetrics& metrics)
{
    return {{"total_ms", metrics.total_ms}, {"mel_ms", metrics.mel_ms},
        {"encoder_ms", metrics.encoder_ms}, {"prepare_ms", metrics.prepare_ms},
        {"prefill_ms", metrics.prefill_ms}, {"generate_ms", metrics.generate_ms},
        {"llm_graph_ms", metrics.llm_graph_ms}, {"llm_alloc_ms", metrics.llm_alloc_ms},
        {"llm_compute_ms", metrics.llm_compute_ms}, {"llm_transfer_ms", metrics.llm_transfer_ms},
        {"audio_frames", metrics.audio_frames}, {"prompt_tokens", metrics.prompt_tokens},
        {"generated_tokens", metrics.generated_tokens}, {"prefix_reused", metrics.prefix_reused},
        {"stem_reused", metrics.stem_reused}, {"stem_computed", metrics.stem_computed},
        {"encoder_regions_reused", metrics.encoder_regions_reused},
        {"encoder_regions_computed", metrics.encoder_regions_computed},
        {"encoder_window_chunks", metrics.encoder_window_chunks},
        {"encoder_cached_bytes", static_cast<qint64>(metrics.encoder_cached_bytes)}};
}

QString sha256(const QString& path)
{
    QFile input(path);
    if (!input.open(QIODevice::ReadOnly)) throw std::runtime_error("cannot hash input file");
    QCryptographicHash hash(QCryptographicHash::Sha256);
    if (!hash.addData(&input)) throw std::runtime_error("cannot read input file for hash");
    return QString::fromLatin1(hash.result().toHex());
}

QString cpuName()
{
    QFile input("/proc/cpuinfo");
    if (input.open(QIODevice::ReadOnly)) {
        const auto lines = input.readAll().split('\n');
        for (const auto& line : lines) {
            if (line.startsWith("model name")) return QString::fromUtf8(line.mid(line.indexOf(':') + 1).trimmed());
        }
    }
    return QSysInfo::currentCpuArchitecture();
}

double median(std::vector<double> values)
{
    std::sort(values.begin(), values.end());
    const size_t count = values.size();
    return count % 2 ? values[count / 2] : (values[count / 2 - 1] + values[count / 2]) / 2;
}
}

int main(int argc, char** argv)
{
    QCoreApplication app(argc, argv);
    QCommandLineParser parser;
    parser.setApplicationDescription("Compare per-segment and model-owned fixed prompt KV caches; no capture required.");
    parser.addHelpOption();
    parser.addOption({"encoder-cache", "Compare incremental encoder cache; both variants retain fixed prompt KV."});
    parser.addOptions({
        {{"m", "model"}, "Qwen ASR GGUF model.", "path"},
        {{"w", "wav"}, "Speech WAV fixture with at least three seconds of audio.", "path"},
        {{"t", "threads"}, "Inference threads.", "count", "2"},
        {{"r", "repeats"}, "Measured repetitions per duration, alternating order.", "count", "3"},
        {{"l", "language"}, "Qwen language name used in the prompt.", "name", "Korean"},
        {{"o", "output"}, "Write JSON report atomically; stdout if omitted.", "path"}});
    parser.process(app);
    bool validThreads = false, validRepeats = false;
    const int threads = parser.value("threads").toInt(&validThreads);
    const int repeats = parser.value("repeats").toInt(&validRepeats);
    if (parser.value("model").isEmpty() || parser.value("wav").isEmpty()
        || !validThreads || threads <= 0 || !validRepeats || repeats <= 0) {
        QTextStream(stderr) << "Specify --model, --wav, and positive --threads/--repeats.\n";
        return 2;
    }
    try {
        const auto clip = echoflow::CrispSession::readWavF32(parser.value("wav").toStdString());
        if (clip.size() < 48000) throw std::runtime_error("fixture must contain at least 3 seconds");
        auto params = qwen3_asr_context_default_params();
        params.n_threads = threads;
        params.verbosity = 0;
        params.use_gpu = false;
        using Context = std::unique_ptr<qwen3_asr_context, decltype(&qwen3_asr_free)>;
        std::array<Context, 2> contexts{
            Context(qwen3_asr_init_from_file(parser.value("model").toLocal8Bit().constData(), params), &qwen3_asr_free),
            Context(qwen3_asr_init_from_file(parser.value("model").toLocal8Bit().constData(), params), &qwen3_asr_free)};
        if (!contexts[0] || !contexts[1]) throw std::runtime_error("failed to load benchmark model");
        const std::string suffix = "language " + parser.value("language").toStdString() + "<asr_text>";
        const bool compareEncoder = parser.isSet("encoder-cache");
        PrefixKvCache retained;
        PrefixKvCache retainedBaseline;
        std::unique_ptr<crisp_audio_encoder_cache, decltype(&crisp_audio_encoder_cache_free)> encoderCache(
            compareEncoder ? crisp_audio_encoder_cache_new() : nullptr, &crisp_audio_encoder_cache_free);
        if (compareEncoder && !encoderCache) throw std::runtime_error("cannot create encoder cache");
        std::array<std::vector<double>, 2> totals;
        QJsonArray rows;
        // Exclude model loading and first graph/backend initialization from both
        // variants. Separate contexts avoid cross-variant KV invalidation.
        for (int variant : {0, 1}) {
            PrefixKvCache local;
            bool ok = false;
            core_qwen3_stream_model::generate(contexts[variant].get(), false,
                std::vector<float>(clip.begin(), clip.begin() + 8000), suffix, 64, &ok,
                variant ? &retained : (compareEncoder ? &retainedBaseline : &local),
                nullptr, !compareEncoder && variant == 0);
            if (!ok) throw std::runtime_error("warmup inference failed");
        }
        for (int repetition = 0; repetition < repeats; ++repetition) {
            if (encoderCache) crisp_audio_encoder_cache_reset(encoderCache.get());
            for (int count : {8000, 24000, 48000}) {
                const std::vector<float> audio(clip.begin(), clip.begin() + count);
                std::array<std::string, 2> text;
                std::array<DecodeMetrics, 2> timing;
                for (int position : {0, 1}) {
                    const int variant = (position + repetition) % 2;
                    PrefixKvCache local;
                    bool ok = false;
                    text[variant] = core_qwen3_stream_model::generate(contexts[variant].get(), false,
                        audio, suffix, 64, &ok, variant ? &retained : (compareEncoder ? &retainedBaseline : &local),
                        &timing[variant], !compareEncoder && variant == 0,
                        compareEncoder && variant == 1 ? encoderCache.get() : nullptr);
                    if (!ok || !timing[variant].success) throw std::runtime_error("measured inference failed");
                    if (timing[variant].prefix_reused != (compareEncoder || variant == 1)) {
                        throw std::runtime_error("unexpected prefix cache state");
                    }
                    totals[variant].push_back(timing[variant].total_ms);
                }
                if (text[0] != text[1] || timing[0].generated_tokens != timing[1].generated_tokens) {
                    throw std::runtime_error("prefix cache changed the transcript or token count");
                }
                rows.append(QJsonObject{{"repetition", repetition}, {"audio_ms", count / 16},
                    {"transcript_sha256", QString::fromLatin1(QCryptographicHash::hash(
                        QByteArray::fromStdString(text[0]), QCryptographicHash::Sha256).toHex())},
                    {"baseline", metricsJson(timing[0])}, {"optimized", metricsJson(timing[1])}});
            }
        }
        const auto baseline = median(totals[0]);
        const auto optimized = median(totals[1]);
        const QJsonObject report{{"schema_version", 1}, {"created_at", QDateTime::currentDateTimeUtc().toString(Qt::ISODate)},
            {"cpu", cpuName()}, {"architecture", QSysInfo::currentCpuArchitecture()}, {"threads", threads},
            {"model", QFileInfo(parser.value("model")).fileName()}, {"model_sha256", sha256(parser.value("model"))},
            {"wav_sha256", sha256(parser.value("wav"))}, {"repeats", repeats},
            {"language", parser.value("language")},
            {"scope", compareEncoder ? "incremental encoder within an utterance; unchanged attention semantics"
                : "warm fixed prompt KV reuse across independent segments and single cold prefill"},
            {"all_transcripts_match", true}, {"baseline_median_ms", baseline}, {"optimized_median_ms", optimized},
            {"median_latency_change_percent", (optimized / baseline - 1.0) * 100.0}, {"samples", rows}};
        const auto json = QJsonDocument(report).toJson(QJsonDocument::Indented);
        if (parser.value("output").isEmpty()) {
            QTextStream(stdout) << json;
        } else {
            QSaveFile output(parser.value("output"));
            if (!output.open(QIODevice::WriteOnly) || output.write(json) != json.size() || !output.commit()) {
                throw std::runtime_error("cannot write JSON report");
            }
            QTextStream(stdout) << "baseline_median_ms=" << baseline << " optimized_median_ms=" << optimized
                << " transcripts_match=true\n";
        }
        return 0;
    } catch (const std::exception& error) {
        QTextStream(stderr) << error.what() << '\n';
        return 1;
    }
}
