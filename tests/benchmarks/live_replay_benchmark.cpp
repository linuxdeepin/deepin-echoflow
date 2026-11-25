// SPDX-FileCopyrightText: 2026 Hualet Wang
// SPDX-License-Identifier: GPL-3.0-or-later

#include "Config.h"
#include "CrispLiveVoicePipeline.h"
#include "CrispSession.h"

#include <QCommandLineParser>
#include <QCoreApplication>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;

double milliseconds(Clock::duration duration)
{
    return std::chrono::duration<double, std::milli>(duration).count();
}
} // namespace

int main(int argc, char** argv)
{
    QCoreApplication app(argc, argv);
    QCommandLineParser parser;
    parser.addHelpOption();
    parser.addOptions({
        {"config", "EchoFlow configuration.", "path"},
        {"audio", "16 kHz mono WAV to replay.", "path"},
        {"threads", "ASR thread count.", "count", "6"},
        {"iterations", "Number of replays, including warmup.", "count", "1"},
        {"trace-previews", "Include timestamped raw ASR previews in the JSON report."},
    });
    parser.process(app);

    try {
        bool threadsOk = false, iterationsOk = false;
        const int threads = parser.value("threads").toInt(&threadsOk);
        const int iterations = parser.value("iterations").toInt(&iterationsOk);
        if (!parser.isSet("config") || !parser.isSet("audio") || !threadsOk || threads < 1
            || !iterationsOk || iterations < 1) {
            throw std::invalid_argument("specify --config, --audio and positive counts");
        }
        auto cfg = echoflow::loadDtkConf(parser.value("config").toStdString());
        cfg.crispThreads = threads;
        cfg.streamTranscription = true;
        const auto wav = echoflow::CrispSession::readWavF32(parser.value("audio").toStdString());
        if (wav.empty()) throw std::runtime_error("empty or unreadable WAV");
        std::vector<int16_t> pcm;
        pcm.reserve(wav.size());
        for (const auto sample : wav) {
            pcm.push_back(static_cast<int16_t>(std::lround(std::clamp(sample, -1.0f, 32767.f / 32768) * 32768)));
        }

        std::atomic<Clock::rep> firstPreview{0};
        const bool tracePreviews = parser.isSet("trace-previews");
        std::mutex previewMutex;
        std::vector<std::pair<Clock::time_point, std::string>> previewTrace;
        echoflow::CrispLiveVoicePipeline pipeline(cfg);
        pipeline.setPartialTextCallback([&](const std::string& text) {
            const auto now = Clock::now();
            if (!text.empty()) {
                Clock::rep none = 0;
                firstPreview.compare_exchange_strong(none, now.time_since_epoch().count());
            }
            if (tracePreviews) {
                std::lock_guard<std::mutex> lock(previewMutex);
                previewTrace.emplace_back(now, text);
            }
        });
        for (int i = 0; i < iterations; ++i) {
            firstPreview = 0;
            pipeline.startReplay();
            const auto start = Clock::now();
            constexpr size_t kFrameSamples = 320;
            for (size_t begin = 0; begin < pcm.size();) {
                const auto end = std::min(begin + kFrameSamples, pcm.size());
                std::this_thread::sleep_until(start + std::chrono::microseconds(end * 1000000ULL / 16000));
                pipeline.appendReplayAudio({pcm.begin() + begin, pcm.begin() + end});
                begin = end;
            }
            const auto stop = Clock::now();
            auto transcript = pipeline.finish();
            const auto done = Clock::now();
            if (cfg.stripTrailingPunctuation) transcript = echoflow::stripPunctuation(transcript);
            const auto first = Clock::time_point(Clock::duration(firstPreview.load()));
            const QJsonValue preview = first >= start && first < stop
                ? QJsonValue(milliseconds(first - start)) : QJsonValue(QJsonValue::Null);
            QJsonObject row{
                {"mode", "live-replay"},
                {"iteration", i + 1},
                {"first_preview_ms", preview},
                {"stop_to_final_ms", milliseconds(done - stop)},
                {"transcript", QString::fromStdString(transcript)},
                {"model", QString::fromStdString(cfg.crispModelPath)},
            };
            if (tracePreviews) {
                QJsonArray updates;
                std::lock_guard<std::mutex> lock(previewMutex);
                for (const auto& [when, text] : previewTrace) {
                    updates.append(QJsonObject{{"ms", milliseconds(when - start)},
                                               {"after_stop", when >= stop},
                                               {"text", QString::fromStdString(text)}});
                }
                previewTrace.clear();
                row.insert("previews", updates);
            }
            const auto json = QJsonDocument(row).toJson(QJsonDocument::Compact);
            std::puts(json.constData());
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "live replay failed: %s\n", e.what());
        return 1;
    }
    return 0;
}
