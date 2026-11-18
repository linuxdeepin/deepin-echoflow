// SPDX-FileCopyrightText: 2026 Hualet Wang
// SPDX-License-Identifier: GPL-3.0-or-later

#include "AudioLevelMeter.h"
#include "Recorder.h"
#include "CrispSession.h"

#include <QDataStream>
#include <QFile>
#include <QScopeGuard>
#include <QTemporaryDir>
#include <QtTest>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

using namespace echoflow;

namespace {
std::vector<int16_t> tone(int amplitude, size_t count = 800)
{
    std::vector<int16_t> samples(count);
    for (size_t i = 0; i < count; ++i) samples[i] = i % 2 ? -amplitude : amplitude;
    return samples;
}

std::vector<int16_t> frequencies(int rate, const std::vector<std::pair<double, double>>& tones)
{
    std::vector<int16_t> samples(static_cast<size_t>(rate / 20));
    for (size_t i = 0; i < samples.size(); ++i) {
        double value = 0;
        for (const auto& tone : tones) {
            value += tone.second * std::sin(2 * 3.141592653589793 * tone.first * i / rate);
        }
        samples[i] = static_cast<int16_t>(std::lround(std::clamp(value, -1.0, 1.0) * 32767));
    }
    return samples;
}

QByteArray waveHeader(int format, int bits, int channels, bool extensible)
{
    QByteArray bytes;
    QDataStream stream(&bytes, QIODevice::WriteOnly);
    stream.setByteOrder(QDataStream::LittleEndian);
    stream.writeRawData("RIFF", 4);
    stream << quint32(0); // Unfinalized size, as during pw-record capture.
    stream.writeRawData("WAVEJUNK", 8);
    stream << quint32(1);
    stream.writeRawData("x\0", 2); // Unknown chunk with odd size and padding.
    stream.writeRawData("fmt ", 4);
    stream << quint32(extensible ? 40 : 16) << quint16(extensible ? 0xfffe : format)
           << quint16(channels) << quint32(16000) << quint32(16000 * channels * bits / 8)
           << quint16(channels * bits / 8) << quint16(bits);
    if (extensible) {
        stream << quint16(22) << quint16(bits) << quint32(0) << quint32(format);
        stream.writeRawData(QByteArray::fromHex("00001000800000aa00389b71").constData(), 12);
    }
    stream.writeRawData("data", 4);
    stream << quint32(0);
    return bytes;
}

QByteArray waveSamples(int format, int bits, int channels, bool silence)
{
    QByteArray bytes;
    for (int i = 0; i < 800 * channels; ++i) {
        const double sample = silence ? 0 : ((i / channels) % 2 ? -0.25 : 0.25);
        std::uint64_t raw;
        if (format == 3 && bits == 32) {
            const float value = sample;
            std::uint32_t word;
            std::memcpy(&word, &value, sizeof(word));
            raw = word;
        } else if (format == 3) {
            std::memcpy(&raw, &sample, sizeof(raw));
        } else if (bits == 8) {
            raw = static_cast<unsigned>(128 + sample * 128);
        } else {
            raw = static_cast<std::int64_t>(sample * (std::int64_t(1) << (bits - 1)));
        }
        for (int j = 0; j < bits / 8; ++j) bytes.append(char((raw >> (j * 8)) & 0xff));
    }
    return bytes;
}
}

class TestAudioLevelMeter : public QObject {
    Q_OBJECT
private slots:
    void silenceNoiseAndDcRemainStill();
    void volumeTracksInputAndReturnsToZero();
    void readBoundariesDoNotChangeWindows();
    void waveformComesFromOneAudioWindow();
    void waveformIsIndependentOfPacketBoundaries();
    void spectrumKeepsBandEnergyAndAbsoluteVolume();
    void steadyToneHasStableSpectrum();
    void spectrumRespondsToFrequency_data();
    void spectrumRespondsToFrequency();
    void stereoPhaseDoesNotCancelSpectrum();
    void waveformSpeechGateRejectsBackground_data();
    void waveformSpeechGateRejectsBackground();
    void waveformSpeechGateShowsSpeechAndClearsAtSilence();
    void waveformSpeechGateAcceptsLocalSpeechFixture();
    void backgroundChangesDoNotForceFullHeight();
    void growingWave_data();
    void growingWave();
    void invalidWaveIsIgnored();
    void fileRecorderMetersWithoutAnotherCapture();
};

void TestAudioLevelMeter::silenceNoiseAndDcRemainStill()
{
    std::vector<double> levels;
    AudioLevelMeter meter(16000, 1, [&](const AudioLevelFrame& frame) { levels.push_back(frame.level); });
    for (const auto samples : {std::vector<int16_t>(800), tone(50), std::vector<int16_t>(800, 8000)}) {
        meter.append(samples.data(), samples.size());
    }
    QCOMPARE(levels, (std::vector<double>{0, 0, 0}));
    levels.clear();
    AudioLevelMeter stereo(16000, 2, [&](const AudioLevelFrame& frame) { levels.push_back(frame.level); });
    std::vector<int16_t> offsets(1600);
    for (size_t i = 0; i < offsets.size(); ++i) offsets[i] = i % 2 ? -6000 : 8000;
    stereo.append(offsets.data(), offsets.size());
    QCOMPARE(levels, (std::vector<double>{0}));
}

void TestAudioLevelMeter::volumeTracksInputAndReturnsToZero()
{
    std::vector<double> levels;
    AudioLevelMeter meter(16000, 1, [&](const AudioLevelFrame& frame) { levels.push_back(frame.level); });
    for (int amplitude : {500, 4000, 16000, 0}) {
        const auto samples = tone(amplitude);
        meter.append(samples.data(), samples.size());
    }
    QCOMPARE(levels.size(), size_t(4));
    QVERIFY(levels[0] > 0);
    QVERIFY(levels[1] > levels[0]);
    QVERIFY(levels[2] > levels[1]);
    QVERIFY(levels[2] <= 1);
    QCOMPARE(levels[3], 0.0);
}

void TestAudioLevelMeter::readBoundariesDoNotChangeWindows()
{
    std::vector<double> levels;
    AudioLevelMeter meter(16000, 1, [&](const AudioLevelFrame& frame) { levels.push_back(frame.level); });
    const auto samples = tone(4000, 1600);
    meter.append(samples.data(), 799);
    QVERIFY(levels.empty());
    meter.append(samples.data() + 799, 2);
    QCOMPARE(levels.size(), size_t(1));
    meter.append(samples.data() + 801, 799);
    QCOMPARE(levels.size(), size_t(2));
    QCOMPARE(levels.front(), levels.back());
}

void TestAudioLevelMeter::waveformComesFromOneAudioWindow()
{
    std::vector<AudioLevelFrame> frames;
    AudioLevelMeter meter(16000, 1, [&](const AudioLevelFrame& frame) { frames.push_back(frame); });
    const auto window = frequencies(16000, {{300, 0.04}, {2200, 0.12}});
    meter.append(window.data(), window.size());
    QCOMPARE(frames.size(), size_t(1));
    QVERIFY(frames[0].bars[1] > 0.1);
    QVERIFY(frames[0].bars[4] > frames[0].bars[1] * 2);
    QVERIFY(frames[0].bars[0] < 0.01);
    QVERIFY(frames[0].bars[6] < 0.01);
    const auto silence = tone(0);
    meter.append(silence.data(), silence.size());
    QCOMPARE(frames.size(), size_t(2));
    QCOMPARE(frames[1].bars, (std::array<double, 7>{0, 0, 0, 0, 0, 0, 0}));
}

void TestAudioLevelMeter::waveformIsIndependentOfPacketBoundaries()
{
    std::vector<AudioLevelFrame> whole, fragmented;
    AudioLevelMeter one(16000, 2, [&](const AudioLevelFrame& frame) { whole.push_back(frame); });
    AudioLevelMeter many(16000, 2, [&](const AudioLevelFrame& frame) { fragmented.push_back(frame); });
    std::vector<int16_t> samples(3200);
    for (size_t i = 0; i < samples.size(); ++i) {
        const int amplitude = static_cast<int>((i / 320) % 5) * 2000;
        samples[i] = (i / 2) % 2 ? -amplitude : amplitude;
    }
    one.append(samples.data(), samples.size());
    for (size_t i = 0; i < samples.size(); i += 7) {
        many.append(samples.data() + i, std::min<size_t>(7, samples.size() - i));
    }
    QCOMPARE(whole.size(), size_t(2));
    QCOMPARE(fragmented.size(), whole.size());
    for (size_t i = 0; i < whole.size(); ++i) {
        QCOMPARE(fragmented[i].level, whole[i].level);
        QCOMPARE(fragmented[i].bars, whole[i].bars);
    }
}

void TestAudioLevelMeter::spectrumKeepsBandEnergyAndAbsoluteVolume()
{
    AudioLevelFrame frame;
    AudioLevelMeter meter(16000, 1, [&](const AudioLevelFrame& value) { frame = value; });
    const auto quiet = frequencies(16000, {{300, 0.03}, {2200, 0.12}});
    meter.append(quiet.data(), quiet.size());
    const auto first = frame;
    QVERIFY(first.bars[4] > first.bars[1]);
    QVERIFY(std::abs(first.bars[1] / first.bars[4] - 0.25) < 0.01);
    QVERIFY(*std::max_element(first.bars.begin(), first.bars.end()) < 0.9);
    const auto loud = frequencies(16000, {{300, 0.06}, {2200, 0.24}});
    meter.append(loud.data(), loud.size());
    QVERIFY(frame.level > first.level);
    QVERIFY(frame.bars[4] > first.bars[4]); // No per-frame stretch to maximum.
    QVERIFY(std::abs(frame.bars[1] / frame.bars[4] - first.bars[1] / first.bars[4]) < 0.001);
}

void TestAudioLevelMeter::steadyToneHasStableSpectrum()
{
    std::vector<AudioLevelFrame> frames;
    AudioLevelMeter meter(16000, 1, [&](const AudioLevelFrame& frame) { frames.push_back(frame); });
    const auto input = frequencies(16000, {{600, 0.2}});
    meter.append(input.data(), input.size());
    meter.append(input.data(), input.size());
    QCOMPARE(frames.size(), size_t(2));
    QCOMPARE(frames[0].bars, frames[1].bars);
    QCOMPARE(std::distance(frames[0].bars.begin(), std::max_element(frames[0].bars.begin(), frames[0].bars.end())), 2);
    const auto silence = tone(0);
    meter.append(silence.data(), silence.size());
    QCOMPARE(frames.back().bars, (std::array<double, 7>{0, 0, 0, 0, 0, 0, 0}));
}

void TestAudioLevelMeter::spectrumRespondsToFrequency_data()
{
    QTest::addColumn<int>("rate");
    QTest::addColumn<int>("frequency");
    QTest::addColumn<int>("band");
    const std::array<int, 7> centers{140, 300, 600, 1200, 2100, 3300, 5600};
    for (int rate : {8000, 16000, 44100, 48000}) {
        for (size_t band = 0; band < centers.size(); ++band) {
            if (centers[band] >= rate / 2) continue;
            const auto name = QStringLiteral("%1-%2Hz").arg(rate).arg(centers[band]);
            QTest::newRow(qPrintable(name)) << rate << centers[band] << static_cast<int>(band);
        }
    }
}

void TestAudioLevelMeter::spectrumRespondsToFrequency()
{
    QFETCH(int, rate);
    QFETCH(int, frequency);
    QFETCH(int, band);
    AudioLevelFrame frame;
    AudioLevelMeter meter(rate, 1, [&](const AudioLevelFrame& value) { frame = value; });
    const auto input = frequencies(rate, {{static_cast<double>(frequency), 0.2}});
    meter.append(input.data(), input.size());
    QCOMPARE(std::distance(frame.bars.begin(), std::max_element(frame.bars.begin(), frame.bars.end())), band);
    QVERIFY(frame.bars[static_cast<size_t>(band)] > 0.5);
    for (size_t i = 0; i < frame.bars.size(); ++i) {
        if (i != static_cast<size_t>(band)) QVERIFY(frame.bars[i] < frame.bars[static_cast<size_t>(band)] / 4);
    }
    if (rate == 8000) QCOMPARE(frame.bars[6], 0.0); // No bins above Nyquist.
}

void TestAudioLevelMeter::stereoPhaseDoesNotCancelSpectrum()
{
    AudioLevelFrame mono, stereo;
    AudioLevelMeter one(16000, 1, [&](const AudioLevelFrame& frame) { mono = frame; });
    AudioLevelMeter two(16000, 2, [&](const AudioLevelFrame& frame) { stereo = frame; });
    const auto samples = frequencies(16000, {{1200, 0.2}});
    std::vector<int16_t> opposite;
    for (auto value : samples) {
        opposite.push_back(value);
        opposite.push_back(-value);
    }
    one.append(samples.data(), samples.size());
    two.append(opposite.data(), opposite.size());
    QCOMPARE(stereo.level, mono.level);
    QCOMPARE(stereo.bars, mono.bars);
}

void TestAudioLevelMeter::waveformSpeechGateRejectsBackground_data()
{
    QTest::addColumn<int>("rate");
    QTest::addColumn<double>("volume");
    for (int rate : {8000, 16000, 44100, 48000}) {
        for (double volume : {0.03, 0.1, 0.35}) {
            const auto name = QStringLiteral("%1-%2").arg(rate).arg(volume);
            QTest::newRow(qPrintable(name)) << rate << volume;
        }
    }
}

void TestAudioLevelMeter::waveformSpeechGateRejectsBackground()
{
    QFETCH(int, rate);
    QFETCH(double, volume);
    std::vector<AudioLevelFrame> raw, gated;
    AudioLevelMeter meter(rate, 1, [&](const AudioLevelFrame& frame) { raw.push_back(frame); });
    AudioLevelMeter speechMeter(rate, 1, [&](const AudioLevelFrame& frame) { gated.push_back(frame); }, true);
    std::mt19937 random(42);
    std::normal_distribution<double> noise(0, volume);
    for (int window = 0; window < 40; ++window) {
        std::vector<int16_t> samples(static_cast<size_t>(rate / 20));
        for (auto& sample : samples) {
            sample = static_cast<int16_t>(std::lround(std::clamp(noise(random), -1.0, 1.0) * 32767));
        }
        meter.append(samples.data(), samples.size());
        speechMeter.append(samples.data(), samples.size());
    }
    QCOMPARE(raw.size(), size_t(40));
    QCOMPARE(gated.size(), size_t(40));
    for (size_t i = 0; i < gated.size(); ++i) {
        QVERIFY(raw[i].level > 0.4); // Background previously drove high columns.
        QCOMPARE(gated[i].level, 0.0);
        QCOMPARE(gated[i].bars, (std::array<double, 7>{0, 0, 0, 0, 0, 0, 0}));
    }
}

void TestAudioLevelMeter::waveformSpeechGateShowsSpeechAndClearsAtSilence()
{
    std::vector<AudioLevelFrame> frames;
    AudioLevelMeter meter(16000, 2, [&](const AudioLevelFrame& frame) { frames.push_back(frame); }, true);
    for (int window = 0; window < 20; ++window) {
        std::vector<int16_t> samples(1600);
        for (size_t i = 0; i < 800; ++i) {
            const double t = (window * 800 + i) / 16000.0;
            const double value = (0.1 + 0.07 * std::sin(2 * 3.141592653589793 * 4 * t))
                * (std::sin(2 * 3.141592653589793 * 180 * t)
                   + 0.3 * std::sin(2 * 3.141592653589793 * 720 * t));
            samples[2 * i] = samples[2 * i + 1] = static_cast<int16_t>(value * 32767);
        }
        meter.append(samples.data(), samples.size());
    }
    QCOMPARE(frames.size(), size_t(20));
    QVERIFY(std::any_of(frames.begin(), frames.end(), [](const AudioLevelFrame& frame) { return frame.level > 0.5; }));
    const std::vector<int16_t> silence(1600, 0);
    meter.append(silence.data(), silence.size());
    QCOMPARE(frames.back().level, 0.0);
    QCOMPARE(frames.back().bars, (std::array<double, 7>{0, 0, 0, 0, 0, 0, 0}));
    meter.append(silence.data(), silence.size());
    QCOMPARE(frames.back().level, 0.0);
}

void TestAudioLevelMeter::waveformSpeechGateAcceptsLocalSpeechFixture()
{
    const QString path = QFINDTESTDATA("../third_party/crispasr/samples/ko-369.wav");
    if (path.isEmpty()) QSKIP("optional local speech fixture is not available");
    const auto pcm = CrispSession::readWavF32(path.toStdString());
    QVERIFY(pcm.size() > 16000);
    std::vector<AudioLevelFrame> frames;
    AudioLevelMeter meter(16000, 1, [&](const AudioLevelFrame& frame) { frames.push_back(frame); }, true);
    for (const float sample : pcm) meter.appendSample(sample);
    QVERIFY(std::any_of(frames.begin(), frames.end(), [](const AudioLevelFrame& frame) { return frame.level > 0.1; }));
    const std::vector<int16_t> silence(1600, 0);
    meter.append(silence.data(), silence.size());
    QCOMPARE(frames.back().level, 0.0);
    QCOMPARE(frames.back().bars, (std::array<double, 7>{0, 0, 0, 0, 0, 0, 0}));
}

void TestAudioLevelMeter::backgroundChangesDoNotForceFullHeight()
{
    std::vector<AudioLevelFrame> frames;
    AudioLevelMeter meter(16000, 1, [&](const AudioLevelFrame& frame) { frames.push_back(frame); }, true);
    std::mt19937 random(42);
    for (double volume : {0.03, 0.35, 0.1}) {
        std::normal_distribution<double> noise(0, volume);
        for (int window = 0; window < 20; ++window) {
            std::vector<int16_t> samples(800);
            for (auto& sample : samples) sample = static_cast<int16_t>(std::lround(std::clamp(noise(random), -1.0, 1.0) * 32767));
            meter.append(samples.data(), samples.size());
        }
        for (size_t i = frames.size() - 5; i < frames.size(); ++i) {
            QCOMPARE(frames[i].level, 0.0);
            QCOMPARE(frames[i].bars, (std::array<double, 7>{0, 0, 0, 0, 0, 0, 0}));
        }
    }
    // A voiced, structured signal can still open the display above the newly
    // learned noise floor; it is never discarded from the recognition path.
    for (int window = 0; window < 20; ++window) {
        const auto speech = frequencies(16000, {{180, 0.06}, {720, 0.02}});
        meter.append(speech.data(), speech.size());
    }
    QVERIFY(std::any_of(frames.end() - 10, frames.end(), [](const AudioLevelFrame& frame) { return frame.level > 0.1; }));
}

void TestAudioLevelMeter::growingWave_data()
{
    QTest::addColumn<int>("format");
    QTest::addColumn<int>("bits");
    QTest::addColumn<int>("channels");
    QTest::addColumn<bool>("extensible");
    QTest::newRow("pcm8") << 1 << 8 << 1 << false;
    QTest::newRow("pcm16") << 1 << 16 << 1 << false;
    QTest::newRow("pcm24-extensible-stereo") << 1 << 24 << 2 << true;
    QTest::newRow("pcm32") << 1 << 32 << 1 << false;
    QTest::newRow("float32-stereo") << 3 << 32 << 2 << false;
    QTest::newRow("float64") << 3 << 64 << 1 << false;
}

void TestAudioLevelMeter::growingWave()
{
    QFETCH(int, format);
    QFETCH(int, bits);
    QFETCH(int, channels);
    QFETCH(bool, extensible);
    QTemporaryDir directory;
    QVERIFY(directory.isValid());
    const auto path = directory.filePath("recording.wav");
    WavAudioLevelReader reader(path.toStdString());
    QVERIFY(!reader.poll()); // Recorder has not created the file yet.
    QFile file(path);
    QVERIFY(file.open(QIODevice::WriteOnly));
    const auto header = waveHeader(format, bits, channels, extensible);
    QCOMPARE(file.write(header.left(20)), qint64(20));
    QVERIFY(file.flush());
    QVERIFY(!reader.poll()); // Header may arrive in multiple writes.
    QCOMPARE(file.write(header.mid(20)), qint64(header.size() - 20));
    QVERIFY(file.flush());
    QVERIFY(!reader.poll());
    const auto sound = waveSamples(format, bits, channels, false);
    QCOMPARE(file.write(sound.left(7)), qint64(7));
    QVERIFY(file.flush());
    QVERIFY(!reader.poll());
    QCOMPARE(file.write(sound.mid(7)), qint64(sound.size() - 7));
    QVERIFY(file.flush());
    const auto level = reader.poll();
    QVERIFY(level);
    QVERIFY(level->level > 0.5 && level->level <= 1);
    QVERIFY(!reader.poll()); // Do not replay already consumed samples.
    const auto silence = waveSamples(format, bits, channels, true);
    QCOMPARE(file.write(silence), qint64(silence.size()));
    QVERIFY(file.flush());
    const auto silentLevel = reader.poll();
    QVERIFY(silentLevel);
    QCOMPARE(silentLevel->level, 0.0);
}

void TestAudioLevelMeter::invalidWaveIsIgnored()
{
    QTemporaryDir directory;
    QFile file(directory.filePath("invalid.wav"));
    QVERIFY(file.open(QIODevice::WriteOnly));
    file.write("not a WAV file");
    QVERIFY(file.flush());
    WavAudioLevelReader reader(file.fileName().toStdString());
    QVERIFY(!reader.poll());
}

void TestAudioLevelMeter::fileRecorderMetersWithoutAnotherCapture()
{
    QTemporaryDir directory;
    QVERIFY(directory.isValid());
    QFile recorderProgram(directory.filePath("pw-record"));
    QVERIFY(recorderProgram.open(QIODevice::WriteOnly));
    const QByteArray program = R"PY(#!/usr/bin/env python3
import signal
import struct
import sys
import time
import math

active = True
def stop(signum, frame):
    global active
    active = False
signal.signal(signal.SIGINT, stop)
with open(sys.argv[-1], 'wb', buffering=0) as stream:
    stream.write(struct.pack('<4sI4s4sIHHIIHH4sI',
        b'RIFF', 0, b'WAVE', b'fmt ', 16, 1, 1, 16000, 32000, 2, 16, b'data', 0))
    index = 0
    while active:
        samples = []
        for i in range(800):
            t = (index * 800 + i) / 16000
            value = (0.1 + 0.07 * math.sin(2 * math.pi * 4 * t)) * (math.sin(2 * math.pi * 180 * t) + 0.3 * math.sin(2 * math.pi * 720 * t))
            samples.append(int(value * 32767) if index < 8 else 0)
        stream.write(struct.pack('<800h', *samples))
        index += 1
        time.sleep(0.05)
)PY";
    QCOMPARE(recorderProgram.write(program), qint64(program.size()));
    recorderProgram.close();
    QVERIFY(recorderProgram.setPermissions(QFile::ReadOwner | QFile::WriteOwner | QFile::ExeOwner));
    const auto oldPath = qgetenv("PATH");
    const auto restorePath = qScopeGuard([oldPath] { qputenv("PATH", oldPath); });
    qputenv("PATH", directory.path().toLocal8Bit() + ':' + oldPath);
    Config config = Config::defaultConfig();
    config.recordingsDir = directory.path().toStdString();
    config.minRecordSeconds = 0;
    std::atomic<bool> sound{false};
    std::atomic<bool> silence{false};
    std::atomic<int> notifications{0};
    PipeWireRecorder recorder(config);
    recorder.setAudioLevelCallback([&](const AudioLevelFrame& frame) {
        const double level = frame.level;
        ++notifications;
        if (level > 0.5) sound = true;
        if (sound && level == 0) silence = true;
    });
    recorder.start();
    QTRY_VERIFY(sound.load());
    QTRY_VERIFY(silence.load());
    const auto path = recorder.stop();
    QVERIFY(!path.empty());
    QVERIFY(std::filesystem::file_size(path) > 1024);
    const int stoppedCount = notifications;
    QTest::qWait(100);
    QCOMPARE(notifications.load(), stoppedCount);
}

QTEST_GUILESS_MAIN(TestAudioLevelMeter)
#include "test_audio_level_meter.moc"
