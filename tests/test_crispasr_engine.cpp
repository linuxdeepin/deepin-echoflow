// SPDX-FileCopyrightText: 2026 Hualet Wang
// SPDX-License-Identifier: GPL-3.0-or-later

#include <QtTest/QtTest>

#include "CrispAsrEngine.h"
#include "CrispSession.h"

#include <QDataStream>
#include <QTemporaryFile>
#include <cmath>
#include <cstring>

#include <cstdlib>
#include <map>
#include <string>

using namespace echoflow;

class TestCrispAsrEngine : public QObject {
    Q_OBJECT

private slots:
    void languageCodeUsesSystemLocaleForAuto();
    void languageCodeFallsBackToEnglishForUnsupportedLocale();
    void languageCodePreservesExplicitSelection();
    void transcribeReturnsEmptyWhenModelMissing();
    void wavConversionMatchesRecordingFormats_data();
    void wavConversionMatchesRecordingFormats();
    void truncatedWavIsRejected();
};

namespace {

class LocaleEnvironment {
public:
    LocaleEnvironment()
    {
        for (const char* name : {"LC_ALL", "LC_MESSAGES", "LANG"}) {
            const char* value = std::getenv(name);
            if (value) {
                saved_[name] = value;
            }
            unsetenv(name);
        }
    }

    ~LocaleEnvironment()
    {
        for (const char* name : {"LC_ALL", "LC_MESSAGES", "LANG"}) {
            auto it = saved_.find(name);
            if (it == saved_.end()) {
                unsetenv(name);
            } else {
                setenv(name, it->second.c_str(), 1);
            }
        }
    }

private:
    std::map<std::string, std::string> saved_;
};

} // namespace

void TestCrispAsrEngine::languageCodeUsesSystemLocaleForAuto()
{
    LocaleEnvironment env;
    qputenv("LANG", "zh_CN.UTF-8");
    QCOMPARE(QString::fromStdString(CrispAsrEngine::languageCode("")),
             QStringLiteral("zh"));

    qputenv("LANG", "pt_BR.UTF-8");
    QCOMPARE(QString::fromStdString(CrispAsrEngine::languageCode("")),
             QStringLiteral("pt"));
}

void TestCrispAsrEngine::languageCodeFallsBackToEnglishForUnsupportedLocale()
{
    LocaleEnvironment env;
    qputenv("LANG", "de_DE.UTF-8");
    QCOMPARE(QString::fromStdString(CrispAsrEngine::languageCode("")),
             QStringLiteral("en"));
}

void TestCrispAsrEngine::languageCodePreservesExplicitSelection()
{
    LocaleEnvironment env;
    qputenv("LANG", "zh_CN.UTF-8");
    QCOMPARE(QString::fromStdString(CrispAsrEngine::languageCode("French")),
             QStringLiteral("fr"));
}

void TestCrispAsrEngine::transcribeReturnsEmptyWhenModelMissing()
{
    Config cfg = Config::defaultConfig();
    cfg.crispModelPath = "/tmp/deepin-echoflow-no-such-model.gguf";
    CrispAsrEngine engine(cfg);
    try {
        QCOMPARE(QString::fromStdString(engine.transcribe("/tmp/deepin-echoflow-no-such-audio.wav")),
                 QString());
    } catch (const std::exception& e) {
        QFAIL(qPrintable(QStringLiteral("transcribe threw: %1").arg(e.what())));
    }
}

void TestCrispAsrEngine::wavConversionMatchesRecordingFormats_data()
{
    QTest::addColumn<int>("rate");
    QTest::addColumn<int>("channels");
    QTest::addColumn<int>("format");
    QTest::addColumn<int>("bits");
    for (int rate : {8000, 16000, 22050, 44100, 48000}) {
        for (int channels : {1, 2}) {
            for (const auto spec : {std::pair<int, int>{1, 8}, {1, 16}, {1, 24}, {1, 32}, {3, 32}}) {
                const auto name = QStringLiteral("%1-%2-%3-%4").arg(rate).arg(channels).arg(spec.first).arg(spec.second);
                QTest::newRow(qPrintable(name)) << rate << channels << spec.first << spec.second;
            }
        }
    }
}

void TestCrispAsrEngine::wavConversionMatchesRecordingFormats()
{
    QFETCH(int, rate);
    QFETCH(int, channels);
    QFETCH(int, format);
    QFETCH(int, bits);
    QTemporaryFile file;
    QVERIFY(file.open());
    const int frames = rate / 10; // 100 ms at every rate.
    const int dataBytes = frames * channels * bits / 8;
    QDataStream stream(&file);
    stream.setByteOrder(QDataStream::LittleEndian);
    stream.writeRawData("RIFF", 4);
    stream << quint32(46 + dataBytes);
    stream.writeRawData("WAVEJUNK", 8);
    stream << quint32(1);
    stream.writeRawData("x\0", 2);
    stream.writeRawData("fmt ", 4);
    stream << quint32(16) << quint16(format) << quint16(channels) << quint32(rate)
           << quint32(rate * channels * bits / 8) << quint16(channels * bits / 8) << quint16(bits);
    stream.writeRawData("data", 4);
    stream << quint32(dataBytes);
    for (int i = 0; i < frames; ++i) {
        const double tone = 0.4 * std::sin(2 * 3.141592653589793 * 400 * i / rate);
        for (int channel = 0; channel < channels; ++channel) {
            // Distinct channels ensure downmix does not read only one channel.
            const double sample = channel == 0 ? tone : 0;
            uint64_t raw;
            if (format == 3) {
                float value = static_cast<float>(sample);
                uint32_t word;
                std::memcpy(&word, &value, sizeof(word));
                raw = word;
            } else if (bits == 8) raw = static_cast<unsigned>(128 + sample * 128);
            else raw = static_cast<int64_t>(sample * (int64_t(1) << (bits - 1)));
            for (int byte = 0; byte < bits / 8; ++byte) stream << quint8((raw >> (8 * byte)) & 0xff);
        }
    }
    QVERIFY(file.flush());
    const auto pcm = CrispSession::readWavF32(file.fileName().toStdString());
    QVERIFY(pcm.size() >= 1599 && pcm.size() <= 1601);
    double squares = 0;
    for (size_t i = 160; i < pcm.size() - 160; ++i) {
        QVERIFY(std::isfinite(pcm[i]));
        squares += pcm[i] * pcm[i];
    }
    const double rms = std::sqrt(squares / (pcm.size() - 320));
    QVERIFY2(std::abs(rms - 0.4 / std::sqrt(2.0) / channels) < 0.01,
             qPrintable(QStringLiteral("rms=%1").arg(rms)));
}

void TestCrispAsrEngine::truncatedWavIsRejected()
{
    QTemporaryFile file;
    QVERIFY(file.open());
    file.write(QByteArray::fromHex("52494646ffffffff57415645666d7420ffffffff"));
    QVERIFY(file.flush());
    QVERIFY(CrispSession::readWavF32(file.fileName().toStdString()).empty());
}

QTEST_GUILESS_MAIN(TestCrispAsrEngine)
#include "test_crispasr_engine.moc"
