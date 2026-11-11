// SPDX-FileCopyrightText: 2026 Hualet Wang
// SPDX-License-Identifier: GPL-3.0-or-later

#include <QtTest/QtTest>

#include "CrispAsrEngine.h"

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
    cfg.crispModelPath = "/tmp/echoflow-no-such-model.gguf";
    CrispAsrEngine engine(cfg);
    try {
        QCOMPARE(QString::fromStdString(engine.transcribe("/tmp/echoflow-no-such-audio.wav")),
                 QString());
    } catch (const std::exception& e) {
        QFAIL(qPrintable(QStringLiteral("transcribe threw: %1").arg(e.what())));
    }
}

QTEST_GUILESS_MAIN(TestCrispAsrEngine)
#include "test_crispasr_engine.moc"
