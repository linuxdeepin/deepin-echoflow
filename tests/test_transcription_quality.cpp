// SPDX-FileCopyrightText: 2026 Hualet Wang
// SPDX-License-Identifier: GPL-3.0-or-later

#include "benchmarks/TranscriptionQualityMetrics.h"

#include <QTest>

using echoflow::benchmark::scored;

class TestTranscriptionQuality : public QObject {
    Q_OBJECT
private slots:
    void scoresRecognitionErrors();
    void normalizesPunctuationSpacingAndWidth();
    void countsUnicodeCodePoints();
    void absentReferenceDoesNotInventCer();
    void emptyHypothesisCountsEveryReferenceCharacter();
    void rejectsReferenceWithoutScoredCharacters();
};

void TestTranscriptionQuality::scoresRecognitionErrors()
{
    const auto wrong = scored({{"transcript", QStringLiteral("你号")}}, QStringLiteral("你好"));
    QCOMPARE(wrong.value("character_errors").toInt(), 1);
    QCOMPARE(wrong.value("reference_characters").toInt(), 2);
    QCOMPARE(wrong.value("cer").toDouble(), 0.5);
    const auto corrected = scored({{"transcript", QStringLiteral("你好")}}, QStringLiteral("你好"));
    QCOMPARE(corrected.value("cer").toDouble(), 0.0);
}

void TestTranscriptionQuality::normalizesPunctuationSpacingAndWidth()
{
    const auto row = scored({{"transcript", QStringLiteral("abc你好")}}, QStringLiteral("ＡＢＣ，你好！\n"));
    QCOMPARE(row.value("reference_characters").toInt(), 5);
    QCOMPARE(row.value("cer").toDouble(), 0.0);
}

void TestTranscriptionQuality::countsUnicodeCodePoints()
{
    const auto row = scored({{"transcript", QStringLiteral("\U00020000\U00020002")}},
                           QStringLiteral("\U00020000\U00020001"));
    QCOMPARE(row.value("reference_characters").toInt(), 2);
    QCOMPARE(row.value("character_errors").toInt(), 1);
    QCOMPARE(row.value("cer").toDouble(), 0.5);
}

void TestTranscriptionQuality::absentReferenceDoesNotInventCer()
{
    const auto row = scored({{"transcript", QStringLiteral("你好")}}, {});
    QVERIFY(!row.contains("cer"));
    QVERIFY(!row.contains("character_errors"));
}

void TestTranscriptionQuality::emptyHypothesisCountsEveryReferenceCharacter()
{
    const auto row = scored({{"transcript", ""}}, QStringLiteral("你好"));
    QCOMPARE(row.value("character_errors").toInt(), 2);
    QCOMPARE(row.value("cer").toDouble(), 1.0);
}

void TestTranscriptionQuality::rejectsReferenceWithoutScoredCharacters()
{
    QVERIFY_EXCEPTION_THROWN(scored({{"transcript", "anything"}}, QStringLiteral("！ \n")), std::runtime_error);
}

QTEST_GUILESS_MAIN(TestTranscriptionQuality)
#include "test_transcription_quality.moc"
