// SPDX-FileCopyrightText: 2026 Hualet Wang
// SPDX-License-Identifier: GPL-3.0-or-later

#include "../fcitx-addon/VoiceInputPolicy.h"
#include "../fcitx-addon/VoicePreedit.h"

#include <QtTest/QtTest>

class TestVoicePreedit : public QObject {
    Q_OBJECT
private slots:
    void restartedServiceCanReplaceOldBinding();
    void voiceInputEligibility_data();
    void voiceInputEligibility();
    void inlineTextIsUnderlinedWithUtf8Cursor();
    void missingInlineSupportUsesInputPanel();
    void restorePreservesExistingCompositionOnce();
    void emptyUpdateRetractsPreviewWithoutLosingOriginal();
};

void TestVoicePreedit::restartedServiceCanReplaceOldBinding()
{
    echoflow::VoiceGenerations generations;
    QVERIFY(!generations.canBegin(0));
    QVERIFY(generations.canBegin(42));
    generations.begin(42);
    QVERIFY(!generations.canBegin(42));
    QVERIFY(!generations.canBegin(41));
    QVERIFY(generations.canBegin(84));
    generations.begin(84);
    QVERIFY(!generations.canBegin(42));
}

void TestVoicePreedit::voiceInputEligibility_data()
{
    QTest::addColumn<bool>("password");
    QTest::addColumn<bool>("sensitive");
    QTest::addColumn<bool>("disabled");
    QTest::addColumn<bool>("allowed");

    QTest::newRow("normal") << false << false << false << true;
    QTest::newRow("password-only") << true << false << false << false;
    QTest::newRow("sensitive-only") << false << true << false << false;
    QTest::newRow("password-and-sensitive") << true << true << false << false;
    QTest::newRow("input-method-disabled") << false << false << true << false;
}

void TestVoicePreedit::voiceInputEligibility()
{
    QFETCH(bool, password);
    QFETCH(bool, sensitive);
    QFETCH(bool, disabled);
    QFETCH(bool, allowed);

    fcitx::CapabilityFlags capabilities = fcitx::CapabilityFlag::Preedit;
    if (password) capabilities |= fcitx::CapabilityFlag::Password;
    if (sensitive) capabilities |= fcitx::CapabilityFlag::Sensitive;
    if (disabled) capabilities |= fcitx::CapabilityFlag::Disable;
    QCOMPARE(echoflow::isVoiceInputAllowed(capabilities), allowed);
}

void TestVoicePreedit::inlineTextIsUnderlinedWithUtf8Cursor()
{
    fcitx::InputPanel panel(nullptr);
    echoflow::VoicePreedit voice;
    voice.begin(panel);
    const std::string text = "实时文本";
    voice.update(panel, text, true);
    QCOMPARE(QString::fromStdString(panel.clientPreedit().toString()), QStringLiteral("实时文本"));
    QCOMPARE(panel.clientPreedit().cursor(), static_cast<int>(text.size()));
    QVERIFY(panel.clientPreedit().formatAt(0).test(fcitx::TextFormatFlag::Underline));
    QVERIFY(panel.preedit().empty());
}

void TestVoicePreedit::missingInlineSupportUsesInputPanel()
{
    fcitx::InputPanel panel(nullptr);
    echoflow::VoicePreedit voice;
    voice.begin(panel);
    voice.update(panel, "预览", false);
    QVERIFY(panel.clientPreedit().empty());
    QCOMPARE(QString::fromStdString(panel.preedit().toString()), QStringLiteral("预览"));
    QVERIFY(panel.preedit().formatAt(0).test(fcitx::TextFormatFlag::Underline));
    voice.update(panel, "行内", true);
    QVERIFY(panel.preedit().empty());
    QCOMPARE(QString::fromStdString(panel.clientPreedit().toString()), QStringLiteral("行内"));
}

void TestVoicePreedit::restorePreservesExistingCompositionOnce()
{
    fcitx::InputPanel panel(nullptr);
    panel.setClientPreedit(fcitx::Text("existing inline"));
    panel.setPreedit(fcitx::Text("existing panel"));
    echoflow::VoicePreedit voice;
    voice.begin(panel);
    voice.update(panel, "voice preview", true);
    QVERIFY(voice.restore(panel));
    QCOMPARE(QString::fromStdString(panel.clientPreedit().toString()), QStringLiteral("existing inline"));
    QCOMPARE(QString::fromStdString(panel.preedit().toString()), QStringLiteral("existing panel"));
    QVERIFY(!voice.restore(panel));
    QCOMPARE(QString::fromStdString(panel.clientPreedit().toString()), QStringLiteral("existing inline"));
}

void TestVoicePreedit::emptyUpdateRetractsPreviewWithoutLosingOriginal()
{
    for (bool inlinePreedit : {false, true}) {
        fcitx::InputPanel panel(nullptr);
        panel.setClientPreedit(fcitx::Text("original composition"));
        echoflow::VoicePreedit voice;
        voice.begin(panel);
        voice.update(panel, "noise", inlinePreedit);
        voice.update(panel, "", inlinePreedit);
        QVERIFY(panel.clientPreedit().empty());
        QVERIFY(panel.preedit().empty());
        voice.update(panel, "valid speech", inlinePreedit);
        QVERIFY(voice.restore(panel));
        QCOMPARE(QString::fromStdString(panel.clientPreedit().toString()),
                 QStringLiteral("original composition"));
    }
}

QTEST_GUILESS_MAIN(TestVoicePreedit)
#include "test_voice_preedit.moc"
