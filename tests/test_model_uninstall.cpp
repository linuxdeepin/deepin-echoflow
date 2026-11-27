// SPDX-FileCopyrightText: 2026 Hualet Wang
// SPDX-License-Identifier: GPL-3.0-or-later

#include "EchoFlowSettings.h"
#include "ModelRowWidget.h"
#include "ModelUseLock.h"
#include "SettingsDialog.h"

#include <DDialog>
#include <DSettings>
#include <DSettingsOption>
#include <QApplication>
#include <QCheckBox>
#include <QComboBox>
#include <QDir>
#include <QFile>
#include <QLabel>
#include <QPushButton>
#include <QSettings>
#include <QTemporaryDir>
#include <QTimer>
#include <QToolButton>
#include <QtTest>

#include <fstream>

using namespace echoflow;

class TestModelUninstall : public QObject {
    Q_OBJECT

private slots:
    void initTestCase();
    void downloadButtonKeepsItsLabel();
    void polishRequiresInstalledModel();
    void uninstallSelectedModel_data();
    void uninstallSelectedModel();

private:
    QTemporaryDir configDirectory_;
};

void TestModelUninstall::initTestCase() {
    QVERIFY(configDirectory_.isValid());
    QVERIFY(EchoFlowSettings::instance()->init(configDirectory_.filePath(QStringLiteral("config.ini"))));
}

void TestModelUninstall::downloadButtonKeepsItsLabel() {
    const ModelEntry entry{"test-model-download", "Test model", "test/model", {"model.gguf"}};
    ModelRowWidget row(&entry, configDirectory_.filePath(QStringLiteral("control.sock")));
    row.show();
    const auto buttons = row.findChildren<QPushButton*>(QString(), Qt::FindDirectChildrenOnly);
    const auto visibleButton = [&buttons]() -> QPushButton* {
        for (auto *button : buttons) {
            if (button->isVisible()) {
                return button;
            }
        }
        return nullptr;
    };
    auto *button = visibleButton();
    QVERIFY(button);
    QCOMPARE(button->text(), QStringLiteral("Download"));
    QVERIFY(button->isEnabled());
    auto *cancelButton = row.findChild<QToolButton*>();
    QVERIFY(cancelButton && cancelButton->isHidden());

    auto *coordinator = ModelDownloadCoordinator::instance();
    const auto directory = EchoFlowSettings::instance()->modelDirectory(entry);
    const auto path = QString::fromStdString(directory.string());
    const auto invalidMirror = QStringLiteral("unsupported-scheme://localhost");
    coordinator->start(entry, path, invalidMirror);
    QCOMPARE(visibleButton(), button);
    QCOMPARE(button->text(), QStringLiteral("Download"));
    QVERIFY(!button->isEnabled());
    QVERIFY(cancelButton->isVisible());
    QVERIFY(cancelButton->text().isEmpty());
    cancelButton->click();
    QVERIFY(cancelButton->isHidden());
    QVERIFY(button->isEnabled());
    coordinator->start(entry, path, invalidMirror);
    QTRY_VERIFY(button->isEnabled());
    QCOMPARE(visibleButton(), button);
    QCOMPARE(button->text(), QStringLiteral("Download"));
    const auto *status = row.findChild<QLabel*>();
    QVERIFY(status && !status->text().isEmpty());

    std::ofstream(directory / entry.files.front()).put('x');
    coordinator->start(entry, path, invalidMirror);
    QVERIFY(visibleButton());
    QCOMPARE(visibleButton()->text(), QStringLiteral("Uninstall"));
}

void TestModelUninstall::polishRequiresInstalledModel() {
    auto *settings = EchoFlowSettings::instance();
    const auto enabled = settings->dsettings()->option(QStringLiteral("basic.polish.enabled"));
    QVERIFY(enabled);
    enabled->setValue(true);
    settings->refreshModelNameItems();
    QVERIFY(!enabled->value().toBool());

    SettingsDialog dialog(settings->dsettings(), configDirectory_.filePath(QStringLiteral("control.sock")));
    auto *checkBox = dialog.findChild<QCheckBox*>(QStringLiteral("polishEnabledCheckBox"));
    QVERIFY(checkBox);
    QVERIFY(!checkBox->text().isEmpty());
    QVERIFY(!checkBox->isEnabled());
    checkBox->click();
    QVERIFY(!enabled->value().toBool());

    const auto &entry = polishModelCatalog().front();
    const auto directory = settings->modelDirectory(entry);
    std::filesystem::create_directories(directory);
    for (const auto &file : entry.files) {
        std::ofstream(directory / file).put('x');
    }
    // The coordinator reports success immediately when the target files exist.
    ModelDownloadCoordinator::instance()->start(entry, QString::fromStdString(directory.string()), {});
    QVERIFY(checkBox->isEnabled());
    QVERIFY(!checkBox->isChecked());
    checkBox->click();
    QVERIFY(enabled->value().toBool());
    checkBox->click();
    QVERIFY(!enabled->value().toBool());

    for (const auto &file : entry.files) {
        QVERIFY(std::filesystem::remove(directory / file));
    }
    checkBox->click();
    QVERIFY(!checkBox->isEnabled());
    QVERIFY(!checkBox->isChecked());
    QVERIFY(!enabled->value().toBool());
}

void TestModelUninstall::uninstallSelectedModel_data() {
    QTest::addColumn<bool>("polish");
    QTest::newRow("recognition") << false;
    QTest::newRow("polish") << true;
}

void TestModelUninstall::uninstallSelectedModel() {
    QFETCH(bool, polish);
    const auto &catalog = polish ? polishModelCatalog() : modelCatalog();
    const auto &selected = catalog[0];
    const auto &replacement = catalog[1];
    auto *settings = EchoFlowSettings::instance();
    auto *option = settings->modelSelection(selected);
    QVERIFY(option);
    option->setValue(QString::fromStdString(selected.id));
    const auto selectedDirectory = settings->modelDirectory(selected);
    std::filesystem::create_directories(selectedDirectory);
    for (const auto &file : selected.files) {
        std::ofstream(selectedDirectory / file).put('x');
    }
    const auto unrelatedFile = selectedDirectory / "notes.txt";
    std::ofstream(unrelatedFile).put('x');

    const QString controlPath = configDirectory_.filePath(QStringLiteral("control.sock"));
    ModelRowWidget row(&selected, controlPath);
    row.show();
    QPushButton *uninstallButton = nullptr;
    for (auto *button : row.findChildren<QPushButton*>()) {
        if (button->text() == QStringLiteral("Uninstall")) {
            uninstallButton = button;
        }
    }
    QVERIFY(uninstallButton && uninstallButton->isVisible());
    const auto *status = row.findChild<QLabel*>();
    QVERIFY(status);
    if (!polish) {
        uninstallButton->click();
        QVERIFY(isModelPresent(selectedDirectory, selected));
        QCOMPARE(status->text(), QStringLiteral("At least one model must remain installed."));
    }

    const auto replacementDirectory = settings->modelDirectory(replacement);
    std::filesystem::create_directories(replacementDirectory);
    for (const auto &file : replacement.files) {
        std::ofstream(replacementDirectory / file).put('x');
    }
    settings->refreshModelNameItems();
    SettingsDialog settingsDialog(settings->dsettings(), controlPath);
    const QString selectedId = QString::fromStdString(selected.id);
    const QString replacementId = QString::fromStdString(replacement.id);
    QComboBox *modelPicker = nullptr;
    for (auto *combo : settingsDialog.findChildren<QComboBox*>()) {
        if (combo->findData(selectedId) >= 0 && combo->findData(replacementId) >= 0) {
            modelPicker = combo;
            break;
        }
    }
    QVERIFY(modelPicker);
    modelPicker->setCurrentIndex(modelPicker->findData(replacementId));
    QVERIFY(uninstallButton->isVisible());

    const auto respondToConfirmation = [](int index) {
        QTimer::singleShot(0, [index] {
            auto *dialog = qobject_cast<Dtk::Widget::DDialog*>(QApplication::activeModalWidget());
            QVERIFY(dialog);
            auto *button = dialog->getButton(index);
            QVERIFY(button);
            button->click();
        });
    };
    respondToConfirmation(1);
    uninstallButton->click();
    QVERIFY(!isModelPresent(selectedDirectory, selected));
    QCOMPARE(option->value().toString(), replacementId);
    for (const auto &file : selected.files) {
        std::ofstream(selectedDirectory / file).put('x');
    }
    settings->refreshModelNameItems();
    ModelRowWidget selectedRow(&selected, controlPath);
    selectedRow.show();
    for (auto *button : selectedRow.findChildren<QPushButton*>()) {
        if (button->text() == QStringLiteral("Uninstall")) {
            uninstallButton = button;
        }
    }
    status = selectedRow.findChild<QLabel*>();
    QVERIFY(status && uninstallButton->isVisible());
    modelPicker->setCurrentIndex(modelPicker->findData(selectedId));

    respondToConfirmation(0);
    uninstallButton->click();
    QVERIFY(isModelPresent(selectedDirectory, selected));
    QCOMPARE(option->value().toString(), QString::fromStdString(selected.id));

    {
        ModelUseLock startupLock(controlPath.toStdString());
        QCOMPARE(startupLock.lockForUse(), ModelUseLock::Result::Acquired);
        respondToConfirmation(1);
        uninstallButton->click();
        QVERIFY(isModelPresent(selectedDirectory, selected));
        QCOMPARE(option->value().toString(), selectedId);
        QCOMPARE(status->text(), QStringLiteral("Models are in use. Try again when voice input is idle."));
    }
    // Make the config path unwritable even when tests run as root. The model
    // directories remain writable and QSettings still has cached values.
    const QString configPath = settings->configPath();
    const QString configBackup = configPath + QStringLiteral(".backup");
    QVERIFY(QFile::rename(configPath, configBackup));
    QVERIFY(QDir().mkdir(configPath));
    respondToConfirmation(1);
    uninstallButton->click();
    QVERIFY(QDir().rmdir(configPath));
    QVERIFY(QFile::rename(configBackup, configPath));
    QCOMPARE(status->text(), QStringLiteral("Could not save the model settings."));
    QVERIFY(isModelPresent(selectedDirectory, selected));
    QCOMPARE(option->value().toString(), selectedId);
    QCOMPARE(modelPicker->currentData().toString(), selectedId);

    respondToConfirmation(1);
    uninstallButton->click();
    QVERIFY(!isModelPresent(selectedDirectory, selected));
    QVERIFY(isModelPresent(replacementDirectory, replacement));
    QVERIFY(std::filesystem::exists(unrelatedFile));
    QVERIFY(uninstallButton->isHidden());
    QCOMPARE(option->value().toString(), replacementId);
    QCOMPARE(modelPicker->currentData().toString(), replacementId);
    QCOMPARE(modelPicker->findData(selectedId), -1);
    QCOMPARE(option->data(QStringLiteral("items")).toMap().value(QStringLiteral("keys")).toStringList(),
             QStringList{replacementId});
    const auto streamOption = settings->dsettings()->option(
        QStringLiteral("basic.recognition.stream_transcription"));
    QVERIFY(streamOption);
    streamOption->setValue(!streamOption->defaultValue().toBool());
    settings->dsettings()->reset();
    settings->sync();
    QCOMPARE(streamOption->value(), streamOption->defaultValue());
    QCOMPARE(option->value().toString(), replacementId);
    QCOMPARE(modelPicker->currentData().toString(), replacementId);
    QTRY_COMPARE(QSettings(configPath, QSettings::IniFormat)
                     .value(option->key() + QStringLiteral("/value")).toString(), replacementId);

    ModelRowWidget lastRow(&replacement, controlPath);
    lastRow.show();
    QPushButton *lastUninstallButton = nullptr;
    for (auto *button : lastRow.findChildren<QPushButton*>()) {
        if (button->text() == QStringLiteral("Uninstall")) {
            lastUninstallButton = button;
        }
    }
    QVERIFY(lastUninstallButton && lastUninstallButton->isVisible());
    if (polish) {
        const auto enabled = settings->dsettings()->option(QStringLiteral("basic.polish.enabled"));
        QVERIFY(enabled);
        auto *checkBox = settingsDialog.findChild<QCheckBox*>(QStringLiteral("polishEnabledCheckBox"));
        QVERIFY(checkBox && checkBox->isEnabled());
        checkBox->click();
        QVERIFY(enabled->value().toBool());
        respondToConfirmation(1);
        lastUninstallButton->click();
        QVERIFY(!isModelPresent(replacementDirectory, replacement));
        QVERIFY(option->value().toString().isEmpty());
        QVERIFY(!enabled->value().toBool());
        QVERIFY(!checkBox->isEnabled());
        QVERIFY(!checkBox->isChecked());
        QCOMPARE(modelPicker->count(), 0);
        const QSettings saved(configPath, QSettings::IniFormat);
        QCOMPARE(saved.value(enabled->key() + QStringLiteral("/value")), QVariant(false));
        QCOMPARE(saved.value(option->key() + QStringLiteral("/value"), replacementId).toString(),
                 QStringLiteral(""));
    } else {
        lastUninstallButton->click();
        QVERIFY(isModelPresent(replacementDirectory, replacement));
        QCOMPARE(option->value().toString(), replacementId);
    }
}

QTEST_MAIN(TestModelUninstall)
#include "test_model_uninstall.moc"
