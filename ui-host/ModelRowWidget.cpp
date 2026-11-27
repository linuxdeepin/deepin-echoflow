// SPDX-FileCopyrightText: 2026 Hualet Wang
// SPDX-License-Identifier: GPL-3.0-or-later

#include "ModelRowWidget.h"

#include "EchoFlowSettings.h"
#include "ModelDownloadCoordinator.h"
#include "ModelUninstallGuard.h"

#include <QFile>
#include <QFileInfo>
#include <QHBoxLayout>
#include <QLabel>
#include <QPushButton>
#include <QSettings>
#include <QStyle>
#include <QToolButton>

#include <DDialog>
#include <DSettings>
#include <DSettingsOption>
#include <DWarningButton>

namespace echoflow {

namespace {

// Read the mirror setting live (so changing the combobox takes effect on the
// NEXT start; an in-flight download keeps the mirror it was started with).
QString baseUrlFromMirror() {
    auto* ds = EchoFlowSettings::instance()->dsettings();
    QString mirror = QStringLiteral("hf-mirror");
    if (ds) {
        auto opt = ds->option(QStringLiteral("basic.model.mirror"));
        if (opt) {
            const QVariant v = opt->value();
            if (v.isValid() && !v.toString().isEmpty()) {
                mirror = v.toString();
            }
        }
    }
    return mirror == QLatin1String("official")
               ? QStringLiteral("https://huggingface.co")
               : QStringLiteral("https://hf-mirror.com");
}

}  // namespace

ModelRowWidget::ModelRowWidget(const ModelEntry* entry, const QString& controlPath, QWidget* parent)
    : QWidget(parent), entry_(entry), controlPath_(controlPath.toStdString())
{
    auto* lay = new QHBoxLayout(this);
    lay->setContentsMargins(0, 0, 0, 0);
    status_ = new QLabel(this);
    button_ = new QPushButton(tr("Download"), this);
    cancelButton_ = new QToolButton(this);
    cancelButton_->setIcon(style()->standardIcon(QStyle::SP_DialogCancelButton));
    cancelButton_->setToolTip(tr("Cancel"));
    cancelButton_->setAccessibleName(tr("Cancel"));
    cancelButton_->hide();
    uninstallButton_ = new Dtk::Widget::DWarningButton(this);
    uninstallButton_->setText(tr("Uninstall"));
    uninstallButton_->hide();
    lay->addStretch(1);
    lay->addWidget(status_);
    lay->addWidget(cancelButton_);
    lay->addWidget(button_);
    lay->addWidget(uninstallButton_);

    connect(button_, &QPushButton::clicked, this, &ModelRowWidget::onClicked);
    connect(uninstallButton_, &QPushButton::clicked, this, &ModelRowWidget::uninstall);
    auto* c = ModelDownloadCoordinator::instance();
    connect(cancelButton_, &QToolButton::clicked, this, [this, c] {
        c->cancel(modelId());
    });
    // Both rows share the coordinator's signals; each slot filters by id.
    // Qt auto-disconnects when `this` is destroyed, so reopening the dialog
    // (which deletes the old widget) needs no manual cleanup.
    connect(c, &ModelDownloadCoordinator::progress, this,
            &ModelRowWidget::onCoordinatorProgress);
    connect(c, &ModelDownloadCoordinator::stateChanged, this,
            &ModelRowWidget::onCoordinatorStateChanged);

    refreshState();
}

QString ModelRowWidget::modelId() const {
    return entry_ ? QString::fromStdString(entry_->id) : QString();
}

void ModelRowWidget::renderProgress(qint64 done, qint64 total) {
    if (total > 0) {
        const int pct = static_cast<int>((done * 100) / total);
        status_->setText(QString::number(qBound(0, pct, 100)) + QStringLiteral("%"));
    } else {
        // Indeterminate (no Content-Length): show downloaded megabytes.
        const double mb = done / (1024.0 * 1024.0);
        status_->setText(QString("%1 MB").arg(mb, 0, 'f', 1));
    }
}

void ModelRowWidget::refreshState() {
    if (!entry_) {
        button_->setEnabled(false);
        return;
    }
    const auto snapshot = ModelDownloadCoordinator::instance()->snapshot(modelId());
    const bool downloading = snapshot.state == DownloadState::Downloading;
    cancelButton_->setVisible(downloading);
    if (downloading) {
        uninstallButton_->hide();
        button_->show();
        button_->setEnabled(false);
        renderProgress(snapshot.done, snapshot.total);
        return;
    }
    const auto dir = EchoFlowSettings::instance()->modelDirectory(*entry_);
    const bool present = isModelPresent(dir, *entry_);
    uninstallButton_->setVisible(present);
    button_->setVisible(!present);
    button_->setEnabled(!present);
    status_->setText(!present && snapshot.state == DownloadState::Failed ? snapshot.error : QString());
}

const ModelEntry* ModelRowWidget::installedReplacement() const {
    const auto &catalog = findPolishModel(entry_->id) ? polishModelCatalog() : modelCatalog();
    auto *settings = EchoFlowSettings::instance();
    auto *coordinator = ModelDownloadCoordinator::instance();
    for (const ModelEntry &entry : catalog) {
        if (entry.id != entry_->id && isModelPresent(settings->modelDirectory(entry), entry)
            && coordinator->snapshot(QString::fromStdString(entry.id)).state
                != DownloadState::Downloading) {
            return &entry;
        }
    }
    return nullptr;
}

void ModelRowWidget::uninstall() {
    if (!entry_) {
        return;
    }
    auto *settings = EchoFlowSettings::instance();
    const auto dir = settings->modelDirectory(*entry_);
    if (!isModelPresent(dir, *entry_) || ModelDownloadCoordinator::instance()->snapshot(modelId()).state
        == DownloadState::Downloading) {
        refreshState();
        return;
    }
    const bool polishModel = findPolishModel(entry_->id) != nullptr;
    if (!polishModel && !installedReplacement()) {
        status_->setText(tr("At least one model must remain installed."));
        return;
    }

    Dtk::Widget::DDialog confirm(tr("Uninstall model?"),
        tr("The files for %1 will be removed from this device.")
            .arg(QString::fromStdString(entry_->displayName)), window());
    confirm.setTextFormat(Qt::PlainText);
    confirm.addButton(tr("Cancel"), true);
    const int uninstallIndex = confirm.addButton(tr("Uninstall"), false,
                                                 Dtk::Widget::DDialog::ButtonWarning);
    if (confirm.exec() != uninstallIndex) {
        return;
    }

    // The modal dialog runs an event loop. Recheck the replacement before removing files.
    const ModelEntry *replacement = installedReplacement();
    if (!polishModel && !replacement) {
        status_->setText(tr("At least one model must remain installed."));
        return;
    }
    if (ModelDownloadCoordinator::instance()->snapshot(modelId()).state == DownloadState::Downloading) {
        return;
    }

    ModelUseLock lock(controlPath_);
    const auto result = acquireModelUninstallLock(controlPath_, lock);
    if (result != ModelUseLock::Result::Acquired) {
        status_->setText(result == ModelUseLock::Result::Busy
            ? tr("Models are in use. Try again when voice input is idle.")
            : tr("Could not release the models. Restart EchoFlow and try again."));
        return;
    }

    if (QFileInfo(QString::fromStdString(dir.string())).isSymLink()) {
        status_->setText(tr("Could not uninstall the model."));
        return;
    }

    auto *selection = settings->modelSelection(*entry_);
    const auto previousSelection = selection->value();
    const bool removingLastPolishModel = polishModel && !replacement;
    QString nextModel = previousSelection.toString();
    if (removingLastPolishModel) {
        nextModel = QStringLiteral("");
    } else if (nextModel == modelId()) {
        nextModel = QString::fromStdString(replacement->id);
    }
    const auto polishEnabled = settings->dsettings()->option(QStringLiteral("basic.polish.enabled"));
    const auto previousPolishEnabled = polishEnabled->value();
    const auto restoreSettings = [&] {
        selection->setValue(previousSelection);
        if (removingLastPolishModel) {
            polishEnabled->setValue(previousPolishEnabled);
        }
        settings->sync();
    };
    selection->setValue(nextModel);
    if (removingLastPolishModel) {
        polishEnabled->setValue(false);
    }
    settings->sync();
    // A second QSettings can still see pending values after a failed write.
    QSettings saved(settings->configPath(), QSettings::IniFormat);
    saved.sync();
    const QVariant savedModel = saved.value(selection->key() + QStringLiteral("/value"),
                                            selection->defaultValue());
    const QVariant savedPolishEnabled = saved.value(polishEnabled->key() + QStringLiteral("/value"),
                                                    polishEnabled->defaultValue());
    if (saved.status() != QSettings::NoError || savedModel.toString() != nextModel
        || (removingLastPolishModel && savedPolishEnabled.toBool())) {
        restoreSettings();
        status_->setText(tr("Could not save the model settings."));
        return;
    }

    bool removed = true;
    for (const auto &file : entry_->files) {
        const QString path = QString::fromStdString((dir / file).string());
        if (QFileInfo::exists(path) && !QFile::remove(path)) {
            removed = false;
            break;
        }
    }
    if (!removed && isModelPresent(dir, *entry_)) {
        restoreSettings();
    }
    settings->refreshModelNameItems();
    refreshState();
    if (!removed) {
        status_->setText(tr("Could not uninstall the model."));
    }
}

void ModelRowWidget::onClicked() {
    if (!entry_) {
        return;
    }
    auto* c = ModelDownloadCoordinator::instance();
    if (c->snapshot(modelId()).state == DownloadState::Downloading) {
        return;
    }
    const auto modelDir = EchoFlowSettings::instance()->modelDirectory(*entry_);
    const QString dir = QString::fromStdString(modelDir.string());
    c->start(*entry_, dir, baseUrlFromMirror());
}

void ModelRowWidget::onCoordinatorProgress(const QString& id, qint64 done, qint64 total,
                                           const QString& /*file*/) {
    if (id != modelId()) {
        return;
    }
    renderProgress(done, total);
}

void ModelRowWidget::onCoordinatorStateChanged(const QString& id, DownloadState state) {
    if (id != modelId()) {
        return;
    }
    refreshState();
    if (state == DownloadState::Succeeded) {
        EchoFlowSettings::instance()->refreshModelNameItems();
    }
}

}  // namespace echoflow
