// SPDX-FileCopyrightText: 2026 Hualet Wang
// SPDX-License-Identifier: GPL-3.0-or-later

#include "OnboardingSetupController.h"

#include "OnboardingState.h"

#include <algorithm>
#include <QCoreApplication>
#include <QDir>

#ifndef ECHOFLOW_AUTOSTART_DESKTOP_PATH
#define ECHOFLOW_AUTOSTART_DESKTOP_PATH ""
#endif

OnboardingSetupController::OnboardingSetupController(ModelSetupSource *model,
                                                     SetupCommandRunner *runner,
                                                     OnboardingState *state,
                                                     QObject *parent)
    : QObject(parent)
    , model_(model)
    , runner_(runner)
    , state_(state)
{
    itemStates_.fill(SetupItemState::Pending);

    connect(model_, &ModelSetupSource::progress, this,
            [this](qint64 done, qint64 total) {
        progressDone_ = done;
        progressTotal_ = total;
        emit progressChanged(done, total);
    });
    connect(model_, &ModelSetupSource::finished, this,
            [this](bool ok, const QString &error) {
        if (itemState(SetupItem::Model) != SetupItemState::Running) {
            return;
        }
        setItemState(SetupItem::Model,
                     ok ? SetupItemState::Succeeded : SetupItemState::Failed,
                     ok ? QString() : error);
        if (ok && serviceStartAfterModel_) {
            serviceStartAfterModel_ = false;
            if (initialProbesRemaining_ > 0) {
                serviceReloadRequested_ = true;
            } else {
                startService(QStringLiteral("service-after-model"));
            }
        }
        checkForCompletion();
    });
    connect(runner_, &SetupCommandRunner::finished, this,
            &OnboardingSetupController::handleCommandFinished);

    if (state_->isComplete()) {
        itemStates_.fill(SetupItemState::Succeeded);
        complete_ = true;
        return;
    }
    beginReconstruction();
}

SetupItemState OnboardingSetupController::itemState(SetupItem item) const
{
    return itemStates_.at(indexOf(item));
}

QString OnboardingSetupController::itemError(SetupItem item) const
{
    return itemErrors_.at(indexOf(item));
}

qint64 OnboardingSetupController::progressDone() const
{
    return progressDone_;
}

qint64 OnboardingSetupController::progressTotal() const
{
    return progressTotal_;
}

bool OnboardingSetupController::isRunning() const
{
    return std::any_of(itemStates_.cbegin(), itemStates_.cend(),
                       [](SetupItemState state) {
        return state == SetupItemState::Running;
    });
}

bool OnboardingSetupController::isComplete() const
{
    return complete_;
}

bool OnboardingSetupController::hasStarted() const
{
    return started_;
}

QString OnboardingSetupController::aggregateError() const
{
    return aggregateError_;
}

void OnboardingSetupController::start()
{
    if (complete_) {
        return;
    }
    const bool wasStarted = started_;
    started_ = true;
    if (initialProbesRemaining_ > 0) {
        startRequested_ = true;
        return;
    }
    if (wasStarted && isRunning()) {
        return;
    }
    if (!aggregateError_.isEmpty()) {
        aggregateError_.clear();
        emit aggregateErrorChanged({});
    }

    launchPendingSetup();
}

void OnboardingSetupController::launchPendingSetup()
{
    startRequested_ = false;

    for (SetupItem item : {SetupItem::Model, SetupItem::UiAutostart,
                           SetupItem::Fcitx}) {
        if (itemState(item) == SetupItemState::Pending
            || itemState(item) == SetupItemState::Failed) {
            startItem(item);
        }
    }
    if (itemState(SetupItem::Model) == SetupItemState::Succeeded
        && (itemState(SetupItem::Service) == SetupItemState::Pending
            || itemState(SetupItem::Service) == SetupItemState::Failed)) {
        startService(QStringLiteral("service"));
    }
    checkForCompletion();
}

void OnboardingSetupController::retryFailed()
{
    if (isRunning()) {
        return;
    }
    if (!aggregateError_.isEmpty()) {
        aggregateError_.clear();
        emit aggregateErrorChanged({});
    }

    bool retried = false;
    for (SetupItem item : {SetupItem::Model, SetupItem::UiAutostart,
                           SetupItem::Fcitx}) {
        if (itemState(item) == SetupItemState::Failed) {
            startItem(item);
            retried = true;
        }
    }
    if (itemState(SetupItem::Model) == SetupItemState::Succeeded
        && (itemState(SetupItem::Service) == SetupItemState::Pending
            || itemState(SetupItem::Service) == SetupItemState::Failed)) {
        startService(QStringLiteral("service"));
        retried = true;
    }
    if (!retried) {
        checkForCompletion();
    }
}

std::size_t OnboardingSetupController::indexOf(SetupItem item)
{
    return static_cast<std::size_t>(item);
}

void OnboardingSetupController::beginReconstruction()
{
    if (model_->modelPresent()) {
        itemStates_[indexOf(SetupItem::Model)] = SetupItemState::Succeeded;
    } else if (model_->downloadRunning()) {
        itemStates_[indexOf(SetupItem::Model)] = SetupItemState::Running;
        serviceStartAfterModel_ = true;
    } else {
        serviceStartAfterModel_ = true;
    }

    itemStates_[indexOf(SetupItem::UiAutostart)] = SetupItemState::Pending;
    itemStates_[indexOf(SetupItem::Service)] = SetupItemState::Running;
    initialProbesRemaining_ = 1;
    runner_->run(QStringLiteral("service-initial-check"),
                 QStringLiteral("systemctl"),
                 {QStringLiteral("--user"), QStringLiteral("is-active"),
                  QStringLiteral("echoflow.socket")});
}

void OnboardingSetupController::startItem(SetupItem item)
{
    setItemState(item, SetupItemState::Running);
    switch (item) {
    case SetupItem::Model:
        if (model_->modelPresent()) {
            setItemState(item, SetupItemState::Succeeded);
        } else if (!model_->downloadRunning()) {
            serviceStartAfterModel_ = true;
            model_->startDownload();
        }
        break;
    case SetupItem::UiAutostart: {
        QString source = QStringLiteral(ECHOFLOW_AUTOSTART_DESKTOP_PATH);
        if (source.isEmpty()) {
            source = QDir(QCoreApplication::applicationDirPath())
                .filePath(QStringLiteral("../share/echoflow/echoflow-autostart.desktop"));
        }
        const QString destination = QDir::home().filePath(
            QStringLiteral(".config/autostart/echoflow-autostart.desktop"));
        runner_->run(QStringLiteral("ui-autostart"), QStringLiteral("install"),
                     {QStringLiteral("-D"), source, destination});
        break;
    }
    case SetupItem::Service:
        startService(QStringLiteral("service"));
        break;
    case SetupItem::Fcitx:
        runner_->run(QStringLiteral("fcitx"), QStringLiteral("fcitx5"),
                     {QStringLiteral("-rd")});
        break;
    }
}

void OnboardingSetupController::startService(const QString &id)
{
    setItemState(SetupItem::Service, SetupItemState::Running);
    runner_->run(id,
                 QDir(QCoreApplication::applicationDirPath())
                     .filePath(QStringLiteral("echoflow-session-start")),
                 {QStringLiteral("--socket-only")});
}

void OnboardingSetupController::setItemState(SetupItem item,
                                             SetupItemState state,
                                             const QString &error)
{
    const bool wasRunning = isRunning();
    const std::size_t index = indexOf(item);
    if (itemStates_[index] == state && itemErrors_[index] == error) {
        return;
    }
    itemStates_[index] = state;
    itemErrors_[index] = error;
    emit itemStateChanged(item, state, error);
    if (wasRunning != isRunning()) {
        emit runningChanged(isRunning());
    }
}

void OnboardingSetupController::handleCommandFinished(const QString &id,
                                                      bool ok,
                                                      const QString &error)
{
    SetupItem item;
    if (id == QStringLiteral("service-initial-check")) {
        item = SetupItem::Service;
        if (initialProbesRemaining_ <= 0
            || itemState(item) != SetupItemState::Running) {
            return;
        }
        setItemState(item, ok ? SetupItemState::Succeeded
                              : SetupItemState::Pending);
        --initialProbesRemaining_;
        if (initialProbesRemaining_ == 0) {
            if (serviceReloadRequested_) {
                serviceReloadRequested_ = false;
                startService(QStringLiteral("service-after-model"));
            }
            if (startRequested_) {
                launchPendingSetup();
            }
        }
        return;
    } else if (id == QStringLiteral("ui-autostart")) {
        item = SetupItem::UiAutostart;
        if (itemState(item) != SetupItemState::Running) {
            return;
        }
    } else if (id == QStringLiteral("service")
               || id == QStringLiteral("service-after-model")) {
        item = SetupItem::Service;
        if (itemState(item) != SetupItemState::Running) {
            return;
        }
        if (ok) {
            runner_->run(QStringLiteral("service-check"),
                         QStringLiteral("systemctl"),
                         {QStringLiteral("--user"), QStringLiteral("is-active"),
                          QStringLiteral("echoflow.socket")});
            return;
        }
    } else if (id == QStringLiteral("service-check")) {
        item = SetupItem::Service;
    } else if (id == QStringLiteral("fcitx")) {
        item = SetupItem::Fcitx;
    } else {
        return;
    }

    if (itemState(item) != SetupItemState::Running) {
        return;
    }
    setItemState(item, ok ? SetupItemState::Succeeded : SetupItemState::Failed,
                 ok ? QString() : error);
    checkForCompletion();
}

void OnboardingSetupController::checkForCompletion()
{
    if (complete_ || !aggregateError_.isEmpty() || !allItemsSucceeded()) {
        return;
    }

    QString error;
    if (!state_->markComplete(&error)) {
        aggregateError_ = error.isEmpty()
            ? QStringLiteral("Failed to save onboarding completion")
            : error;
        emit aggregateErrorChanged(aggregateError_);
        emit setupFailed(aggregateError_);
        return;
    }
    complete_ = true;
    emit setupComplete();
}

bool OnboardingSetupController::allItemsSucceeded() const
{
    return std::all_of(itemStates_.cbegin(), itemStates_.cend(),
                       [](SetupItemState state) {
        return state == SetupItemState::Succeeded;
    });
}
