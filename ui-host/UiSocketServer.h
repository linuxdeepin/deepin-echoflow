// SPDX-FileCopyrightText: 2026 Hualet Wang
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef ECHOFLOW_UI_SOCKET_SERVER_H
#define ECHOFLOW_UI_SOCKET_SERVER_H

#include <QObject>
#include <QString>
#include <memory>

class QSocketNotifier;
class CapsuleController;

class UiSocketServer final : public QObject {
    Q_OBJECT
public:
    UiSocketServer(QString path, CapsuleController* controller, QObject* parent = nullptr);
    ~UiSocketServer() override;
    bool start(QString* error);

signals:
    void modelRequired();

private:
    bool bindSocket(const QString& path, int& fd, bool& bound, QString* error);
    void readPending();
    void readAudioPending();
    void discardAudioPending();
    void applyMessage(const QString& message);

    QString path_;
    QString audioPath_;
    CapsuleController* controller_;
    int fd_ = -1;
    int audioFd_ = -1;
    bool bound_ = false;
    bool audioBound_ = false;
    std::unique_ptr<QSocketNotifier> notifier_;
    std::unique_ptr<QSocketNotifier> audioNotifier_;
};

#endif
