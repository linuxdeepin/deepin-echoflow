// SPDX-FileCopyrightText: 2026 Hualet Wang
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef ECHOFLOW_MODEL_ROW_WIDGET_H
#define ECHOFLOW_MODEL_ROW_WIDGET_H

#include "ModelCatalog.h"
#include "ModelDownloadCoordinator.h"

#include <QWidget>

class QLabel;
class QPushButton;
class QToolButton;

namespace Dtk::Widget {
class DWarningButton;
}

namespace echoflow {

// Settings row for one model's download and removal. Owns no downloader; it queries
// ModelDownloadCoordinator for the current snapshot and connects to its
// id-tagged signals. Constructed fresh every time the (destroy-on-close)
// settings dialog opens, so it always reflects live progress if a download is
// in flight, or fresh disk state otherwise.
class ModelRowWidget : public QWidget {
    Q_OBJECT
public:
    ModelRowWidget(const ModelEntry* entry, const QString& controlPath, QWidget* parent = nullptr);

private slots:
    void onClicked();
    void uninstall();
    void onCoordinatorProgress(const QString& id, qint64 done, qint64 total, const QString& file);
    void onCoordinatorStateChanged(const QString& id, DownloadState state);

private:
    void refreshState();
    const ModelEntry* installedReplacement() const;
    void renderProgress(qint64 done, qint64 total);
    QString modelId() const;

    const ModelEntry* entry_;
    const std::filesystem::path controlPath_;
    QLabel* status_ = nullptr;
    QPushButton* button_ = nullptr;
    QToolButton* cancelButton_ = nullptr;
    Dtk::Widget::DWarningButton* uninstallButton_ = nullptr;
};

}  // namespace echoflow

#endif  // ECHOFLOW_MODEL_ROW_WIDGET_H
