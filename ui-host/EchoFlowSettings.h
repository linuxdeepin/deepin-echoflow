// SPDX-FileCopyrightText: 2026 Hualet Wang
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <filesystem>

#include <QObject>
#include <QMap>
#include <QString>
#include <QVariant>

namespace Dtk {
namespace Core {
class DSettings;
class DSettingsOption;
class QSettingBackend;
}
} // namespace Dtk

namespace echoflow {

struct ModelEntry;

class EchoFlowSettings : public QObject {
    Q_OBJECT
public:
    static EchoFlowSettings *instance();

    bool init(const QString &configPath);
    QString configPath() const;
    Dtk::Core::DSettings *dsettings() const;
    void sync();

    // Rebuild model choices from disk and keep the current recognition selection visible.
    void refreshModelNameItems();
    Dtk::Core::DSettingsOption *modelSelection(const ModelEntry &entry) const;
    std::filesystem::path modelDirectory(const ModelEntry &entry) const;

private:
    explicit EchoFlowSettings(QObject *parent = nullptr);
    ~EchoFlowSettings() override;

    void populateComboBoxes();
    static void setComboBoxItems(Dtk::Core::DSettings *settings,
                                 const QString &path,
                                 const QStringList &items);
    static void setComboBoxItems(Dtk::Core::DSettings *settings,
                                 const QString &path,
                                 const QStringList &keys,
                                 const QStringList &values);

    Dtk::Core::DSettings *dsettings_ = nullptr;
    Dtk::Core::QSettingBackend *backend_ = nullptr;
    QString configPath_;
};

} // namespace echoflow
