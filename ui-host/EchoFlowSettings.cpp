// SPDX-FileCopyrightText: 2026 Hualet Wang
// SPDX-License-Identifier: GPL-3.0-or-later

#include "EchoFlowSettings.h"

#include "ModelCatalog.h"

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QList>
#include <QMap>
#include <QProcess>
#include <QSettings>
#include <QStandardPaths>
#include <QStringList>
#include <QVariant>

#include <DSettings>
#include <DSettingsOption>
#include <qsettingbackend.h>

namespace echoflow {

namespace {

const QString kRecognitionModel = QStringLiteral("basic.model.model_name");
const QString kPolishModel = QStringLiteral("basic.polish.model_name");
const QString kPolishEnabled = QStringLiteral("basic.polish.enabled");

struct AudioSourceItem {
    QString name;
    QString description;
    bool available = true;
};

QString sourceDisplayName(const AudioSourceItem &source, const QMap<QString, int> &descriptionCounts) {
    QString description = source.description;
    const QString controllerPrefix =
        QStringLiteral("Family 17h/19h HD Audio Controller ");
    if (description.startsWith(controllerPrefix)) {
        description = description.mid(controllerPrefix.size());
    }

    if (descriptionCounts.value(source.description) <= 1) {
        return description;
    }
    return description + QStringLiteral(" (") + source.name + QStringLiteral(")");
}

QString fallbackSourceDescription(const QString &name) {
    if (name.contains(QStringLiteral("__Mic1__source"))) {
        return QStringLiteral("Digital Microphone");
    }
    if (name.contains(QStringLiteral("__Mic2__source"))) {
        return QStringLiteral("Stereo Microphone");
    }
    return name;
}

QList<AudioSourceItem> pipeWireSourcesFromShortList() {
    QProcess process;
    process.start(QStringLiteral("pactl"),
                  QStringList{QStringLiteral("list"), QStringLiteral("sources"), QStringLiteral("short")});
    if (!process.waitForFinished(3000) || process.exitStatus() != QProcess::NormalExit
        || process.exitCode() != 0) {
        return {};
    }

    QList<AudioSourceItem> sources;
    const QString output = QString::fromLocal8Bit(process.readAllStandardOutput());
    const QStringList lines = output.split(QLatin1Char('\n'), Qt::SkipEmptyParts);
    for (const QString &line : lines) {
        const QStringList fields = line.split(QLatin1Char('\t'));
        if (fields.size() < 2 || fields.at(1).isEmpty()
            || fields.at(1).endsWith(QStringLiteral(".monitor"))) {
            continue;
        }
        const QString name = fields.at(1);
        sources << AudioSourceItem{name, fallbackSourceDescription(name)};
    }
    return sources;
}

QList<AudioSourceItem> pipeWireSources() {
    QProcess process;
    process.start(QStringLiteral("pactl"),
                  QStringList{QStringLiteral("list"), QStringLiteral("sources")});
    if (!process.waitForFinished(3000) || process.exitStatus() != QProcess::NormalExit
        || process.exitCode() != 0) {
        return pipeWireSourcesFromShortList();
    }

    QList<AudioSourceItem> sources;
    AudioSourceItem current;
    const QString output = QString::fromLocal8Bit(process.readAllStandardOutput());
    const QStringList lines = output.split(QLatin1Char('\n'), Qt::SkipEmptyParts);
    for (const QString &line : lines) {
        if (line.startsWith(QStringLiteral("Source #"))) {
            if (!current.name.isEmpty() && !current.name.endsWith(QStringLiteral(".monitor"))
                && current.available) {
                if (current.description.isEmpty()) {
                    current.description = current.name;
                }
                sources << current;
            }
            current = {};
            continue;
        }

        const QString trimmed = line.trimmed();
        if (trimmed.startsWith(QStringLiteral("Name: "))) {
            current.name = trimmed.mid(QStringLiteral("Name: ").size());
        } else if (trimmed.startsWith(QStringLiteral("Description: "))) {
            current.description = trimmed.mid(QStringLiteral("Description: ").size());
        } else if (trimmed.startsWith(QStringLiteral("[In] "))
                   && trimmed.contains(QStringLiteral("not available"))) {
            current.available = false;
        }
    }
    if (!current.name.isEmpty() && !current.name.endsWith(QStringLiteral(".monitor"))
        && current.available) {
        if (current.description.isEmpty()) {
            current.description = current.name;
        }
        sources << current;
    }

    if (sources.empty()) {
        return pipeWireSourcesFromShortList();
    }
    return sources;
}

} // namespace

EchoFlowSettings *EchoFlowSettings::instance() {
    static EchoFlowSettings instance;
    return &instance;
}

EchoFlowSettings::EchoFlowSettings(QObject *parent) : QObject(parent) {}

EchoFlowSettings::~EchoFlowSettings() = default;

bool EchoFlowSettings::init(const QString &configPath) {
    QFile schemaFile(QStringLiteral(":/settings/settings-schema.json"));
    if (!schemaFile.open(QIODevice::ReadOnly)) {
        return false;
    }

    dsettings_ = Dtk::Core::DSettings::fromJson(schemaFile.readAll());
    if (!dsettings_) {
        return false;
    }

    populateComboBoxes();

    configPath_ = configPath;
    QDir().mkpath(QFileInfo(configPath_).path());

    if (!QFile::exists(configPath_)) {
        const QStringList paths = {
            QStringLiteral("basic.model.model_name"),
            QStringLiteral("basic.recognition.language"),
            QStringLiteral("basic.recognition.stream_transcription"),
            QStringLiteral("basic.recognition.noise_suppression"),
            QStringLiteral("basic.recognition.stream_step_ms"),
            kPolishEnabled,
            QStringLiteral("basic.polish.model_name"),
            QStringLiteral("basic.recording.min_record_seconds"),
            QStringLiteral("basic.recording.rate"),
            QStringLiteral("basic.recording.channels"),
            QStringLiteral("basic.recording.format"),
            QStringLiteral("basic.recording.source"),
            QStringLiteral("basic.interaction.focus_hint"),
            QStringLiteral("basic.model.mirror"),
            QStringLiteral("advanced.runtime.asr_timeout_seconds"),
            QStringLiteral("advanced.storage.recordings_dir"),
            QStringLiteral("advanced.storage.save_live_debug_audio"),
        };
        QSettings defaultSettings(configPath_, QSettings::IniFormat);
        for (const QString &path : paths) {
            Dtk::Core::DSettingsOption *option = dsettings_->option(path);
            if (option) {
                defaultSettings.setValue(path + QStringLiteral("/value"), option->defaultValue());
            }
        }
        defaultSettings.sync();
    }

    backend_ = new Dtk::Core::QSettingBackend(configPath_);
    dsettings_->setBackend(backend_);
    refreshModelNameItems();

    return true;
}

QString EchoFlowSettings::configPath() const {
    return configPath_;
}

Dtk::Core::DSettings *EchoFlowSettings::dsettings() const {
    return dsettings_;
}

void EchoFlowSettings::sync() {
    if (dsettings_) {
        dsettings_->sync();
    }
}

void EchoFlowSettings::populateComboBoxes() {
    // model_name items are populated dynamically by refreshModelNameItems()
    // (only downloaded models, plus the current recognition selection).
    setComboBoxItems(dsettings_, QStringLiteral("basic.model.mirror"),
                     QStringList{QStringLiteral("hf-mirror"),
                                 QStringLiteral("official")});
    setComboBoxItems(dsettings_, QStringLiteral("basic.recognition.language"),
                     QStringList{QString(), QStringLiteral("English"),
                                  QStringLiteral("Chinese"), QStringLiteral("Spanish"),
                                  QStringLiteral("Portuguese"), QStringLiteral("French"),
                                  QStringLiteral("Italian")},
                     QStringList{tr("System default"), QStringLiteral("English"),
                                  QStringLiteral("Chinese"), QStringLiteral("Spanish"),
                                  QStringLiteral("Portuguese"), QStringLiteral("French"),
                                  QStringLiteral("Italian")});
    setComboBoxItems(dsettings_, QStringLiteral("basic.recording.rate"),
                     QStringList{QStringLiteral("8000"), QStringLiteral("16000"),
                                  QStringLiteral("22050"), QStringLiteral("44100"),
                                  QStringLiteral("48000")});
    setComboBoxItems(dsettings_, QStringLiteral("basic.recording.channels"),
                     QStringList{QStringLiteral("1"), QStringLiteral("2")});
    setComboBoxItems(dsettings_, QStringLiteral("basic.recording.format"),
                     QStringList{QStringLiteral("s16"), QStringLiteral("s32"),
                                  QStringLiteral("f32"), QStringLiteral("u8")});

    QStringList sourceKeys{QString()};
    QStringList sourceValues{QStringLiteral("System default")};
    const QList<AudioSourceItem> sources = pipeWireSources();
    QMap<QString, int> descriptionCounts;
    for (const AudioSourceItem &source : sources) {
        descriptionCounts[source.description] += 1;
    }
    for (const AudioSourceItem &source : sources) {
        sourceKeys << source.name;
        sourceValues << sourceDisplayName(source, descriptionCounts);
    }
    setComboBoxItems(dsettings_, QStringLiteral("basic.recording.source"),
                     sourceKeys, sourceValues);
}

Dtk::Core::DSettingsOption *EchoFlowSettings::modelSelection(const ModelEntry &entry) const {
    return dsettings_ ? dsettings_->option(findPolishModel(entry.id) ? kPolishModel
                                                                  : kRecognitionModel)
                      : nullptr;
}

std::filesystem::path EchoFlowSettings::modelDirectory(const ModelEntry &entry) const {
    return std::filesystem::path(QFileInfo(configPath_).absolutePath().toStdString()) / entry.id;
}

void EchoFlowSettings::refreshModelNameItems() {
    if (!dsettings_) {
        return;
    }
    const auto refresh = [this](const QString &key, const std::vector<ModelEntry> &catalog) {
        auto option = dsettings_->option(key);
        if (!option) {
            return;
        }
        const QString current = option->value().toString();
        const bool isPolish = key == kPolishModel;
        QStringList keys;
        QStringList names;
        for (const ModelEntry &entry : catalog) {
            const QString id = QString::fromStdString(entry.id);
            if ((!isPolish && id == current) || isModelPresent(modelDirectory(entry), entry)) {
                keys << id;
                names << (isPolish ? QString::fromStdString(entry.displayName) : id);
            }
        }
        if (isPolish) {
            if (!keys.contains(current)) {
                option->setValue(keys.value(0, QStringLiteral("")));
            }
            if (keys.isEmpty()) {
                if (auto enabled = dsettings_->option(kPolishEnabled)) {
                    enabled->setValue(false);
                }
            }
        } else if (!current.isEmpty() && !keys.contains(current)) {
            keys.prepend(current);
            names.prepend(current);
        }
        setComboBoxItems(dsettings_, key, keys, names);
    };
    refresh(kRecognitionModel, modelCatalog());
    refresh(kPolishModel, polishModelCatalog());
}

void EchoFlowSettings::setComboBoxItems(Dtk::Core::DSettings *settings,
                                        const QString &path,
                                        const QStringList &items) {
    setComboBoxItems(settings, path, items, items);
}

void EchoFlowSettings::setComboBoxItems(Dtk::Core::DSettings *settings,
                                        const QString &path,
                                        const QStringList &keys,
                                        const QStringList &values) {
    if (!settings) {
        return;
    }
    Dtk::Core::DSettingsOption *option = settings->option(path);
    if (!option) {
        return;
    }
    option->setData(QStringLiteral("items"),
                    QMap<QString, QVariant>{{QStringLiteral("keys"), keys},
                                            {QStringLiteral("values"), values}});
}

} // namespace echoflow
