// SPDX-FileCopyrightText: 2026 Hualet Wang
// SPDX-License-Identifier: GPL-3.0-or-later

#include "SettingsDialog.h"

#include "EchoFlowSettings.h"

#include "ModelCatalog.h"
#include "ModelRowWidget.h"

#include <QCheckBox>
#include <QLabel>
#include <QPushButton>
#include <QSignalBlocker>

#include <DComboBox>
#include <DSettings>
#include <DSettingsOption>
#include <DSettingsWidgetFactory>

namespace echoflow {

namespace {

const ModelEntry* entryForOption(Dtk::Core::DSettingsOption* opt) {
    if (!opt) {
        return nullptr;
    }
    // DTK6 hands the factory a QObject*; map by option key (reliable across
    // DTK versions — schema JSON -> option data wiring is not guaranteed).
    const QString key = opt->key();
    if (key.endsWith(QStringLiteral("download_0.6b"))) {
        return findModel("qwen3-asr-0.6b");
    }
    if (key.endsWith(QStringLiteral("download_1.7b"))) {
        return findModel("qwen3-asr-1.7b");
    }
    if (key.endsWith(QStringLiteral("download_polish_0.6b"))) {
        return findPolishModel("qwen3-0.6b");
    }
    if (key.endsWith(QStringLiteral("download_polish_1.7b"))) {
        return findPolishModel("qwen3-1.7b");
    }
    return nullptr;
}

Dtk::Widget::DComboBox* createModelPicker(Dtk::Core::DSettingsOption* option) {
    auto* picker = new Dtk::Widget::DComboBox;
    const auto refresh = [picker, option] {
        // Rebuilding the choices must not write a transient empty selection to settings.
        const QSignalBlocker blocker(picker);
        const auto items = option->data(QStringLiteral("items")).toMap();
        const auto keys = items.value(QStringLiteral("keys")).toStringList();
        const auto names = items.value(QStringLiteral("values")).toStringList();
        picker->clear();
        for (qsizetype index = 0; index < keys.size(); ++index) {
            picker->addItem(names.value(index, keys[index]), keys[index]);
        }
        picker->setCurrentIndex(picker->findData(option->value()));
    };
    QObject::connect(option, &Dtk::Core::DSettingsOption::dataChanged, picker, refresh);
    QObject::connect(option, &Dtk::Core::DSettingsOption::valueChanged, picker,
                     [picker](const QVariant &value) {
                         const QSignalBlocker blocker(picker);
                         picker->setCurrentIndex(picker->findData(value));
                     });
    QObject::connect(picker, &QComboBox::currentIndexChanged, option, [picker, option](int index) {
        if (index >= 0) {
            option->setValue(picker->itemData(index));
        }
    });
    refresh();
    return picker;
}

}  // namespace

SettingsDialog::SettingsDialog(Dtk::Core::DSettings *settings, const QString &controlPath, QWidget *parent)
    : Dtk::Widget::DSettingsDialog(parent) {
    setWindowTitle(tr("EchoFlow Settings"));

    // DSettingsDialog owns its factory (no global singleton). Register the
    // custom type on it BEFORE updateSettings builds the rows. Use the
    // ItemCreateHandler (pair) form so we supply BOTH the left-hand name label
    // and the right-hand download widget — DTK does not render the option
    // `name` as a label for custom-registered widget types.
    auto* factory = widgetFactory();
    factory->registerWidget(QStringLiteral("guide"),
        [this](QObject*) -> QPair<QWidget*, QWidget*> {
            auto* name = new QLabel(tr("Usage Guide"));
            name->setAlignment(Qt::AlignLeft | Qt::AlignVCenter);
            auto* button = new QPushButton(tr("Open"));
            button->setObjectName(QStringLiteral("usageGuideButton"));
            button->setAccessibleName(tr("Open the usage guide"));
            connect(button, &QPushButton::clicked, this,
                    [this] { emit usageGuideRequested(); });
            return {name, button};
        });
    factory->registerWidget(QStringLiteral("polishcheckbox"),
        [settings](QObject* obj) -> QPair<QWidget*, QWidget*> {
            auto* option = qobject_cast<Dtk::Core::DSettingsOption*>(obj);
            const auto model = settings->option(QStringLiteral("basic.polish.model_name"));
            if (!option || !model) {
                return {};
            }
            const auto text = option->data(QStringLiteral("text")).toString().toUtf8();
            auto* checkBox = new QCheckBox(QObject::tr(text.constData()));
            checkBox->setObjectName(QStringLiteral("polishEnabledCheckBox"));
            const auto hasModel = [model] {
                const auto items = model->data(QStringLiteral("items")).toMap();
                return !items.value(QStringLiteral("keys")).toStringList().isEmpty();
            };
            const auto refresh = [checkBox, option, hasModel] {
                const bool available = hasModel();
                checkBox->setEnabled(available);
                checkBox->setChecked(available && option->value().toBool());
                checkBox->setToolTip(available ? QString() : tr("Download a polish model to enable polishing."));
            };
            QObject::connect(model, &Dtk::Core::DSettingsOption::dataChanged, checkBox, refresh);
            QObject::connect(option, &Dtk::Core::DSettingsOption::valueChanged, checkBox, refresh);
            QObject::connect(checkBox, &QCheckBox::clicked, option, [option, hasModel](bool checked) {
                if (checked) {
                    EchoFlowSettings::instance()->refreshModelNameItems();
                }
                option->setValue(checked && hasModel());
            });
            refresh();
            return Dtk::Widget::DSettingsWidgetFactory::createStandardItem({}, option, checkBox);
        });
    factory->registerWidget(QStringLiteral("modelpicker"),
        [](QObject* obj) -> QPair<QWidget*, QWidget*> {
            auto* option = qobject_cast<Dtk::Core::DSettingsOption*>(obj);
            if (!option) {
                return {};
            }
            auto* name = new QLabel(QObject::tr("Model"));
            auto* picker = createModelPicker(option);
            name->setBuddy(picker);
            return {name, picker};
        });
    factory->registerWidget(QStringLiteral("modeldownload"),
        [controlPath](QObject* obj) -> QPair<QWidget*, QWidget*> {
            auto* opt = qobject_cast<Dtk::Core::DSettingsOption*>(obj);
            const ModelEntry* e = entryForOption(opt);
            auto* name = new QLabel(e ? QString::fromStdString(e->displayName)
                                      : QObject::tr("Unknown model"));
            name->setAlignment(Qt::AlignLeft | Qt::AlignVCenter);
            return {name, new ModelRowWidget(e, controlPath)};
        });

    updateSettings(settings);
}

}  // namespace echoflow
