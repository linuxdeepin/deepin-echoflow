// SPDX-FileCopyrightText: 2026 Hualet Wang
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <QJsonObject>
#include <QString>

#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace echoflow::benchmark {

inline QList<uint> normalized(const QString& text)
{
    QString result;
    for (const auto character : text.normalized(QString::NormalizationForm_KC).toCaseFolded()) {
        if (!character.isSpace() && !character.isPunct()) result += character;
    }
    return result.toUcs4();
}

inline QJsonObject scored(QJsonObject row, const QString& reference)
{
    if (reference.isEmpty()) return row;
    const auto expected = normalized(reference);
    const auto actual = normalized(row.value("transcript").toString());
    if (expected.isEmpty()) throw std::runtime_error("reference contains no scored characters");
    std::vector<int> previous(static_cast<size_t>(actual.size()) + 1);
    std::iota(previous.begin(), previous.end(), 0);
    for (qsizetype i = 0; i < expected.size(); ++i) {
        std::vector<int> current(previous.size());
        current[0] = static_cast<int>(i + 1);
        for (qsizetype j = 0; j < actual.size(); ++j) {
            current[j + 1] = std::min({current[j] + 1, previous[j + 1] + 1,
                                      previous[j] + (expected[i] != actual[j])});
        }
        previous = std::move(current);
    }
    row.insert("character_errors", previous.back());
    row.insert("reference_characters", static_cast<int>(expected.size()));
    row.insert("cer", static_cast<double>(previous.back()) / expected.size());
    return row;
}


} // namespace echoflow::benchmark
