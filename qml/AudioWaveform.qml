// SPDX-FileCopyrightText: 2026 Hualet Wang
// SPDX-License-Identifier: GPL-3.0-or-later

import QtQuick 2.15

Item {
    id: root
    property var levels: [0, 0, 0, 0, 0, 0, 0]
    property bool active: false
    property bool animate: true
    property color color: "#0081ff"
    readonly property int barCount: 7
    readonly property int barWidth: 3
    readonly property int barSpacing: 3
    implicitWidth: barCount * barWidth + (barCount - 1) * barSpacing
    implicitHeight: 24

    Repeater {
        model: root.barCount
        Rectangle {
            id: bar
            required property int index
            objectName: "audioBar" + index
            readonly property real level: {
                if (!root.active || root.levels.length === 0) return 0
                // Resample older five-column senders within the same frame.
                var position = index * (root.levels.length - 1) / (root.barCount - 1)
                var left = Math.floor(position)
                var right = Math.min(left + 1, root.levels.length - 1)
                var fraction = position - left
                return Math.max(0, Math.min(1,
                    root.levels[left] * (1 - fraction) + root.levels[right] * fraction))
            }
            x: index * (root.barWidth + root.barSpacing)
            anchors.verticalCenter: parent.verticalCenter
            width: root.barWidth
            // The capture side already applies one shared gain. Keep the
            // current frame's relative amplitudes when converting to pixels.
            property real displayedLevel: 0
            property bool ready: false
            height: root.barWidth + (root.height - root.barWidth) * displayedLevel
            radius: width / 2
            color: root.color

            function updateLevel() {
                if (!ready) return
                levelAnimation.stop()
                if (!root.active || !root.visible) {
                    displayedLevel = 0
                } else if (!root.animate) {
                    displayedLevel = level
                } else {
                    levelAnimation.from = displayedLevel
                    levelAnimation.to = level
                    levelAnimation.duration = level > displayedLevel ? 40 : 150
                    levelAnimation.start()
                }
            }
            onLevelChanged: updateLevel()
            Component.onCompleted: {
                ready = true
                updateLevel()
            }
            Connections {
                target: root
                function onActiveChanged() { bar.updateLevel() }
                function onVisibleChanged() { bar.updateLevel() }
                function onAnimateChanged() { bar.updateLevel() }
            }
            NumberAnimation {
                id: levelAnimation
                target: bar
                property: "displayedLevel"
            }
        }
    }
}
