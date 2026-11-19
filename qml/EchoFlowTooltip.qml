// SPDX-FileCopyrightText: 2026 Hualet Wang
// SPDX-License-Identifier: GPL-3.0-or-later

import QtQuick 2.15
import QtQuick.Window 2.15
import QtQuick.Controls 2.15
import org.deepin.dtk 1.0 as D

// Voice capsule follows the controller's explicit session phase. Idle offers a
// microphone action beside a static waveform. During recording the left X
// button cancels (discard), the right check button stops and starts
// transcription, and the waveform animates. Status phases show centered text.
Window {
    id: root
    flags: Qt.ToolTip | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.WindowDoesNotAcceptFocus
    color: "transparent"
    D.DWindow.enableBlurWindow: root.visible && capsule.opacity > 0
    D.DWindow.enabled: true
    D.DWindow.windowRadius: capsule.kRadius
    visible: false

    property string message: ""
    property string phase: "idle"
    property int targetX: 0
    property int targetY: 0
    readonly property color capsuleBackground: typeof theme !== "undefined" ? theme.capsuleBackground : "#232629"
    readonly property color capsuleBlend: Qt.rgba(capsuleBackground.r,
                                                   capsuleBackground.g,
                                                   capsuleBackground.b,
                                                   D.DTK.themeType === D.ApplicationHelper.DarkType ? 0.6 : 0.4)
    readonly property color capsuleOutsideBorder: D.DTK.themeType === D.ApplicationHelper.DarkType
                                                    ? Qt.rgba(0, 0, 0, 0.6)
                                                    : Qt.rgba(0, 0, 0, 0.1)
    readonly property color capsuleInsideBorder: D.DTK.themeType === D.ApplicationHelper.DarkType
                                                   ? Qt.rgba(1, 1, 1, 0.1)
                                                   : Qt.rgba(1, 1, 1, 0.2)
    readonly property color capsuleBorder: typeof theme !== "undefined" ? theme.capsuleBorder : "#3f4348"
    readonly property color capsuleText: typeof theme !== "undefined" ? theme.capsuleText : "#f4f6f8"
    readonly property color accent: typeof theme !== "undefined" ? theme.accent : "#0081ff"
    readonly property color accentText: typeof theme !== "undefined" ? theme.accentText : "white"
    onAccentTextChanged: {
        microphoneCanvas.requestPaint()
        confirmCanvas.requestPaint()
    }
    onCapsuleTextChanged: cancelCanvas.requestPaint()

    readonly property bool idle: root.phase === "idle"
    readonly property bool recording: root.phase === "recording"
    readonly property bool transcribing: root.phase === "transcribing"
    readonly property bool cancelling: root.phase === "cancelling"

    width: capsule.width
    height: capsule.height

    function updatePosition() {
        root.x = targetX - Math.ceil(width / 2)
        root.y = targetY - height
    }

    // Moving to another screen can update the status label's width limit.
    onWidthChanged: Qt.callLater(root.updatePosition)
    onHeightChanged: Qt.callLater(root.updatePosition)

    Connections {
        target: tooltipController
        function onCapsuleChanged(visibility, newPhase, msg, hasPosition, moveX, moveY) {
            root.phase = newPhase
            root.message = msg
            if (hasPosition) {
                root.targetX = moveX
                root.targetY = moveY
                root.updatePosition()
            }
            if (visibility) {
                root.visible = true
                capsule.opacity = 1
            } else {
                capsule.opacity = 0
                root.visible = false
            }
        }
    }

    // ---------------- capsule body ----------------
    Rectangle {
        id: capsule
        anchors.centerIn: parent

        readonly property int kHeight: 50
        readonly property int kButtonSize: 30
        readonly property int kRadius: kHeight / 2
        readonly property int kButtonInset: (kHeight - kButtonSize) / 2
        readonly property int kHPad: 12
        readonly property int kGap: 10
        readonly property real kWaveWidth: waveArea.implicitWidth
        readonly property real kIdleWidth: kHPad + kWaveWidth + kGap
                                                + kButtonSize + kHPad
        readonly property real kRecordingWidth: 2 * kHPad + 2 * kButtonSize
                                                + 3 * kGap + kWaveWidth + recordingTimer.width
        readonly property real kStatusWidth: Math.min(statusLabel.implicitWidth + 2 * kHPad,
                                                      Screen.width / 3)

        height: kHeight
        radius: kRadius
        color: "transparent"
        clip: false

        // idle: [pad][static wave][gap][microphone][pad]
        // recording: [pad][X][gap][wave][gap][timer][gap][✓][pad]
        // transcribing/cancelling: [ centered status text ]
        width: root.idle ? kIdleWidth
                         : root.recording ? kRecordingWidth : kStatusWidth

        D.StyledBehindWindowBlur {
            anchors.fill: parent
            control: root
            cornerRadius: capsule.kRadius
            blendColor: root.capsuleBlend
        }

        D.OutsideBoxBorder {
            anchors.fill: parent
            radius: capsule.kRadius
            color: root.capsuleOutsideBorder
            z: D.DTK.AboveOrder
        }

        D.InsideBoxBorder {
            anchors.fill: parent
            radius: capsule.kRadius
            color: root.capsuleInsideBorder
            z: D.DTK.AboveOrder
        }

        // ---- idle: start-recording microphone button ----
        Item {
            id: microphoneButton
            width: capsule.kButtonSize
            height: capsule.kButtonSize
            anchors {
                right: parent.right
                rightMargin: capsule.kHPad
                verticalCenter: parent.verticalCenter
            }
            visible: root.idle
            Accessible.name: qsTr("Start recording")
            Accessible.role: Accessible.Button
            Accessible.onPressAction: tooltipController.requestToggle()

            Rectangle {
                anchors.fill: parent
                radius: width / 2
                color: microphoneArea.pressed
                       ? Qt.darker(root.accent, 1.18)
                       : microphoneArea.containsMouse
                         ? Qt.lighter(root.accent, 1.12) : root.accent
                Behavior on color { ColorAnimation { duration: 120 } }
            }

            // symbolic microphone body, pickup arc, stem, and stand
            Canvas {
                id: microphoneCanvas
                anchors.centerIn: parent
                width: 14
                height: 14
                onPaint: {
                    var ctx = getContext("2d")
                    ctx.reset()
                    ctx.strokeStyle = root.accentText
                    ctx.lineWidth = 1.7
                    ctx.lineCap = "round"
                    ctx.lineJoin = "round"

                    ctx.beginPath()
                    ctx.moveTo(7, 1.5)
                    ctx.bezierCurveTo(5.3, 1.5, 4.5, 2.7, 4.5, 4.2)
                    ctx.lineTo(4.5, 6.5)
                    ctx.bezierCurveTo(4.5, 8, 5.3, 9.2, 7, 9.2)
                    ctx.bezierCurveTo(8.7, 9.2, 9.5, 8, 9.5, 6.5)
                    ctx.lineTo(9.5, 4.2)
                    ctx.bezierCurveTo(9.5, 2.7, 8.7, 1.5, 7, 1.5)
                    ctx.stroke()

                    ctx.beginPath()
                    ctx.moveTo(2.5, 6.5)
                    ctx.bezierCurveTo(2.5, 9.3, 4.3, 11, 7, 11)
                    ctx.bezierCurveTo(9.7, 11, 11.5, 9.3, 11.5, 6.5)
                    ctx.moveTo(7, 11)
                    ctx.lineTo(7, 13)
                    ctx.moveTo(4.5, 13)
                    ctx.lineTo(9.5, 13)
                    ctx.stroke()
                }
            }

            MouseArea {
                id: microphoneArea
                anchors.fill: parent
                hoverEnabled: true
                cursorShape: Qt.PointingHandCursor
                onClicked: tooltipController.requestToggle()
            }
        }

        // ---- left: cancel (X) button ----
        Item {
            id: cancelButton
            width: capsule.kButtonSize
            height: capsule.kButtonSize
            anchors {
                left: parent.left
                leftMargin: capsule.kHPad
                verticalCenter: parent.verticalCenter
            }
            visible: root.recording
            Accessible.name: qsTr("Cancel and discard the recording")
            Accessible.role: Accessible.Button
            Accessible.onPressAction: tooltipController.requestCancel()

            Rectangle {
                id: cancelBg
                anchors.fill: parent
                radius: width / 2
                color: cancelArea.pressed
                       ? Qt.rgba(1, 1, 1, 0.20)
                       : cancelArea.containsMouse
                         ? Qt.rgba(1, 1, 1, 0.12) : "transparent"
                Behavior on color { ColorAnimation { duration: 120 } }
            }

            // symbolic X glyph
            Canvas {
                id: cancelCanvas
                anchors.centerIn: parent
                width: 12; height: 12
                onPaint: {
                    var ctx = getContext("2d")
                    ctx.reset()
                    ctx.strokeStyle = root.capsuleText
                    ctx.lineWidth = 1.8
                    ctx.lineCap = "round"
                    ctx.beginPath()
                    ctx.moveTo(2, 2); ctx.lineTo(10, 10)
                    ctx.moveTo(10, 2); ctx.lineTo(2, 10)
                    ctx.stroke()
                }
            }

            MouseArea {
                id: cancelArea
                anchors.fill: parent
                hoverEnabled: true
                cursorShape: Qt.PointingHandCursor
                onClicked: tooltipController.requestCancel()
            }
        }

        // ---- idle/recording: current microphone waveform ----
        AudioWaveform {
            id: waveArea
            anchors {
                right: root.idle ? microphoneButton.left : recordingTimer.left
                rightMargin: capsule.kGap
                verticalCenter: parent.verticalCenter
            }
            width: implicitWidth
            height: implicitHeight
            visible: root.idle || root.recording
            active: root.recording
            levels: tooltipController.audioLevels
            color: root.accent
        }

        Label {
            id: recordingTimer
            anchors {
                right: confirmButton.left
                rightMargin: capsule.kGap
                verticalCenter: parent.verticalCenter
            }
            width: Math.max(34, implicitWidth)
            visible: root.phase === "recording"
            text: tooltipController.recordingDuration
            color: root.capsuleText
            opacity: 0.7
            font.pixelSize: 12
            horizontalAlignment: Text.AlignHCenter
        }

        // ---- transcribing/cancelling: centered status ----
        Label {
            id: statusLabel
            anchors.centerIn: parent
            visible: root.transcribing || root.cancelling
            width: parent.width - 2 * capsule.kHPad
            elide: Text.ElideLeft
            horizontalAlignment: Text.AlignHCenter
            text: root.message
            color: root.capsuleText
            font.pixelSize: 13
        }

        // ---- right: stop & transcribe (✓) button ----
        Item {
            id: confirmButton
            width: capsule.kButtonSize
            height: capsule.kButtonSize
            anchors {
                right: parent.right
                rightMargin: capsule.kHPad
                verticalCenter: parent.verticalCenter
            }
            visible: root.recording
            Accessible.name: qsTr("Finish recording and start transcription")
            Accessible.role: Accessible.Button
            Accessible.onPressAction: tooltipController.requestToggle()

            Rectangle {
                id: confirmBg
                anchors.fill: parent
                radius: width / 2
                color: confirmArea.pressed
                       ? Qt.darker(root.accent, 1.18)
                       : confirmArea.containsMouse
                         ? Qt.lighter(root.accent, 1.12) : root.accent
                Behavior on color { ColorAnimation { duration: 120 } }
            }

            // symbolic check glyph
            Canvas {
                id: confirmCanvas
                anchors.centerIn: parent
                width: 14; height: 14
                onPaint: {
                    var ctx = getContext("2d")
                    ctx.reset()
                    ctx.strokeStyle = root.accentText
                    ctx.lineWidth = 2.0
                    ctx.lineCap = "round"
                    ctx.lineJoin = "round"
                    ctx.beginPath()
                    ctx.moveTo(2, 7.5)
                    ctx.lineTo(5.5, 11)
                    ctx.lineTo(12, 3.5)
                    ctx.stroke()
                }
            }

            MouseArea {
                id: confirmArea
                anchors.fill: parent
                hoverEnabled: true
                cursorShape: Qt.PointingHandCursor
                onClicked: tooltipController.requestToggle()
            }
        }
    }

}
