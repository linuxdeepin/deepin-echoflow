# SPDX-FileCopyrightText: 2026 Hualet Wang
# SPDX-License-Identifier: GPL-3.0-or-later

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    cmake_minimum_required(VERSION 3.16)
    project(deepin-echoflow-ui-host LANGUAGES C CXX)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

find_package(Qt6 REQUIRED COMPONENTS Core Gui Qml Quick Widgets Network DBus LinguistTools)
find_package(Qt6 QUIET COMPONENTS WaylandClient)
find_package(Dtk6 COMPONENTS Widget Core Gui REQUIRED)
include(GNUInstallDirs)
# Long-press checks the active window through the X server directly.
find_path(XCB_INCLUDE_DIR xcb/xcb.h)
find_library(XCB_LIBRARY xcb)
if(NOT XCB_INCLUDE_DIR OR NOT XCB_LIBRARY)
    message(FATAL_ERROR "libxcb development files are required")
endif()

add_executable(deepin-echoflow-ui
    main.cpp
    CapsuleWindowPolicy.cpp
    CapsuleController.cpp
    UiSocketServer.cpp
    LongPressDetector.cpp
    ScreenGeometry.cpp
    X11ActiveWindow.cpp
    XEventMonitorSource.cpp
    EchoFlowSettings.cpp
    SettingsDialog.cpp
    ModelDownloader.cpp
    ModelDownloadCoordinator.cpp
    ModelRowWidget.cpp
    UiActivationServer.cpp
    UiActivationHost.cpp
    OnboardingState.cpp
    SetupCommandRunner.cpp
    ModelSetupAdapter.cpp
    OnboardingSetupController.cpp
    OnboardingIllustration.cpp
    OnboardingDialog.cpp
    SettingsTranslation.cpp
    icons.qrc
    onboarding.qrc
    settings.qrc
    qml.qrc
)

target_include_directories(deepin-echoflow-ui PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/../service
    ${XCB_INCLUDE_DIR}
)

target_link_libraries(deepin-echoflow-ui
    PRIVATE
        Qt6::Core
        Qt6::Gui
        Qt6::Qml
        Qt6::Quick
        Qt6::Widgets
        Qt6::Network
        Qt6::DBus
        ${XCB_LIBRARY}
        Dtk6::Widget
        Dtk6::Core
        Dtk6::Gui
)

if(TARGET Qt6::WaylandClient AND TARGET Qt6::GuiPrivate)
    target_link_libraries(deepin-echoflow-ui PRIVATE Qt6::WaylandClient Qt6::GuiPrivate)
    target_compile_definitions(deepin-echoflow-ui PRIVATE ECHOFLOW_HAVE_TREELAND_DDE_SHELL)
    qt_generate_wayland_protocol_client_sources(deepin-echoflow-ui
        NO_INCLUDE_CORE_ONLY
        FILES
            ${CMAKE_CURRENT_SOURCE_DIR}/protocols/treeland-dde-shell-unstable-v2.xml
    )
else()
    message(STATUS "Treeland DDE shell support disabled; using Qt window hints")
endif()

# The capsule needs its own shell integration to bind to an explicit output.
# Other windows continue to use Qt's default xdg-shell integration.
if(TARGET Qt6::WaylandClientPrivate AND TARGET Qt6::GuiPrivate
   AND Qt6WaylandClient_VERSION VERSION_GREATER_EQUAL 6.8)
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        pkg_get_variable(WAYLAND_PROTOCOLS_DIR wayland-protocols pkgdatadir)
    endif()
    set(XDG_SHELL_XML "${WAYLAND_PROTOCOLS_DIR}/stable/xdg-shell/xdg-shell.xml")
    if(EXISTS "${XDG_SHELL_XML}")
        target_link_libraries(deepin-echoflow-ui PRIVATE Qt6::WaylandClientPrivate)
        target_compile_definitions(deepin-echoflow-ui PRIVATE ECHOFLOW_HAVE_LAYER_SHELL)
        qt_generate_wayland_protocol_client_sources(deepin-echoflow-ui
            NO_INCLUDE_CORE_ONLY PRIVATE_CODE
            FILES
                ${CMAKE_CURRENT_SOURCE_DIR}/protocols/wlr-layer-shell-unstable-v1.xml
                ${XDG_SHELL_XML}
        )
    else()
        message(STATUS "Layer shell support disabled; wayland-protocols is missing")
    endif()
endif()

if(NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    target_compile_definitions(deepin-echoflow-ui PRIVATE
        ECHOFLOW_SYSTEM_AUTOSTART_PATH="${CMAKE_INSTALL_FULL_SYSCONFDIR}/xdg/autostart/deepin-echoflow-autostart.desktop")
endif()

# lupdate runs on this target's sources (the .qrc pulls in the QML files);
# rebuild the update_translations target to refresh the .ts files. Install
# generated .qm files where DApplication::loadTranslator() looks for them.
qt_add_translations(deepin-echoflow-ui
    SOURCES
        ../qml/EchoFlowTooltip.qml
        ../qml/EchoFlowHint.qml
    TS_FILES
        translations/deepin-echoflow_zh_CN.ts
        translations/deepin-echoflow_es.ts
        translations/deepin-echoflow_fr.ts
        translations/deepin-echoflow_it.ts
        translations/deepin-echoflow_pt.ts
    QM_FILES_OUTPUT_VARIABLE ECHOFLOW_TRANSLATION_QM_FILES
)
foreach(translation_file IN LISTS ECHOFLOW_TRANSLATION_QM_FILES)
    get_filename_component(translation_name "${translation_file}" NAME)
    string(REGEX REPLACE "^deepin-echoflow_(.+)\\.qm$" "\\1" translation_locale
        "${translation_name}")
    install(FILES "${translation_file}"
        DESTINATION "${CMAKE_INSTALL_DATADIR}/deepin-echoflow-ui/translations"
        RENAME "deepin-echoflow-ui_${translation_locale}.qm")
endforeach()

install(TARGETS deepin-echoflow-ui RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")
install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/icons/deepin-echoflow.svg"
    DESTINATION "${CMAKE_INSTALL_DATADIR}/icons/hicolor/scalable/apps")
