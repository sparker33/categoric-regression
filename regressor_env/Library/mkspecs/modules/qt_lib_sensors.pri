QT.sensors.VERSION = 5.15.3
QT.sensors.name = QtSensors
QT.sensors.module = Qt5Sensors_conda
QT.sensors.libs = $$QT_MODULE_LIB_BASE
QT.sensors.includes = $$QT_MODULE_INCLUDE_BASE $$QT_MODULE_INCLUDE_BASE/QtSensors
QT.sensors.frameworks =
QT.sensors.bins = $$QT_MODULE_BIN_BASE
QT.sensors.plugin_types = sensors sensorgestures
QT.sensors.depends = core
QT.sensors.uses =
QT.sensors.module_config = v2
QT.sensors.DEFINES = QT_SENSORS_LIB
QT.sensors.enabled_features =
QT.sensors.disabled_features =
QT_CONFIG +=
QT_MODULES += sensors
