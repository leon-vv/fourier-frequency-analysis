# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file './interface.ui'
#
# Created by: PyQt5 UI code generator 5.11.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(902, 618)
        MainWindow.setAccessibleName("")
        MainWindow.setUnifiedTitleAndToolBarOnMac(False)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.groupBox_3 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_3.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.groupBox_3.sizePolicy().hasHeightForWidth())
        self.groupBox_3.setSizePolicy(sizePolicy)
        self.groupBox_3.setObjectName("groupBox_3")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.groupBox_3)
        self.verticalLayout_2.setContentsMargins(0, -1, 0, -1)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.widget_2 = QtWidgets.QWidget(self.groupBox_3)
        self.widget_2.setObjectName("widget_2")
        self.formLayout = QtWidgets.QFormLayout(self.widget_2)
        self.formLayout.setObjectName("formLayout")
        self.label_3 = QtWidgets.QLabel(self.widget_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_3.sizePolicy().hasHeightForWidth())
        self.label_3.setSizePolicy(sizePolicy)
        self.label_3.setObjectName("label_3")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label_3)
        self.source_picker = QtWidgets.QComboBox(self.widget_2)
        self.source_picker.setObjectName("source_picker")
        self.source_picker.addItem("")
        self.source_picker.addItem("")
        self.source_picker.addItem("")
        self.source_picker.addItem("")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.source_picker)
        self.verticalLayout_2.addWidget(self.widget_2)
        self.sine_controls = QtWidgets.QWidget(self.groupBox_3)
        self.sine_controls.setObjectName("sine_controls")
        self.formLayout_5 = QtWidgets.QFormLayout(self.sine_controls)
        self.formLayout_5.setObjectName("formLayout_5")
        self.label_2 = QtWidgets.QLabel(self.sine_controls)
        self.label_2.setObjectName("label_2")
        self.formLayout_5.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label_2)
        self.amplitude_edit = QtWidgets.QLineEdit(self.sine_controls)
        self.amplitude_edit.setObjectName("amplitude_edit")
        self.formLayout_5.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.amplitude_edit)
        self.label_6 = QtWidgets.QLabel(self.sine_controls)
        self.label_6.setObjectName("label_6")
        self.formLayout_5.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label_6)
        self.frequency_edit = QtWidgets.QLineEdit(self.sine_controls)
        self.frequency_edit.setObjectName("frequency_edit")
        self.formLayout_5.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.frequency_edit)
        self.label_7 = QtWidgets.QLabel(self.sine_controls)
        self.label_7.setObjectName("label_7")
        self.formLayout_5.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.label_7)
        self.endtime_edit = QtWidgets.QLineEdit(self.sine_controls)
        self.endtime_edit.setObjectName("endtime_edit")
        self.formLayout_5.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.endtime_edit)
        self.label_8 = QtWidgets.QLabel(self.sine_controls)
        self.label_8.setObjectName("label_8")
        self.formLayout_5.setWidget(3, QtWidgets.QFormLayout.LabelRole, self.label_8)
        self.mean_edit = QtWidgets.QLineEdit(self.sine_controls)
        self.mean_edit.setObjectName("mean_edit")
        self.formLayout_5.setWidget(3, QtWidgets.QFormLayout.FieldRole, self.mean_edit)
        self.label_10 = QtWidgets.QLabel(self.sine_controls)
        self.label_10.setObjectName("label_10")
        self.formLayout_5.setWidget(4, QtWidgets.QFormLayout.LabelRole, self.label_10)
        self.deviation_edit = QtWidgets.QLineEdit(self.sine_controls)
        self.deviation_edit.setObjectName("deviation_edit")
        self.formLayout_5.setWidget(4, QtWidgets.QFormLayout.FieldRole, self.deviation_edit)
        self.verticalLayout_2.addWidget(self.sine_controls)
        self.file_control = QtWidgets.QWidget(self.groupBox_3)
        self.file_control.setObjectName("file_control")
        self.formLayout_4 = QtWidgets.QFormLayout(self.file_control)
        self.formLayout_4.setObjectName("formLayout_4")
        self.label_4 = QtWidgets.QLabel(self.file_control)
        self.label_4.setObjectName("label_4")
        self.formLayout_4.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label_4)
        self.pick_file_button = QtWidgets.QToolButton(self.file_control)
        self.pick_file_button.setObjectName("pick_file_button")
        self.formLayout_4.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.pick_file_button)
        self.label_11 = QtWidgets.QLabel(self.file_control)
        self.label_11.setObjectName("label_11")
        self.formLayout_4.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label_11)
        self.sample_time_edit = QtWidgets.QLineEdit(self.file_control)
        self.sample_time_edit.setObjectName("sample_time_edit")
        self.formLayout_4.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.sample_time_edit)
        self.verticalLayout_2.addWidget(self.file_control)
        self.verticalLayout.addWidget(self.groupBox_3)
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.groupBox.sizePolicy().hasHeightForWidth())
        self.groupBox.setSizePolicy(sizePolicy)
        self.groupBox.setObjectName("groupBox")
        self.formLayout_2 = QtWidgets.QFormLayout(self.groupBox)
        self.formLayout_2.setHorizontalSpacing(22)
        self.formLayout_2.setObjectName("formLayout_2")
        self.label_5 = QtWidgets.QLabel(self.groupBox)
        self.label_5.setObjectName("label_5")
        self.formLayout_2.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label_5)
        self.domain_picker = QtWidgets.QComboBox(self.groupBox)
        self.domain_picker.setObjectName("domain_picker")
        self.domain_picker.addItem("")
        self.domain_picker.addItem("")
        self.formLayout_2.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.domain_picker)
        self.verticalLayout.addWidget(self.groupBox)
        self.groupBox_2 = QtWidgets.QGroupBox(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.groupBox_2.sizePolicy().hasHeightForWidth())
        self.groupBox_2.setSizePolicy(sizePolicy)
        self.groupBox_2.setObjectName("groupBox_2")
        self.formLayout_3 = QtWidgets.QFormLayout(self.groupBox_2)
        self.formLayout_3.setHorizontalSpacing(22)
        self.formLayout_3.setVerticalSpacing(15)
        self.formLayout_3.setObjectName("formLayout_3")
        self.label_9 = QtWidgets.QLabel(self.groupBox_2)
        self.label_9.setObjectName("label_9")
        self.formLayout_3.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label_9)
        self.output_target = QtWidgets.QComboBox(self.groupBox_2)
        self.output_target.setObjectName("output_target")
        self.output_target.addItem("")
        self.output_target.addItem("")
        self.formLayout_3.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.output_target)
        self.write_button = QtWidgets.QPushButton(self.groupBox_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.write_button.sizePolicy().hasHeightForWidth())
        self.write_button.setSizePolicy(sizePolicy)
        self.write_button.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.write_button.setFlat(False)
        self.write_button.setObjectName("write_button")
        self.formLayout_3.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.write_button)
        self.verticalLayout.addWidget(self.groupBox_2)
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem)
        self.horizontalLayout.addLayout(self.verticalLayout)
        self.plot_view = PlotWidget(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(2)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.plot_view.sizePolicy().hasHeightForWidth())
        self.plot_view.setSizePolicy(sizePolicy)
        self.plot_view.setObjectName("plot_view")
        self.horizontalLayout.addWidget(self.plot_view)
        self.verticalLayout_4.addLayout(self.horizontalLayout)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setSizeConstraint(QtWidgets.QLayout.SetFixedSize)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        spacerItem1 = QtWidgets.QSpacerItem(0, 0, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem1)
        self.label = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label.sizePolicy().hasHeightForWidth())
        self.label.setSizePolicy(sizePolicy)
        self.label.setObjectName("label")
        self.horizontalLayout_2.addWidget(self.label)
        self.verticalLayout_4.addLayout(self.horizontalLayout_2)
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Physics Experiments Data Analysis"))
        self.groupBox_3.setTitle(_translate("MainWindow", "Input"))
        self.label_3.setText(_translate("MainWindow", "Source"))
        self.source_picker.setItemText(0, _translate("MainWindow", "File"))
        self.source_picker.setItemText(1, _translate("MainWindow", "Sine"))
        self.source_picker.setItemText(2, _translate("MainWindow", "MyDaq device"))
        self.source_picker.setItemText(3, _translate("MainWindow", "Rigol Oscilloscope"))
        self.label_2.setText(_translate("MainWindow", "Amplitude (V)"))
        self.amplitude_edit.setText(_translate("MainWindow", "1"))
        self.label_6.setText(_translate("MainWindow", "Frequency (Hz)"))
        self.label_7.setText(_translate("MainWindow", "Eindtijd (s)"))
        self.label_8.setText(_translate("MainWindow", "Mean Noise"))
        self.label_10.setText(_translate("MainWindow", "Deviation Noise"))
        self.label_4.setText(_translate("MainWindow", "Pick file"))
        self.pick_file_button.setText(_translate("MainWindow", "..."))
        self.label_11.setText(_translate("MainWindow", "Sample Time"))
        self.groupBox.setTitle(_translate("MainWindow", "Domain"))
        self.label_5.setText(_translate("MainWindow", "Show"))
        self.domain_picker.setItemText(0, _translate("MainWindow", "Frequency Domain"))
        self.domain_picker.setItemText(1, _translate("MainWindow", "Time Domain"))
        self.groupBox_2.setTitle(_translate("MainWindow", "Output"))
        self.label_9.setText(_translate("MainWindow", "Target"))
        self.output_target.setItemText(0, _translate("MainWindow", "File"))
        self.output_target.setItemText(1, _translate("MainWindow", "MyDaq device"))
        self.write_button.setText(_translate("MainWindow", "Write"))
        self.label.setText(_translate("MainWindow", "© Tobias Göbel & Léon van Velzen"))

from pyqtgraph import PlotWidget
