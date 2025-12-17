# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'LoadDataSettingsDlg.ui'
##
## Created by: Qt User Interface Compiler version 6.8.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QAbstractButton, QApplication, QComboBox, QDialog,
    QDialogButtonBox, QGroupBox, QHBoxLayout, QLabel,
    QLineEdit, QPushButton, QSizePolicy, QSpacerItem,
    QVBoxLayout, QWidget)

class Ui_LoadDataSettingsDlg(object):
    def setupUi(self, LoadDataSettingsDlg):
        if not LoadDataSettingsDlg.objectName():
            LoadDataSettingsDlg.setObjectName(u"LoadDataSettingsDlg")
        LoadDataSettingsDlg.resize(356, 152)
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(LoadDataSettingsDlg.sizePolicy().hasHeightForWidth())
        LoadDataSettingsDlg.setSizePolicy(sizePolicy)
        LoadDataSettingsDlg.setSizeGripEnabled(False)
        LoadDataSettingsDlg.setModal(True)
        self.verticalLayout = QVBoxLayout(LoadDataSettingsDlg)
        self.verticalLayout.setSpacing(0)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.groupBox = QGroupBox(LoadDataSettingsDlg)
        self.groupBox.setObjectName(u"groupBox")
        self.verticalLayout_2 = QVBoxLayout(self.groupBox)
        self.verticalLayout_2.setSpacing(5)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.verticalLayout_2.setContentsMargins(7, 7, 7, 7)
        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.label = QLabel(self.groupBox)
        self.label.setObjectName(u"label")

        self.horizontalLayout.addWidget(self.label)

        self.lineEdit_path = QLineEdit(self.groupBox)
        self.lineEdit_path.setObjectName(u"lineEdit_path")

        self.horizontalLayout.addWidget(self.lineEdit_path)

        self.pushButton_select_file = QPushButton(self.groupBox)
        self.pushButton_select_file.setObjectName(u"pushButton_select_file")

        self.horizontalLayout.addWidget(self.pushButton_select_file)

        self.pushButton_select_folder = QPushButton(self.groupBox)
        self.pushButton_select_folder.setObjectName(u"pushButton_select_folder")

        self.horizontalLayout.addWidget(self.pushButton_select_folder)

        self.horizontalLayout.setStretch(1, 1)

        self.verticalLayout_2.addLayout(self.horizontalLayout)

        self.horizontalLayout_2 = QHBoxLayout()
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.label_2 = QLabel(self.groupBox)
        self.label_2.setObjectName(u"label_2")
        self.label_2.setMinimumSize(QSize(90, 0))
        self.label_2.setMaximumSize(QSize(90, 16777215))
        self.label_2.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.horizontalLayout_2.addWidget(self.label_2)

        self.comboBox_oct_module = QComboBox(self.groupBox)
        self.comboBox_oct_module.setObjectName(u"comboBox_oct_module")
        self.comboBox_oct_module.setStyleSheet(u"")

        self.horizontalLayout_2.addWidget(self.comboBox_oct_module)

        self.label_5 = QLabel(self.groupBox)
        self.label_5.setObjectName(u"label_5")

        self.horizontalLayout_2.addWidget(self.label_5)

        self.comboBox_oct_func = QComboBox(self.groupBox)
        self.comboBox_oct_func.setObjectName(u"comboBox_oct_func")
        self.comboBox_oct_func.setStyleSheet(u"")

        self.horizontalLayout_2.addWidget(self.comboBox_oct_func)

        self.horizontalLayout_2.setStretch(1, 1)
        self.horizontalLayout_2.setStretch(3, 1)

        self.verticalLayout_2.addLayout(self.horizontalLayout_2)

        self.horizontalLayout_3 = QHBoxLayout()
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.label_3 = QLabel(self.groupBox)
        self.label_3.setObjectName(u"label_3")
        self.label_3.setMinimumSize(QSize(90, 0))
        self.label_3.setMaximumSize(QSize(90, 16777215))
        self.label_3.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.horizontalLayout_3.addWidget(self.label_3)

        self.comboBox_octa_module = QComboBox(self.groupBox)
        self.comboBox_octa_module.setObjectName(u"comboBox_octa_module")
        self.comboBox_octa_module.setStyleSheet(u"")

        self.horizontalLayout_3.addWidget(self.comboBox_octa_module)

        self.label_6 = QLabel(self.groupBox)
        self.label_6.setObjectName(u"label_6")

        self.horizontalLayout_3.addWidget(self.label_6)

        self.comboBox_octa_func = QComboBox(self.groupBox)
        self.comboBox_octa_func.setObjectName(u"comboBox_octa_func")
        self.comboBox_octa_func.setStyleSheet(u"")

        self.horizontalLayout_3.addWidget(self.comboBox_octa_func)

        self.horizontalLayout_3.setStretch(1, 1)
        self.horizontalLayout_3.setStretch(3, 1)

        self.verticalLayout_2.addLayout(self.horizontalLayout_3)

        self.horizontalLayout_4 = QHBoxLayout()
        self.horizontalLayout_4.setObjectName(u"horizontalLayout_4")
        self.label_4 = QLabel(self.groupBox)
        self.label_4.setObjectName(u"label_4")
        self.label_4.setMinimumSize(QSize(90, 0))
        self.label_4.setMaximumSize(QSize(90, 16777215))
        self.label_4.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.horizontalLayout_4.addWidget(self.label_4)

        self.comboBox_seg_module = QComboBox(self.groupBox)
        self.comboBox_seg_module.setObjectName(u"comboBox_seg_module")
        self.comboBox_seg_module.setStyleSheet(u"")

        self.horizontalLayout_4.addWidget(self.comboBox_seg_module)

        self.label_7 = QLabel(self.groupBox)
        self.label_7.setObjectName(u"label_7")

        self.horizontalLayout_4.addWidget(self.label_7)

        self.comboBox_seg_func = QComboBox(self.groupBox)
        self.comboBox_seg_func.setObjectName(u"comboBox_seg_func")
        self.comboBox_seg_func.setStyleSheet(u"")

        self.horizontalLayout_4.addWidget(self.comboBox_seg_func)

        self.horizontalLayout_4.setStretch(1, 1)
        self.horizontalLayout_4.setStretch(3, 1)

        self.verticalLayout_2.addLayout(self.horizontalLayout_4)

        self.verticalSpacer = QSpacerItem(20, 3, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.verticalLayout_2.addItem(self.verticalSpacer)

        self.buttonBox = QDialogButtonBox(self.groupBox)
        self.buttonBox.setObjectName(u"buttonBox")
        self.buttonBox.setOrientation(Qt.Horizontal)
        self.buttonBox.setStandardButtons(QDialogButtonBox.Cancel|QDialogButtonBox.Ok)

        self.verticalLayout_2.addWidget(self.buttonBox)


        self.verticalLayout.addWidget(self.groupBox)


        self.retranslateUi(LoadDataSettingsDlg)
        self.buttonBox.accepted.connect(LoadDataSettingsDlg.accept)
        self.buttonBox.rejected.connect(LoadDataSettingsDlg.reject)

        QMetaObject.connectSlotsByName(LoadDataSettingsDlg)
    # setupUi

    def retranslateUi(self, LoadDataSettingsDlg):
        LoadDataSettingsDlg.setWindowTitle(QCoreApplication.translate("LoadDataSettingsDlg", u"Load Data Settings", None))
        self.groupBox.setTitle("")
        self.label.setText(QCoreApplication.translate("LoadDataSettingsDlg", u"Loader Path", None))
        self.pushButton_select_file.setText(QCoreApplication.translate("LoadDataSettingsDlg", u"select file", None))
        self.pushButton_select_folder.setText(QCoreApplication.translate("LoadDataSettingsDlg", u"select folder", None))
        self.label_2.setText(QCoreApplication.translate("LoadDataSettingsDlg", u"OCT Loader:", None))
        self.label_5.setText(QCoreApplication.translate("LoadDataSettingsDlg", u".", None))
        self.label_3.setText(QCoreApplication.translate("LoadDataSettingsDlg", u"OCTA Loader:", None))
        self.label_6.setText(QCoreApplication.translate("LoadDataSettingsDlg", u".", None))
        self.label_4.setText(QCoreApplication.translate("LoadDataSettingsDlg", u"Seg Loader:", None))
        self.label_7.setText(QCoreApplication.translate("LoadDataSettingsDlg", u".", None))
    # retranslateUi

