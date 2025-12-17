# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'LoadMedDataDlg.ui'
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
from PySide6.QtWidgets import (QAbstractButton, QApplication, QDialogButtonBox, QHBoxLayout,
    QLabel, QLineEdit, QListView, QPushButton,
    QSizePolicy, QVBoxLayout, QWidget)

class Ui_LoadMedData(object):
    def setupUi(self, LoadMedData):
        if not LoadMedData.objectName():
            LoadMedData.setObjectName(u"LoadMedData")
        LoadMedData.resize(330, 312)
        self.verticalLayout_2 = QVBoxLayout(LoadMedData)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.verticalLayout_2.setContentsMargins(6, 6, 6, 6)
        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.label = QLabel(LoadMedData)
        self.label.setObjectName(u"label")

        self.horizontalLayout.addWidget(self.label)

        self.lineEdit_filename = QLineEdit(LoadMedData)
        self.lineEdit_filename.setObjectName(u"lineEdit_filename")

        self.horizontalLayout.addWidget(self.lineEdit_filename)

        self.pushButton_browser = QPushButton(LoadMedData)
        self.pushButton_browser.setObjectName(u"pushButton_browser")

        self.horizontalLayout.addWidget(self.pushButton_browser)


        self.verticalLayout_2.addLayout(self.horizontalLayout)

        self.verticalLayout = QVBoxLayout()
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.horizontalLayout_2 = QHBoxLayout()
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.label_2 = QLabel(LoadMedData)
        self.label_2.setObjectName(u"label_2")

        self.horizontalLayout_2.addWidget(self.label_2)

        self.lineEdit_selected_field = QLineEdit(LoadMedData)
        self.lineEdit_selected_field.setObjectName(u"lineEdit_selected_field")
        self.lineEdit_selected_field.setReadOnly(True)

        self.horizontalLayout_2.addWidget(self.lineEdit_selected_field)


        self.verticalLayout.addLayout(self.horizontalLayout_2)

        self.listView_source = QListView(LoadMedData)
        self.listView_source.setObjectName(u"listView_source")

        self.verticalLayout.addWidget(self.listView_source)


        self.verticalLayout_2.addLayout(self.verticalLayout)

        self.buttonBox = QDialogButtonBox(LoadMedData)
        self.buttonBox.setObjectName(u"buttonBox")
        self.buttonBox.setStandardButtons(QDialogButtonBox.StandardButton.Cancel|QDialogButtonBox.StandardButton.Ok)

        self.verticalLayout_2.addWidget(self.buttonBox)


        self.retranslateUi(LoadMedData)

        QMetaObject.connectSlotsByName(LoadMedData)
    # setupUi

    def retranslateUi(self, LoadMedData):
        LoadMedData.setWindowTitle(QCoreApplication.translate("LoadMedData", u"Load Med Data", None))
        self.label.setText(QCoreApplication.translate("LoadMedData", u"File:", None))
        self.pushButton_browser.setText(QCoreApplication.translate("LoadMedData", u"...", None))
        self.label_2.setText(QCoreApplication.translate("LoadMedData", u"Selected Data field:", None))
    # retranslateUi

