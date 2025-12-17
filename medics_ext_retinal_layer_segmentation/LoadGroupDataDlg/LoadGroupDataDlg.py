# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'LoadGroupDataDlg.ui'
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
from PySide6.QtWidgets import (QAbstractButton, QApplication, QDialog, QDialogButtonBox,
    QHBoxLayout, QLabel, QLineEdit, QListView,
    QPushButton, QSizePolicy, QVBoxLayout, QWidget)

class Ui_LoadGroupDataDlg(object):
    def setupUi(self, LoadGroupDataDlg):
        if not LoadGroupDataDlg.objectName():
            LoadGroupDataDlg.setObjectName(u"LoadGroupDataDlg")
        LoadGroupDataDlg.resize(496, 343)
        LoadGroupDataDlg.setModal(True)
        self.verticalLayout_4 = QVBoxLayout(LoadGroupDataDlg)
        self.verticalLayout_4.setObjectName(u"verticalLayout_4")
        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.label = QLabel(LoadGroupDataDlg)
        self.label.setObjectName(u"label")

        self.horizontalLayout.addWidget(self.label)

        self.lineEdit_filename = QLineEdit(LoadGroupDataDlg)
        self.lineEdit_filename.setObjectName(u"lineEdit_filename")

        self.horizontalLayout.addWidget(self.lineEdit_filename)

        self.pushButton_browser = QPushButton(LoadGroupDataDlg)
        self.pushButton_browser.setObjectName(u"pushButton_browser")

        self.horizontalLayout.addWidget(self.pushButton_browser)


        self.verticalLayout_4.addLayout(self.horizontalLayout)

        self.horizontalLayout_5 = QHBoxLayout()
        self.horizontalLayout_5.setObjectName(u"horizontalLayout_5")
        self.verticalLayout = QVBoxLayout()
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.horizontalLayout_2 = QHBoxLayout()
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.label_2 = QLabel(LoadGroupDataDlg)
        self.label_2.setObjectName(u"label_2")

        self.horizontalLayout_2.addWidget(self.label_2)

        self.lineEdit_oct_field = QLineEdit(LoadGroupDataDlg)
        self.lineEdit_oct_field.setObjectName(u"lineEdit_oct_field")
        self.lineEdit_oct_field.setReadOnly(True)

        self.horizontalLayout_2.addWidget(self.lineEdit_oct_field)


        self.verticalLayout.addLayout(self.horizontalLayout_2)

        self.listView_source_oct = QListView(LoadGroupDataDlg)
        self.listView_source_oct.setObjectName(u"listView_source_oct")

        self.verticalLayout.addWidget(self.listView_source_oct)


        self.horizontalLayout_5.addLayout(self.verticalLayout)

        self.verticalLayout_2 = QVBoxLayout()
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.horizontalLayout_3 = QHBoxLayout()
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.label_3 = QLabel(LoadGroupDataDlg)
        self.label_3.setObjectName(u"label_3")

        self.horizontalLayout_3.addWidget(self.label_3)

        self.lineEdit_octa_field = QLineEdit(LoadGroupDataDlg)
        self.lineEdit_octa_field.setObjectName(u"lineEdit_octa_field")
        self.lineEdit_octa_field.setReadOnly(True)

        self.horizontalLayout_3.addWidget(self.lineEdit_octa_field)


        self.verticalLayout_2.addLayout(self.horizontalLayout_3)

        self.listView_source_octa = QListView(LoadGroupDataDlg)
        self.listView_source_octa.setObjectName(u"listView_source_octa")

        self.verticalLayout_2.addWidget(self.listView_source_octa)


        self.horizontalLayout_5.addLayout(self.verticalLayout_2)

        self.verticalLayout_3 = QVBoxLayout()
        self.verticalLayout_3.setObjectName(u"verticalLayout_3")
        self.horizontalLayout_4 = QHBoxLayout()
        self.horizontalLayout_4.setObjectName(u"horizontalLayout_4")
        self.label_4 = QLabel(LoadGroupDataDlg)
        self.label_4.setObjectName(u"label_4")

        self.horizontalLayout_4.addWidget(self.label_4)

        self.lineEdit_seg_field = QLineEdit(LoadGroupDataDlg)
        self.lineEdit_seg_field.setObjectName(u"lineEdit_seg_field")
        self.lineEdit_seg_field.setReadOnly(True)

        self.horizontalLayout_4.addWidget(self.lineEdit_seg_field)


        self.verticalLayout_3.addLayout(self.horizontalLayout_4)

        self.listView_source_seg = QListView(LoadGroupDataDlg)
        self.listView_source_seg.setObjectName(u"listView_source_seg")

        self.verticalLayout_3.addWidget(self.listView_source_seg)


        self.horizontalLayout_5.addLayout(self.verticalLayout_3)


        self.verticalLayout_4.addLayout(self.horizontalLayout_5)

        self.buttonBox = QDialogButtonBox(LoadGroupDataDlg)
        self.buttonBox.setObjectName(u"buttonBox")
        self.buttonBox.setOrientation(Qt.Orientation.Horizontal)
        self.buttonBox.setStandardButtons(QDialogButtonBox.StandardButton.Cancel|QDialogButtonBox.StandardButton.Ok)

        self.verticalLayout_4.addWidget(self.buttonBox)


        self.retranslateUi(LoadGroupDataDlg)
        self.buttonBox.accepted.connect(LoadGroupDataDlg.accept)
        self.buttonBox.rejected.connect(LoadGroupDataDlg.reject)

        QMetaObject.connectSlotsByName(LoadGroupDataDlg)
    # setupUi

    def retranslateUi(self, LoadGroupDataDlg):
        LoadGroupDataDlg.setWindowTitle(QCoreApplication.translate("LoadGroupDataDlg", u"Dialog", None))
        self.label.setText(QCoreApplication.translate("LoadGroupDataDlg", u"File:", None))
        self.pushButton_browser.setText(QCoreApplication.translate("LoadGroupDataDlg", u"...", None))
        self.label_2.setText(QCoreApplication.translate("LoadGroupDataDlg", u"OCT:", None))
        self.label_3.setText(QCoreApplication.translate("LoadGroupDataDlg", u"OCTA", None))
        self.label_4.setText(QCoreApplication.translate("LoadGroupDataDlg", u"Seg:", None))
    # retranslateUi

