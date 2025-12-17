# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'CalculateResolutionDlg.ui'
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
    QFrame, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QSizePolicy, QVBoxLayout, QWidget)

class Ui_CalculateResolutionDlg(object):
    def setupUi(self, CalculateResolutionDlg):
        if not CalculateResolutionDlg.objectName():
            CalculateResolutionDlg.setObjectName(u"CalculateResolutionDlg")
        CalculateResolutionDlg.resize(400, 299)
        self.verticalLayout = QVBoxLayout(CalculateResolutionDlg)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout.setContentsMargins(6, 6, 6, 6)
        self.label_9 = QLabel(CalculateResolutionDlg)
        self.label_9.setObjectName(u"label_9")

        self.verticalLayout.addWidget(self.label_9)

        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.label = QLabel(CalculateResolutionDlg)
        self.label.setObjectName(u"label")

        self.horizontalLayout.addWidget(self.label)

        self.lineEdit = QLineEdit(CalculateResolutionDlg)
        self.lineEdit.setObjectName(u"lineEdit")

        self.horizontalLayout.addWidget(self.lineEdit)

        self.label_2 = QLabel(CalculateResolutionDlg)
        self.label_2.setObjectName(u"label_2")

        self.horizontalLayout.addWidget(self.label_2)

        self.lineEdit_2 = QLineEdit(CalculateResolutionDlg)
        self.lineEdit_2.setObjectName(u"lineEdit_2")

        self.horizontalLayout.addWidget(self.lineEdit_2)

        self.label_3 = QLabel(CalculateResolutionDlg)
        self.label_3.setObjectName(u"label_3")

        self.horizontalLayout.addWidget(self.label_3)


        self.verticalLayout.addLayout(self.horizontalLayout)

        self.horizontalLayout_5 = QHBoxLayout()
        self.horizontalLayout_5.setObjectName(u"horizontalLayout_5")
        self.label_8 = QLabel(CalculateResolutionDlg)
        self.label_8.setObjectName(u"label_8")

        self.horizontalLayout_5.addWidget(self.label_8)

        self.lineEdit_6 = QLineEdit(CalculateResolutionDlg)
        self.lineEdit_6.setObjectName(u"lineEdit_6")
        self.lineEdit_6.setReadOnly(False)

        self.horizontalLayout_5.addWidget(self.lineEdit_6)

        self.label_10 = QLabel(CalculateResolutionDlg)
        self.label_10.setObjectName(u"label_10")

        self.horizontalLayout_5.addWidget(self.label_10)

        self.pushButton_help = QPushButton(CalculateResolutionDlg)
        self.pushButton_help.setObjectName(u"pushButton_help")
        icon = QIcon(QIcon.fromTheme(u"help"))
        self.pushButton_help.setIcon(icon)

        self.horizontalLayout_5.addWidget(self.pushButton_help)


        self.verticalLayout.addLayout(self.horizontalLayout_5)

        self.line = QFrame(CalculateResolutionDlg)
        self.line.setObjectName(u"line")
        self.line.setFrameShape(QFrame.Shape.HLine)
        self.line.setFrameShadow(QFrame.Shadow.Sunken)

        self.verticalLayout.addWidget(self.line)

        self.label_14 = QLabel(CalculateResolutionDlg)
        self.label_14.setObjectName(u"label_14")

        self.verticalLayout.addWidget(self.label_14)

        self.horizontalLayout_3 = QHBoxLayout()
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.label_7 = QLabel(CalculateResolutionDlg)
        self.label_7.setObjectName(u"label_7")

        self.horizontalLayout_3.addWidget(self.label_7)

        self.lineEdit_5 = QLineEdit(CalculateResolutionDlg)
        self.lineEdit_5.setObjectName(u"lineEdit_5")
        self.lineEdit_5.setReadOnly(True)

        self.horizontalLayout_3.addWidget(self.lineEdit_5)

        self.label_11 = QLabel(CalculateResolutionDlg)
        self.label_11.setObjectName(u"label_11")

        self.horizontalLayout_3.addWidget(self.label_11)


        self.verticalLayout.addLayout(self.horizontalLayout_3)

        self.horizontalLayout_6 = QHBoxLayout()
        self.horizontalLayout_6.setObjectName(u"horizontalLayout_6")
        self.label_12 = QLabel(CalculateResolutionDlg)
        self.label_12.setObjectName(u"label_12")

        self.horizontalLayout_6.addWidget(self.label_12)

        self.lineEdit_8 = QLineEdit(CalculateResolutionDlg)
        self.lineEdit_8.setObjectName(u"lineEdit_8")
        self.lineEdit_8.setReadOnly(True)

        self.horizontalLayout_6.addWidget(self.lineEdit_8)

        self.label_13 = QLabel(CalculateResolutionDlg)
        self.label_13.setObjectName(u"label_13")

        self.horizontalLayout_6.addWidget(self.label_13)


        self.verticalLayout.addLayout(self.horizontalLayout_6)

        self.buttonBox = QDialogButtonBox(CalculateResolutionDlg)
        self.buttonBox.setObjectName(u"buttonBox")
        self.buttonBox.setOrientation(Qt.Orientation.Horizontal)
        self.buttonBox.setStandardButtons(QDialogButtonBox.StandardButton.Cancel|QDialogButtonBox.StandardButton.Ok)

        self.verticalLayout.addWidget(self.buttonBox)


        self.retranslateUi(CalculateResolutionDlg)
        self.buttonBox.accepted.connect(CalculateResolutionDlg.accept)
        self.buttonBox.rejected.connect(CalculateResolutionDlg.reject)

        QMetaObject.connectSlotsByName(CalculateResolutionDlg)
    # setupUi

    def retranslateUi(self, CalculateResolutionDlg):
        CalculateResolutionDlg.setWindowTitle(QCoreApplication.translate("CalculateResolutionDlg", u"Scan Resolution", None))
        self.label_9.setText(QCoreApplication.translate("CalculateResolutionDlg", u"Please confirm the following is correct:", None))
        self.label.setText(QCoreApplication.translate("CalculateResolutionDlg", u"<html><head/><body><p><span style=\" color:#ff0000;\">* </span>Scan size in mm:</p></body></html>", None))
        self.lineEdit.setText(QCoreApplication.translate("CalculateResolutionDlg", u"1", None))
        self.label_2.setText(QCoreApplication.translate("CalculateResolutionDlg", u",", None))
        self.lineEdit_2.setText(QCoreApplication.translate("CalculateResolutionDlg", u"1", None))
        self.label_3.setText(QCoreApplication.translate("CalculateResolutionDlg", u"(width, height)", None))
        self.label_8.setText(QCoreApplication.translate("CalculateResolutionDlg", u"<html><head/><body><p><span style=\" color:#ff0000;\">*</span> Axel resolution:</p></body></html>", None))
        self.lineEdit_6.setText(QCoreApplication.translate("CalculateResolutionDlg", u"1", None))
        self.label_10.setText(QCoreApplication.translate("CalculateResolutionDlg", u"mm/pixel", None))
        self.pushButton_help.setText("")
        self.label_14.setText(QCoreApplication.translate("CalculateResolutionDlg", u"Calculated lateral resolution:", None))
        self.label_7.setText(QCoreApplication.translate("CalculateResolutionDlg", u"<html><head/><body><p>Lateral resolution (width):</p></body></html>", None))
        self.lineEdit_5.setText(QCoreApplication.translate("CalculateResolutionDlg", u"1", None))
        self.label_11.setText(QCoreApplication.translate("CalculateResolutionDlg", u" mm/pixel", None))
        self.label_12.setText(QCoreApplication.translate("CalculateResolutionDlg", u"<html><head/><body><p>Lateral resolution (height):</p></body></html>", None))
        self.lineEdit_8.setText(QCoreApplication.translate("CalculateResolutionDlg", u"1", None))
        self.label_13.setText(QCoreApplication.translate("CalculateResolutionDlg", u" mm/pixel", None))
    # retranslateUi

