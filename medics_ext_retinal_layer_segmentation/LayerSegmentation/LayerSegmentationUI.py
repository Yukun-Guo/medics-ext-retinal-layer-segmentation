# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'LayerSegmentationUI.ui'
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
from PySide6.QtWidgets import (QApplication, QComboBox, QDoubleSpinBox, QHBoxLayout,
    QLabel, QPushButton, QSizePolicy, QSpacerItem,
    QVBoxLayout, QWidget)

class Ui_LayerSegmentation(object):
    def setupUi(self, LayerSegmentation):
        if not LayerSegmentation.objectName():
            LayerSegmentation.setObjectName(u"LayerSegmentation")
        LayerSegmentation.resize(974, 574)
        self.horizontalLayout = QHBoxLayout(LayerSegmentation)
        self.horizontalLayout.setSpacing(2)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_3 = QHBoxLayout()
        self.horizontalLayout_3.setSpacing(3)
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.verticalLayout = QVBoxLayout()
        self.verticalLayout.setSpacing(1)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.horizontalLayout_33 = QHBoxLayout()
        self.horizontalLayout_33.setSpacing(2)
        self.horizontalLayout_33.setObjectName(u"horizontalLayout_33")
        self.horizontalLayout_33.setContentsMargins(1, -1, 1, 0)
        self.pushButton_hide_right = QPushButton(LayerSegmentation)
        self.pushButton_hide_right.setObjectName(u"pushButton_hide_right")
        self.pushButton_hide_right.setMinimumSize(QSize(22, 22))
        self.pushButton_hide_right.setMaximumSize(QSize(22, 22))

        self.horizontalLayout_33.addWidget(self.pushButton_hide_right)

        self.pushButton_layoutdirection = QPushButton(LayerSegmentation)
        self.pushButton_layoutdirection.setObjectName(u"pushButton_layoutdirection")
        self.pushButton_layoutdirection.setMinimumSize(QSize(22, 22))
        self.pushButton_layoutdirection.setMaximumSize(QSize(22, 22))

        self.horizontalLayout_33.addWidget(self.pushButton_layoutdirection)

        self.doubleSpinBox_bframeA_cmbar_low = QDoubleSpinBox(LayerSegmentation)
        self.doubleSpinBox_bframeA_cmbar_low.setObjectName(u"doubleSpinBox_bframeA_cmbar_low")
        self.doubleSpinBox_bframeA_cmbar_low.setMinimumSize(QSize(0, 21))
        self.doubleSpinBox_bframeA_cmbar_low.setMaximumSize(QSize(65, 16777215))
        self.doubleSpinBox_bframeA_cmbar_low.setDecimals(3)
        self.doubleSpinBox_bframeA_cmbar_low.setMaximum(0.999000000000000)
        self.doubleSpinBox_bframeA_cmbar_low.setSingleStep(0.010000000000000)

        self.horizontalLayout_33.addWidget(self.doubleSpinBox_bframeA_cmbar_low)

        self.doubleSpinBox_bframeA_cmbar_high = QDoubleSpinBox(LayerSegmentation)
        self.doubleSpinBox_bframeA_cmbar_high.setObjectName(u"doubleSpinBox_bframeA_cmbar_high")
        self.doubleSpinBox_bframeA_cmbar_high.setMinimumSize(QSize(0, 21))
        self.doubleSpinBox_bframeA_cmbar_high.setMaximumSize(QSize(65, 16777215))
        self.doubleSpinBox_bframeA_cmbar_high.setDecimals(3)
        self.doubleSpinBox_bframeA_cmbar_high.setMinimum(0.001000000000000)
        self.doubleSpinBox_bframeA_cmbar_high.setMaximum(1.000000000000000)
        self.doubleSpinBox_bframeA_cmbar_high.setSingleStep(0.010000000000000)
        self.doubleSpinBox_bframeA_cmbar_high.setValue(1.000000000000000)

        self.horizontalLayout_33.addWidget(self.doubleSpinBox_bframeA_cmbar_high)


        self.verticalLayout.addLayout(self.horizontalLayout_33)


        self.horizontalLayout_3.addLayout(self.verticalLayout)

        self.verticalLayout_2 = QVBoxLayout()
        self.verticalLayout_2.setSpacing(1)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.horizontalLayout_36 = QHBoxLayout()
        self.horizontalLayout_36.setSpacing(2)
        self.horizontalLayout_36.setObjectName(u"horizontalLayout_36")
        self.horizontalLayout_36.setContentsMargins(1, -1, 1, -1)
        self.pushButton_bframe_type = QPushButton(LayerSegmentation)
        self.pushButton_bframe_type.setObjectName(u"pushButton_bframe_type")
        self.pushButton_bframe_type.setMinimumSize(QSize(22, 22))
        self.pushButton_bframe_type.setMaximumSize(QSize(22, 22))

        self.horizontalLayout_36.addWidget(self.pushButton_bframe_type)

        self.doubleSpinBox_bframeB_cmbar_low = QDoubleSpinBox(LayerSegmentation)
        self.doubleSpinBox_bframeB_cmbar_low.setObjectName(u"doubleSpinBox_bframeB_cmbar_low")
        self.doubleSpinBox_bframeB_cmbar_low.setMinimumSize(QSize(0, 21))
        self.doubleSpinBox_bframeB_cmbar_low.setMaximumSize(QSize(60, 16777215))
        self.doubleSpinBox_bframeB_cmbar_low.setDecimals(3)
        self.doubleSpinBox_bframeB_cmbar_low.setMaximum(0.999000000000000)
        self.doubleSpinBox_bframeB_cmbar_low.setSingleStep(0.010000000000000)

        self.horizontalLayout_36.addWidget(self.doubleSpinBox_bframeB_cmbar_low)

        self.doubleSpinBox_bframeB_cmbar_high = QDoubleSpinBox(LayerSegmentation)
        self.doubleSpinBox_bframeB_cmbar_high.setObjectName(u"doubleSpinBox_bframeB_cmbar_high")
        self.doubleSpinBox_bframeB_cmbar_high.setMinimumSize(QSize(0, 21))
        self.doubleSpinBox_bframeB_cmbar_high.setMaximumSize(QSize(60, 16777215))
        self.doubleSpinBox_bframeB_cmbar_high.setDecimals(3)
        self.doubleSpinBox_bframeB_cmbar_high.setMinimum(0.001000000000000)
        self.doubleSpinBox_bframeB_cmbar_high.setMaximum(1.000000000000000)
        self.doubleSpinBox_bframeB_cmbar_high.setSingleStep(0.010000000000000)
        self.doubleSpinBox_bframeB_cmbar_high.setValue(1.000000000000000)

        self.horizontalLayout_36.addWidget(self.doubleSpinBox_bframeB_cmbar_high)

        self.pushButton_hide_left = QPushButton(LayerSegmentation)
        self.pushButton_hide_left.setObjectName(u"pushButton_hide_left")
        self.pushButton_hide_left.setMinimumSize(QSize(22, 22))
        self.pushButton_hide_left.setMaximumSize(QSize(22, 22))

        self.horizontalLayout_36.addWidget(self.pushButton_hide_left)


        self.verticalLayout_2.addLayout(self.horizontalLayout_36)


        self.horizontalLayout_3.addLayout(self.verticalLayout_2)

        self.horizontalLayout_3.setStretch(0, 1)
        self.horizontalLayout_3.setStretch(1, 1)

        self.horizontalLayout.addLayout(self.horizontalLayout_3)

        self.verticalLayout_3 = QVBoxLayout()
        self.verticalLayout_3.setSpacing(2)
        self.verticalLayout_3.setObjectName(u"verticalLayout_3")
        self.horizontalLayout_31 = QHBoxLayout()
        self.horizontalLayout_31.setSpacing(0)
        self.horizontalLayout_31.setObjectName(u"horizontalLayout_31")
        self.comboBox_enfaceA_Slab = QComboBox(LayerSegmentation)
        self.comboBox_enfaceA_Slab.setObjectName(u"comboBox_enfaceA_Slab")
        self.comboBox_enfaceA_Slab.setMinimumSize(QSize(90, 21))
        self.comboBox_enfaceA_Slab.setMaximumSize(QSize(120, 16777215))
        font = QFont()
        font.setPointSize(10)
        self.comboBox_enfaceA_Slab.setFont(font)

        self.horizontalLayout_31.addWidget(self.comboBox_enfaceA_Slab)

        self.doubleSpinBox_enfaceA_cmbar_low = QDoubleSpinBox(LayerSegmentation)
        self.doubleSpinBox_enfaceA_cmbar_low.setObjectName(u"doubleSpinBox_enfaceA_cmbar_low")
        self.doubleSpinBox_enfaceA_cmbar_low.setMinimumSize(QSize(0, 21))
        self.doubleSpinBox_enfaceA_cmbar_low.setMaximumSize(QSize(65, 16777215))
        self.doubleSpinBox_enfaceA_cmbar_low.setDecimals(3)
        self.doubleSpinBox_enfaceA_cmbar_low.setMaximum(0.999000000000000)
        self.doubleSpinBox_enfaceA_cmbar_low.setSingleStep(0.010000000000000)

        self.horizontalLayout_31.addWidget(self.doubleSpinBox_enfaceA_cmbar_low)

        self.doubleSpinBox_enfaceA_cmbar_high = QDoubleSpinBox(LayerSegmentation)
        self.doubleSpinBox_enfaceA_cmbar_high.setObjectName(u"doubleSpinBox_enfaceA_cmbar_high")
        self.doubleSpinBox_enfaceA_cmbar_high.setMinimumSize(QSize(0, 21))
        self.doubleSpinBox_enfaceA_cmbar_high.setMaximumSize(QSize(65, 16777215))
        self.doubleSpinBox_enfaceA_cmbar_high.setDecimals(3)
        self.doubleSpinBox_enfaceA_cmbar_high.setMinimum(0.001000000000000)
        self.doubleSpinBox_enfaceA_cmbar_high.setMaximum(1.000000000000000)
        self.doubleSpinBox_enfaceA_cmbar_high.setSingleStep(0.010000000000000)
        self.doubleSpinBox_enfaceA_cmbar_high.setValue(1.000000000000000)

        self.horizontalLayout_31.addWidget(self.doubleSpinBox_enfaceA_cmbar_high)

        self.pushButton_transp = QPushButton(LayerSegmentation)
        self.pushButton_transp.setObjectName(u"pushButton_transp")
        self.pushButton_transp.setMaximumSize(QSize(24, 24))
        self.pushButton_transp.setStyleSheet(u"border:none")
        self.pushButton_transp.setIconSize(QSize(22, 22))

        self.horizontalLayout_31.addWidget(self.pushButton_transp)


        self.verticalLayout_3.addLayout(self.horizontalLayout_31)

        self.horizontalLayout_32 = QHBoxLayout()
        self.horizontalLayout_32.setSpacing(0)
        self.horizontalLayout_32.setObjectName(u"horizontalLayout_32")
        self.label_10 = QLabel(LayerSegmentation)
        self.label_10.setObjectName(u"label_10")

        self.horizontalLayout_32.addWidget(self.label_10)

        self.horizontalSpacer_3 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_32.addItem(self.horizontalSpacer_3)

        self.pushButton_reset = QPushButton(LayerSegmentation)
        self.pushButton_reset.setObjectName(u"pushButton_reset")
        self.pushButton_reset.setMaximumSize(QSize(24, 24))
        self.pushButton_reset.setStyleSheet(u"border:none")
        self.pushButton_reset.setIconSize(QSize(22, 22))

        self.horizontalLayout_32.addWidget(self.pushButton_reset)


        self.verticalLayout_3.addLayout(self.horizontalLayout_32)


        self.horizontalLayout.addLayout(self.verticalLayout_3)

        self.horizontalLayout.setStretch(0, 5)
        self.horizontalLayout.setStretch(1, 3)

        self.retranslateUi(LayerSegmentation)

        QMetaObject.connectSlotsByName(LayerSegmentation)
    # setupUi

    def retranslateUi(self, LayerSegmentation):
        LayerSegmentation.setWindowTitle(QCoreApplication.translate("LayerSegmentation", u"Form", None))
        self.pushButton_hide_right.setText("")
        self.pushButton_layoutdirection.setText("")
        self.pushButton_bframe_type.setText("")
        self.pushButton_hide_left.setText("")
        self.pushButton_transp.setText("")
        self.label_10.setText(QCoreApplication.translate("LayerSegmentation", u"3D Viewer", None))
        self.pushButton_reset.setText("")
    # retranslateUi

