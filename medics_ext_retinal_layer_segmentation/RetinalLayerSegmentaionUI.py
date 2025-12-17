# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'RetinalLayerSegmentaionUI.ui'
##
## Created by: Qt User Interface Compiler version 6.8.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QAction, QBrush, QColor, QConicalGradient,
    QCursor, QFont, QFontDatabase, QGradient,
    QIcon, QImage, QKeySequence, QLinearGradient,
    QPainter, QPalette, QPixmap, QRadialGradient,
    QTransform)
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QGroupBox,
    QHBoxLayout, QLabel, QLineEdit, QMainWindow,
    QPushButton, QScrollArea, QSizePolicy, QSpinBox,
    QStackedWidget, QStatusBar, QTabWidget, QToolBar,
    QVBoxLayout, QWidget)

class Ui_RetinalLayerSegmentaion(object):
    def setupUi(self, RetinalLayerSegmentaion):
        if not RetinalLayerSegmentaion.objectName():
            RetinalLayerSegmentaion.setObjectName(u"RetinalLayerSegmentaion")
        RetinalLayerSegmentaion.resize(1138, 854)
        self.actionSave = QAction(RetinalLayerSegmentaion)
        self.actionSave.setObjectName(u"actionSave")
        self.actionSave.setMenuRole(QAction.MenuRole.NoRole)
        self.actionCleanUp = QAction(RetinalLayerSegmentaion)
        self.actionCleanUp.setObjectName(u"actionCleanUp")
        self.actionCleanUp.setMenuRole(QAction.MenuRole.NoRole)
        self.centralwidget = QWidget(RetinalLayerSegmentaion)
        self.centralwidget.setObjectName(u"centralwidget")
        self.horizontalLayout_15 = QHBoxLayout(self.centralwidget)
        self.horizontalLayout_15.setSpacing(1)
        self.horizontalLayout_15.setObjectName(u"horizontalLayout_15")
        self.horizontalLayout_15.setContentsMargins(1, 1, 1, 2)
        self.widget_side_panel = QWidget(self.centralwidget)
        self.widget_side_panel.setObjectName(u"widget_side_panel")
        self.widget_side_panel.setMinimumSize(QSize(253, 0))
        self.widget_side_panel.setMaximumSize(QSize(253, 16777215))
        self.widget_side_panel.setStyleSheet(u"b")
        self.verticalLayout_3 = QVBoxLayout(self.widget_side_panel)
        self.verticalLayout_3.setSpacing(5)
        self.verticalLayout_3.setObjectName(u"verticalLayout_3")
        self.verticalLayout_3.setContentsMargins(1, 1, 1, 1)
        self.groupBox_4 = QGroupBox(self.widget_side_panel)
        self.groupBox_4.setObjectName(u"groupBox_4")
        self.groupBox_4.setMinimumSize(QSize(0, 150))
        self.groupBox_4.setMaximumSize(QSize(16777215, 150))
        font = QFont()
        font.setPointSize(10)
        self.groupBox_4.setFont(font)
        self.horizontalLayout_16 = QHBoxLayout(self.groupBox_4)
        self.horizontalLayout_16.setSpacing(0)
        self.horizontalLayout_16.setObjectName(u"horizontalLayout_16")
        self.horizontalLayout_16.setContentsMargins(0, 0, 0, 0)
        self.tabWidget_dataSource = QTabWidget(self.groupBox_4)
        self.tabWidget_dataSource.setObjectName(u"tabWidget_dataSource")
        self.tabWidget_dataSource.setFont(font)
        self.tabWidget_dataSource.setStyleSheet(u"")
        self.tab = QWidget()
        self.tab.setObjectName(u"tab")
        self.verticalLayout_10 = QVBoxLayout(self.tab)
        self.verticalLayout_10.setSpacing(3)
        self.verticalLayout_10.setObjectName(u"verticalLayout_10")
        self.verticalLayout_10.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_17 = QHBoxLayout()
        self.horizontalLayout_17.setSpacing(2)
        self.horizontalLayout_17.setObjectName(u"horizontalLayout_17")
        self.label_5 = QLabel(self.tab)
        self.label_5.setObjectName(u"label_5")
        self.label_5.setMinimumSize(QSize(57, 0))
        self.label_5.setMaximumSize(QSize(57, 16777215))
        self.label_5.setFont(font)
        self.label_5.setStyleSheet(u"border:none")
        self.label_5.setLineWidth(0)

        self.horizontalLayout_17.addWidget(self.label_5)

        self.comboBox_OCT = QComboBox(self.tab)
        self.comboBox_OCT.addItem("")
        self.comboBox_OCT.setObjectName(u"comboBox_OCT")
        self.comboBox_OCT.setMinimumSize(QSize(0, 21))
        self.comboBox_OCT.setMaximumSize(QSize(16777215, 16777215))

        self.horizontalLayout_17.addWidget(self.comboBox_OCT)

        self.pushButton_refreshDatalist = QPushButton(self.tab)
        self.pushButton_refreshDatalist.setObjectName(u"pushButton_refreshDatalist")
        self.pushButton_refreshDatalist.setMaximumSize(QSize(21, 21))
        icon = QIcon(QIcon.fromTheme(u"view-refresh"))
        self.pushButton_refreshDatalist.setIcon(icon)

        self.horizontalLayout_17.addWidget(self.pushButton_refreshDatalist)


        self.verticalLayout_10.addLayout(self.horizontalLayout_17)

        self.horizontalLayout_18 = QHBoxLayout()
        self.horizontalLayout_18.setSpacing(2)
        self.horizontalLayout_18.setObjectName(u"horizontalLayout_18")
        self.label_6 = QLabel(self.tab)
        self.label_6.setObjectName(u"label_6")
        self.label_6.setMinimumSize(QSize(57, 0))
        self.label_6.setMaximumSize(QSize(57, 16777215))
        self.label_6.setFont(font)
        self.label_6.setStyleSheet(u"border:none")
        self.label_6.setLineWidth(0)

        self.horizontalLayout_18.addWidget(self.label_6)

        self.comboBox_OCTA = QComboBox(self.tab)
        self.comboBox_OCTA.addItem("")
        self.comboBox_OCTA.setObjectName(u"comboBox_OCTA")
        self.comboBox_OCTA.setMinimumSize(QSize(0, 21))
        self.comboBox_OCTA.setMaximumSize(QSize(16777215, 16777215))

        self.horizontalLayout_18.addWidget(self.comboBox_OCTA)


        self.verticalLayout_10.addLayout(self.horizontalLayout_18)

        self.horizontalLayout_19 = QHBoxLayout()
        self.horizontalLayout_19.setSpacing(2)
        self.horizontalLayout_19.setObjectName(u"horizontalLayout_19")
        self.label_7 = QLabel(self.tab)
        self.label_7.setObjectName(u"label_7")
        self.label_7.setMinimumSize(QSize(57, 0))
        self.label_7.setMaximumSize(QSize(57, 16777215))
        self.label_7.setFont(font)
        self.label_7.setStyleSheet(u"border:none")
        self.label_7.setLineWidth(0)

        self.horizontalLayout_19.addWidget(self.label_7)

        self.comboBox_Seg = QComboBox(self.tab)
        self.comboBox_Seg.addItem("")
        self.comboBox_Seg.setObjectName(u"comboBox_Seg")
        self.comboBox_Seg.setMinimumSize(QSize(0, 21))
        self.comboBox_Seg.setMaximumSize(QSize(16777215, 16777215))

        self.horizontalLayout_19.addWidget(self.comboBox_Seg)


        self.verticalLayout_10.addLayout(self.horizontalLayout_19)

        self.tabWidget_dataSource.addTab(self.tab, "")
        self.tab_2 = QWidget()
        self.tab_2.setObjectName(u"tab_2")
        self.verticalLayout_4 = QVBoxLayout(self.tab_2)
        self.verticalLayout_4.setSpacing(3)
        self.verticalLayout_4.setObjectName(u"verticalLayout_4")
        self.verticalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_23 = QHBoxLayout()
        self.horizontalLayout_23.setSpacing(2)
        self.horizontalLayout_23.setObjectName(u"horizontalLayout_23")
        self.label_8 = QLabel(self.tab_2)
        self.label_8.setObjectName(u"label_8")
        self.label_8.setMinimumSize(QSize(75, 0))
        self.label_8.setMaximumSize(QSize(75, 16777215))
        self.label_8.setFont(font)
        self.label_8.setStyleSheet(u"border:none")
        self.label_8.setLineWidth(0)

        self.horizontalLayout_23.addWidget(self.label_8)

        self.lineEdit_OCTFilename = QLineEdit(self.tab_2)
        self.lineEdit_OCTFilename.setObjectName(u"lineEdit_OCTFilename")
        self.lineEdit_OCTFilename.setMinimumSize(QSize(0, 21))
        self.lineEdit_OCTFilename.setFont(font)

        self.horizontalLayout_23.addWidget(self.lineEdit_OCTFilename)

        self.pushButton_openOCT = QPushButton(self.tab_2)
        self.pushButton_openOCT.setObjectName(u"pushButton_openOCT")
        self.pushButton_openOCT.setMinimumSize(QSize(0, 21))
        self.pushButton_openOCT.setStyleSheet(u"border:none")
        icon1 = QIcon(QIcon.fromTheme(u"folder-open"))
        self.pushButton_openOCT.setIcon(icon1)

        self.horizontalLayout_23.addWidget(self.pushButton_openOCT)


        self.verticalLayout_4.addLayout(self.horizontalLayout_23)

        self.horizontalLayout_27 = QHBoxLayout()
        self.horizontalLayout_27.setSpacing(2)
        self.horizontalLayout_27.setObjectName(u"horizontalLayout_27")
        self.checkBox_OCTA = QCheckBox(self.tab_2)
        self.checkBox_OCTA.setObjectName(u"checkBox_OCTA")
        self.checkBox_OCTA.setMinimumSize(QSize(15, 0))
        self.checkBox_OCTA.setMaximumSize(QSize(15, 16777215))
        self.checkBox_OCTA.setLayoutDirection(Qt.LayoutDirection.LeftToRight)
        self.checkBox_OCTA.setAutoFillBackground(False)
        self.checkBox_OCTA.setChecked(True)

        self.horizontalLayout_27.addWidget(self.checkBox_OCTA)

        self.label_10 = QLabel(self.tab_2)
        self.label_10.setObjectName(u"label_10")
        self.label_10.setMinimumSize(QSize(58, 0))
        self.label_10.setMaximumSize(QSize(60, 16777215))
        self.label_10.setFont(font)
        self.label_10.setStyleSheet(u"border:none")
        self.label_10.setLineWidth(0)

        self.horizontalLayout_27.addWidget(self.label_10)

        self.lineEdit_OCTAFilename = QLineEdit(self.tab_2)
        self.lineEdit_OCTAFilename.setObjectName(u"lineEdit_OCTAFilename")
        self.lineEdit_OCTAFilename.setMinimumSize(QSize(0, 21))
        self.lineEdit_OCTAFilename.setFont(font)

        self.horizontalLayout_27.addWidget(self.lineEdit_OCTAFilename)

        self.pushButton_openOCTA = QPushButton(self.tab_2)
        self.pushButton_openOCTA.setObjectName(u"pushButton_openOCTA")
        self.pushButton_openOCTA.setMinimumSize(QSize(0, 21))
        self.pushButton_openOCTA.setStyleSheet(u"border:none")
        self.pushButton_openOCTA.setIcon(icon1)

        self.horizontalLayout_27.addWidget(self.pushButton_openOCTA)


        self.verticalLayout_4.addLayout(self.horizontalLayout_27)

        self.horizontalLayout_28 = QHBoxLayout()
        self.horizontalLayout_28.setSpacing(2)
        self.horizontalLayout_28.setObjectName(u"horizontalLayout_28")
        self.label_9 = QLabel(self.tab_2)
        self.label_9.setObjectName(u"label_9")
        self.label_9.setMinimumSize(QSize(75, 0))
        self.label_9.setMaximumSize(QSize(75, 16777215))
        self.label_9.setFont(font)
        self.label_9.setStyleSheet(u"border:none")
        self.label_9.setLineWidth(0)

        self.horizontalLayout_28.addWidget(self.label_9)

        self.lineEdit_SegFilename = QLineEdit(self.tab_2)
        self.lineEdit_SegFilename.setObjectName(u"lineEdit_SegFilename")
        self.lineEdit_SegFilename.setMinimumSize(QSize(21, 0))
        self.lineEdit_SegFilename.setFont(font)

        self.horizontalLayout_28.addWidget(self.lineEdit_SegFilename)

        self.pushButton_openSeg = QPushButton(self.tab_2)
        self.pushButton_openSeg.setObjectName(u"pushButton_openSeg")
        self.pushButton_openSeg.setMinimumSize(QSize(0, 21))
        self.pushButton_openSeg.setStyleSheet(u"border:none")
        self.pushButton_openSeg.setIcon(icon1)

        self.horizontalLayout_28.addWidget(self.pushButton_openSeg)


        self.verticalLayout_4.addLayout(self.horizontalLayout_28)

        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.pushButton_load_settings = QPushButton(self.tab_2)
        self.pushButton_load_settings.setObjectName(u"pushButton_load_settings")

        self.horizontalLayout.addWidget(self.pushButton_load_settings)

        self.pushButton_loadData = QPushButton(self.tab_2)
        self.pushButton_loadData.setObjectName(u"pushButton_loadData")
        self.pushButton_loadData.setAcceptDrops(True)

        self.horizontalLayout.addWidget(self.pushButton_loadData)

        self.horizontalLayout.setStretch(1, 1)

        self.verticalLayout_4.addLayout(self.horizontalLayout)

        self.tabWidget_dataSource.addTab(self.tab_2, "")
        self.tab_3 = QWidget()
        self.tab_3.setObjectName(u"tab_3")
        self.verticalLayout_8 = QVBoxLayout(self.tab_3)
        self.verticalLayout_8.setSpacing(3)
        self.verticalLayout_8.setObjectName(u"verticalLayout_8")
        self.verticalLayout_8.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_34 = QHBoxLayout()
        self.horizontalLayout_34.setSpacing(1)
        self.horizontalLayout_34.setObjectName(u"horizontalLayout_34")
        self.label_11 = QLabel(self.tab_3)
        self.label_11.setObjectName(u"label_11")
        self.label_11.setMinimumSize(QSize(57, 0))
        self.label_11.setMaximumSize(QSize(57, 16777215))
        self.label_11.setFont(font)
        self.label_11.setStyleSheet(u"border:none")
        self.label_11.setLineWidth(0)

        self.horizontalLayout_34.addWidget(self.label_11)

        self.lineEdit_transfer_oct_name = QLineEdit(self.tab_3)
        self.lineEdit_transfer_oct_name.setObjectName(u"lineEdit_transfer_oct_name")
        self.lineEdit_transfer_oct_name.setMinimumSize(QSize(0, 24))

        self.horizontalLayout_34.addWidget(self.lineEdit_transfer_oct_name)

        self.pushButton_transfer_oct = QPushButton(self.tab_3)
        self.pushButton_transfer_oct.setObjectName(u"pushButton_transfer_oct")
        self.pushButton_transfer_oct.setMaximumSize(QSize(48, 24))
        self.pushButton_transfer_oct.setIconSize(QSize(22, 22))

        self.horizontalLayout_34.addWidget(self.pushButton_transfer_oct)


        self.verticalLayout_8.addLayout(self.horizontalLayout_34)

        self.horizontalLayout_35 = QHBoxLayout()
        self.horizontalLayout_35.setSpacing(1)
        self.horizontalLayout_35.setObjectName(u"horizontalLayout_35")
        self.label_12 = QLabel(self.tab_3)
        self.label_12.setObjectName(u"label_12")
        self.label_12.setMinimumSize(QSize(57, 0))
        self.label_12.setMaximumSize(QSize(57, 16777215))
        self.label_12.setFont(font)
        self.label_12.setStyleSheet(u"border:none")
        self.label_12.setLineWidth(0)

        self.horizontalLayout_35.addWidget(self.label_12)

        self.lineEdit_transfer_octa_name = QLineEdit(self.tab_3)
        self.lineEdit_transfer_octa_name.setObjectName(u"lineEdit_transfer_octa_name")
        self.lineEdit_transfer_octa_name.setMinimumSize(QSize(0, 24))

        self.horizontalLayout_35.addWidget(self.lineEdit_transfer_octa_name)

        self.pushButton_transfer_octa = QPushButton(self.tab_3)
        self.pushButton_transfer_octa.setObjectName(u"pushButton_transfer_octa")
        self.pushButton_transfer_octa.setMaximumSize(QSize(48, 24))

        self.horizontalLayout_35.addWidget(self.pushButton_transfer_octa)


        self.verticalLayout_8.addLayout(self.horizontalLayout_35)

        self.horizontalLayout_37 = QHBoxLayout()
        self.horizontalLayout_37.setSpacing(1)
        self.horizontalLayout_37.setObjectName(u"horizontalLayout_37")
        self.label_13 = QLabel(self.tab_3)
        self.label_13.setObjectName(u"label_13")
        self.label_13.setMinimumSize(QSize(57, 0))
        self.label_13.setMaximumSize(QSize(57, 16777215))
        self.label_13.setFont(font)
        self.label_13.setStyleSheet(u"border:none")
        self.label_13.setLineWidth(0)

        self.horizontalLayout_37.addWidget(self.label_13)

        self.lineEdit_transfer_seg_name = QLineEdit(self.tab_3)
        self.lineEdit_transfer_seg_name.setObjectName(u"lineEdit_transfer_seg_name")
        self.lineEdit_transfer_seg_name.setMinimumSize(QSize(0, 24))

        self.horizontalLayout_37.addWidget(self.lineEdit_transfer_seg_name)

        self.pushButton_transfer_seg = QPushButton(self.tab_3)
        self.pushButton_transfer_seg.setObjectName(u"pushButton_transfer_seg")
        self.pushButton_transfer_seg.setMaximumSize(QSize(48, 24))

        self.horizontalLayout_37.addWidget(self.pushButton_transfer_seg)


        self.verticalLayout_8.addLayout(self.horizontalLayout_37)

        self.tabWidget_dataSource.addTab(self.tab_3, "")

        self.horizontalLayout_16.addWidget(self.tabWidget_dataSource)


        self.verticalLayout_3.addWidget(self.groupBox_4)

        self.groupBox_frame = QGroupBox(self.widget_side_panel)
        self.groupBox_frame.setObjectName(u"groupBox_frame")
        self.verticalLayout_2 = QVBoxLayout(self.groupBox_frame)
        self.verticalLayout_2.setSpacing(3)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_24 = QHBoxLayout()
        self.horizontalLayout_24.setSpacing(5)
        self.horizontalLayout_24.setObjectName(u"horizontalLayout_24")
        self.label = QLabel(self.groupBox_frame)
        self.label.setObjectName(u"label")
        self.label.setFont(font)
        self.label.setStyleSheet(u"border:none")
        self.label.setLineWidth(0)

        self.horizontalLayout_24.addWidget(self.label)

        self.spinBox_frameIdx = QSpinBox(self.groupBox_frame)
        self.spinBox_frameIdx.setObjectName(u"spinBox_frameIdx")
        self.spinBox_frameIdx.setMinimumSize(QSize(68, 0))
        self.spinBox_frameIdx.setMaximumSize(QSize(68, 16777215))
        self.spinBox_frameIdx.setMinimum(1)
        self.spinBox_frameIdx.setMaximum(1)

        self.horizontalLayout_24.addWidget(self.spinBox_frameIdx)

        self.label_2 = QLabel(self.groupBox_frame)
        self.label_2.setObjectName(u"label_2")
        self.label_2.setFont(font)
        self.label_2.setStyleSheet(u"border:none")

        self.horizontalLayout_24.addWidget(self.label_2)

        self.lineEdit_totalFrame = QLineEdit(self.groupBox_frame)
        self.lineEdit_totalFrame.setObjectName(u"lineEdit_totalFrame")
        self.lineEdit_totalFrame.setMinimumSize(QSize(0, 0))
        self.lineEdit_totalFrame.setFont(font)

        self.horizontalLayout_24.addWidget(self.lineEdit_totalFrame)


        self.verticalLayout_2.addLayout(self.horizontalLayout_24)

        self.pushButton_update_resolution = QPushButton(self.groupBox_frame)
        self.pushButton_update_resolution.setObjectName(u"pushButton_update_resolution")

        self.verticalLayout_2.addWidget(self.pushButton_update_resolution)


        self.verticalLayout_3.addWidget(self.groupBox_frame)

        self.groupBox_5 = QGroupBox(self.widget_side_panel)
        self.groupBox_5.setObjectName(u"groupBox_5")
        self.groupBox_5.setMinimumSize(QSize(0, 140))
        self.groupBox_5.setMaximumSize(QSize(16777215, 140))
        self.verticalLayout_12 = QVBoxLayout(self.groupBox_5)
        self.verticalLayout_12.setSpacing(3)
        self.verticalLayout_12.setObjectName(u"verticalLayout_12")
        self.verticalLayout_12.setContentsMargins(0, 15, 0, 0)
        self.horizontalLayout_50 = QHBoxLayout()
        self.horizontalLayout_50.setSpacing(3)
        self.horizontalLayout_50.setObjectName(u"horizontalLayout_50")
        self.label_27 = QLabel(self.groupBox_5)
        self.label_27.setObjectName(u"label_27")
        self.label_27.setMaximumSize(QSize(60, 16777215))

        self.horizontalLayout_50.addWidget(self.label_27)

        self.comboBox_flatten = QComboBox(self.groupBox_5)
        self.comboBox_flatten.addItem("")
        self.comboBox_flatten.addItem("")
        self.comboBox_flatten.addItem("")
        self.comboBox_flatten.setObjectName(u"comboBox_flatten")

        self.horizontalLayout_50.addWidget(self.comboBox_flatten)


        self.verticalLayout_12.addLayout(self.horizontalLayout_50)

        self.horizontalLayout_29 = QHBoxLayout()
        self.horizontalLayout_29.setSpacing(3)
        self.horizontalLayout_29.setObjectName(u"horizontalLayout_29")
        self.label_26 = QLabel(self.groupBox_5)
        self.label_26.setObjectName(u"label_26")
        self.label_26.setMaximumSize(QSize(60, 16777215))

        self.horizontalLayout_29.addWidget(self.label_26)

        self.comboBox_permute = QComboBox(self.groupBox_5)
        self.comboBox_permute.addItem("")
        self.comboBox_permute.addItem("")
        self.comboBox_permute.addItem("")
        self.comboBox_permute.addItem("")
        self.comboBox_permute.addItem("")
        self.comboBox_permute.addItem("")
        self.comboBox_permute.setObjectName(u"comboBox_permute")
        self.comboBox_permute.setEnabled(True)
        self.comboBox_permute.setMinimumSize(QSize(0, 0))
        self.comboBox_permute.setMaximumSize(QSize(16777215, 21))

        self.horizontalLayout_29.addWidget(self.comboBox_permute)


        self.verticalLayout_12.addLayout(self.horizontalLayout_29)

        self.horizontalLayout_30 = QHBoxLayout()
        self.horizontalLayout_30.setSpacing(3)
        self.horizontalLayout_30.setObjectName(u"horizontalLayout_30")
        self.label_25 = QLabel(self.groupBox_5)
        self.label_25.setObjectName(u"label_25")
        self.label_25.setMaximumSize(QSize(60, 16777215))

        self.horizontalLayout_30.addWidget(self.label_25)

        self.comboBox_flip = QComboBox(self.groupBox_5)
        self.comboBox_flip.addItem("")
        self.comboBox_flip.addItem("")
        self.comboBox_flip.addItem("")
        self.comboBox_flip.setObjectName(u"comboBox_flip")
        self.comboBox_flip.setEnabled(True)
        self.comboBox_flip.setMinimumSize(QSize(0, 0))
        self.comboBox_flip.setMaximumSize(QSize(16777215, 21))

        self.horizontalLayout_30.addWidget(self.comboBox_flip)


        self.verticalLayout_12.addLayout(self.horizontalLayout_30)

        self.horizontalLayout_51 = QHBoxLayout()
        self.horizontalLayout_51.setSpacing(2)
        self.horizontalLayout_51.setObjectName(u"horizontalLayout_51")
        self.label_28 = QLabel(self.groupBox_5)
        self.label_28.setObjectName(u"label_28")
        self.label_28.setMaximumSize(QSize(30, 16777215))

        self.horizontalLayout_51.addWidget(self.label_28)

        self.spinBox_roi_top = QSpinBox(self.groupBox_5)
        self.spinBox_roi_top.setObjectName(u"spinBox_roi_top")
        self.spinBox_roi_top.setMinimumSize(QSize(0, 0))
        self.spinBox_roi_top.setMaximumSize(QSize(16777215, 24))

        self.horizontalLayout_51.addWidget(self.spinBox_roi_top)

        self.label_29 = QLabel(self.groupBox_5)
        self.label_29.setObjectName(u"label_29")
        self.label_29.setMaximumSize(QSize(3, 16777215))

        self.horizontalLayout_51.addWidget(self.label_29)

        self.spinBox_roi_bot = QSpinBox(self.groupBox_5)
        self.spinBox_roi_bot.setObjectName(u"spinBox_roi_bot")
        self.spinBox_roi_bot.setMinimumSize(QSize(0, 0))
        self.spinBox_roi_bot.setMaximumSize(QSize(16777215, 24))

        self.horizontalLayout_51.addWidget(self.spinBox_roi_bot)

        self.pushButton_auto_roi = QPushButton(self.groupBox_5)
        self.pushButton_auto_roi.setObjectName(u"pushButton_auto_roi")
        self.pushButton_auto_roi.setMinimumSize(QSize(0, 0))
        self.pushButton_auto_roi.setMaximumSize(QSize(50, 21))

        self.horizontalLayout_51.addWidget(self.pushButton_auto_roi)


        self.verticalLayout_12.addLayout(self.horizontalLayout_51)


        self.verticalLayout_3.addWidget(self.groupBox_5)

        self.scrollArea = QScrollArea(self.widget_side_panel)
        self.scrollArea.setObjectName(u"scrollArea")
        self.scrollArea.setStyleSheet(u"QScrollBar {width:2px;}")
        self.scrollArea.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.scrollArea.setWidgetResizable(True)
        self.scrollAreaWidgetContents = QWidget()
        self.scrollAreaWidgetContents.setObjectName(u"scrollAreaWidgetContents")
        self.scrollAreaWidgetContents.setGeometry(QRect(0, 0, 249, 440))
        self.verticalLayout = QVBoxLayout(self.scrollAreaWidgetContents)
        self.verticalLayout.setSpacing(0)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.stackedWidget = QStackedWidget(self.scrollAreaWidgetContents)
        self.stackedWidget.setObjectName(u"stackedWidget")
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.MinimumExpanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.stackedWidget.sizePolicy().hasHeightForWidth())
        self.stackedWidget.setSizePolicy(sizePolicy)
        self.stackedWidget.setMinimumSize(QSize(0, 0))
        self.stackedWidget.setMaximumSize(QSize(248, 16777215))

        self.verticalLayout.addWidget(self.stackedWidget)

        self.scrollArea.setWidget(self.scrollAreaWidgetContents)

        self.verticalLayout_3.addWidget(self.scrollArea)


        self.horizontalLayout_15.addWidget(self.widget_side_panel)

        RetinalLayerSegmentaion.setCentralWidget(self.centralwidget)
        self.statusbar = QStatusBar(RetinalLayerSegmentaion)
        self.statusbar.setObjectName(u"statusbar")
        RetinalLayerSegmentaion.setStatusBar(self.statusbar)
        self.toolBar = QToolBar(RetinalLayerSegmentaion)
        self.toolBar.setObjectName(u"toolBar")
        self.toolBar.setMinimumSize(QSize(0, 0))
        self.toolBar.setFont(font)
        self.toolBar.setMovable(False)
        self.toolBar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        self.toolBar.setFloatable(False)
        RetinalLayerSegmentaion.addToolBar(Qt.ToolBarArea.TopToolBarArea, self.toolBar)

        self.toolBar.addAction(self.actionSave)
        self.toolBar.addSeparator()
        self.toolBar.addAction(self.actionCleanUp)

        self.retranslateUi(RetinalLayerSegmentaion)

        self.tabWidget_dataSource.setCurrentIndex(1)
        self.stackedWidget.setCurrentIndex(-1)


        QMetaObject.connectSlotsByName(RetinalLayerSegmentaion)
    # setupUi

    def retranslateUi(self, RetinalLayerSegmentaion):
        RetinalLayerSegmentaion.setWindowTitle(QCoreApplication.translate("RetinalLayerSegmentaion", u"OCT Analyzer", None))
        self.actionSave.setText(QCoreApplication.translate("RetinalLayerSegmentaion", u"Save", None))
#if QT_CONFIG(tooltip)
        self.actionSave.setToolTip(QCoreApplication.translate("RetinalLayerSegmentaion", u"Save segmentation", None))
#endif // QT_CONFIG(tooltip)
        self.actionCleanUp.setText(QCoreApplication.translate("RetinalLayerSegmentaion", u"Clean", None))
#if QT_CONFIG(tooltip)
        self.actionCleanUp.setToolTip(QCoreApplication.translate("RetinalLayerSegmentaion", u"Clean loaded data", None))
#endif // QT_CONFIG(tooltip)
        self.groupBox_4.setTitle(QCoreApplication.translate("RetinalLayerSegmentaion", u"Data Source", None))
        self.label_5.setText(QCoreApplication.translate("RetinalLayerSegmentaion", u"OCT:", None))
        self.comboBox_OCT.setItemText(0, QCoreApplication.translate("RetinalLayerSegmentaion", u"-", None))

        self.pushButton_refreshDatalist.setText("")
        self.label_6.setText(QCoreApplication.translate("RetinalLayerSegmentaion", u"OCTA:", None))
        self.comboBox_OCTA.setItemText(0, QCoreApplication.translate("RetinalLayerSegmentaion", u"-", None))

        self.label_7.setText(QCoreApplication.translate("RetinalLayerSegmentaion", u"Segment:", None))
        self.comboBox_Seg.setItemText(0, QCoreApplication.translate("RetinalLayerSegmentaion", u"-", None))

        self.tabWidget_dataSource.setTabText(self.tabWidget_dataSource.indexOf(self.tab), QCoreApplication.translate("RetinalLayerSegmentaion", u"Workspace", None))
        self.label_8.setText(QCoreApplication.translate("RetinalLayerSegmentaion", u"OCT: ", None))
#if QT_CONFIG(tooltip)
        self.pushButton_openOCT.setToolTip(QCoreApplication.translate("RetinalLayerSegmentaion", u"Load OCT Data", None))
#endif // QT_CONFIG(tooltip)
        self.pushButton_openOCT.setText("")
#if QT_CONFIG(tooltip)
        self.checkBox_OCTA.setToolTip(QCoreApplication.translate("RetinalLayerSegmentaion", u"Uncheck will not load OCTA data", None))
#endif // QT_CONFIG(tooltip)
        self.checkBox_OCTA.setText("")
        self.label_10.setText(QCoreApplication.translate("RetinalLayerSegmentaion", u"OCTA: ", None))
#if QT_CONFIG(tooltip)
        self.pushButton_openOCTA.setToolTip(QCoreApplication.translate("RetinalLayerSegmentaion", u"Load OCTA Data", None))
#endif // QT_CONFIG(tooltip)
        self.pushButton_openOCTA.setText("")
        self.label_9.setText(QCoreApplication.translate("RetinalLayerSegmentaion", u"Segment: ", None))
#if QT_CONFIG(tooltip)
        self.pushButton_openSeg.setToolTip(QCoreApplication.translate("RetinalLayerSegmentaion", u"Load Segmentatio Data", None))
#endif // QT_CONFIG(tooltip)
        self.pushButton_openSeg.setText("")
        self.pushButton_load_settings.setText(QCoreApplication.translate("RetinalLayerSegmentaion", u"setting...", None))
        self.pushButton_loadData.setText(QCoreApplication.translate("RetinalLayerSegmentaion", u"Load Data", None))
        self.tabWidget_dataSource.setTabText(self.tabWidget_dataSource.indexOf(self.tab_2), QCoreApplication.translate("RetinalLayerSegmentaion", u"File", None))
        self.label_11.setText(QCoreApplication.translate("RetinalLayerSegmentaion", u"OCT:", None))
        self.lineEdit_transfer_oct_name.setText(QCoreApplication.translate("RetinalLayerSegmentaion", u"oct_data", None))
        self.pushButton_transfer_oct.setText(QCoreApplication.translate("RetinalLayerSegmentaion", u">>", None))
        self.label_12.setText(QCoreApplication.translate("RetinalLayerSegmentaion", u"OCTA:", None))
        self.lineEdit_transfer_octa_name.setText(QCoreApplication.translate("RetinalLayerSegmentaion", u"octa_data", None))
        self.pushButton_transfer_octa.setText(QCoreApplication.translate("RetinalLayerSegmentaion", u">>", None))
        self.label_13.setText(QCoreApplication.translate("RetinalLayerSegmentaion", u"Segment:", None))
        self.lineEdit_transfer_seg_name.setText(QCoreApplication.translate("RetinalLayerSegmentaion", u"seg_data", None))
        self.pushButton_transfer_seg.setText(QCoreApplication.translate("RetinalLayerSegmentaion", u">>", None))
        self.tabWidget_dataSource.setTabText(self.tabWidget_dataSource.indexOf(self.tab_3), QCoreApplication.translate("RetinalLayerSegmentaion", u"Transfer", None))
        self.groupBox_frame.setTitle("")
        self.label.setText(QCoreApplication.translate("RetinalLayerSegmentaion", u"Frame:", None))
        self.label_2.setText(QCoreApplication.translate("RetinalLayerSegmentaion", u"/", None))
        self.lineEdit_totalFrame.setText(QCoreApplication.translate("RetinalLayerSegmentaion", u"1", None))
        self.pushButton_update_resolution.setText(QCoreApplication.translate("RetinalLayerSegmentaion", u"Upate data resolution", None))
        self.groupBox_5.setTitle(QCoreApplication.translate("RetinalLayerSegmentaion", u"Data Transform", None))
        self.label_27.setText(QCoreApplication.translate("RetinalLayerSegmentaion", u"Flatten", None))
        self.comboBox_flatten.setItemText(0, QCoreApplication.translate("RetinalLayerSegmentaion", u"None", None))
        self.comboBox_flatten.setItemText(1, QCoreApplication.translate("RetinalLayerSegmentaion", u"Fitting", None))
        self.comboBox_flatten.setItemText(2, QCoreApplication.translate("RetinalLayerSegmentaion", u"RPE-BM", None))

        self.label_26.setText(QCoreApplication.translate("RetinalLayerSegmentaion", u"Permute", None))
        self.comboBox_permute.setItemText(0, QCoreApplication.translate("RetinalLayerSegmentaion", u"0,1,2", None))
        self.comboBox_permute.setItemText(1, QCoreApplication.translate("RetinalLayerSegmentaion", u"0,2,1", None))
        self.comboBox_permute.setItemText(2, QCoreApplication.translate("RetinalLayerSegmentaion", u"1,0,2", None))
        self.comboBox_permute.setItemText(3, QCoreApplication.translate("RetinalLayerSegmentaion", u"1,2,0", None))
        self.comboBox_permute.setItemText(4, QCoreApplication.translate("RetinalLayerSegmentaion", u"2,0,1", None))
        self.comboBox_permute.setItemText(5, QCoreApplication.translate("RetinalLayerSegmentaion", u"2,1,0", None))

        self.label_25.setText(QCoreApplication.translate("RetinalLayerSegmentaion", u"Flip Axis", None))
        self.comboBox_flip.setItemText(0, QCoreApplication.translate("RetinalLayerSegmentaion", u"None", None))
        self.comboBox_flip.setItemText(1, QCoreApplication.translate("RetinalLayerSegmentaion", u"Left-Right", None))
        self.comboBox_flip.setItemText(2, QCoreApplication.translate("RetinalLayerSegmentaion", u"Up-Down", None))

        self.label_28.setText(QCoreApplication.translate("RetinalLayerSegmentaion", u"ROI:", None))
        self.label_29.setText(QCoreApplication.translate("RetinalLayerSegmentaion", u":", None))
        self.pushButton_auto_roi.setText(QCoreApplication.translate("RetinalLayerSegmentaion", u"Auto", None))
        self.toolBar.setWindowTitle(QCoreApplication.translate("RetinalLayerSegmentaion", u"toolBar", None))
    # retranslateUi

