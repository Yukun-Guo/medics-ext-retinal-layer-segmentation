# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'LayerSegSideCtrlPanelUI.ui'
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
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QGroupBox,
    QHBoxLayout, QLabel, QPushButton, QSizePolicy,
    QSpacerItem, QSpinBox, QVBoxLayout, QWidget)

class Ui_LayerSegSideControlPanel(object):
    def setupUi(self, LayerSegSideControlPanel):
        if not LayerSegSideControlPanel.objectName():
            LayerSegSideControlPanel.setObjectName(u"LayerSegSideControlPanel")
        LayerSegSideControlPanel.resize(228, 610)
        self.verticalLayout = QVBoxLayout(LayerSegSideControlPanel)
        self.verticalLayout.setSpacing(0)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.groupBox_3 = QGroupBox(LayerSegSideControlPanel)
        self.groupBox_3.setObjectName(u"groupBox_3")
        self.groupBox_3.setMinimumSize(QSize(0, 0))
        self.groupBox_3.setMaximumSize(QSize(16777215, 16777215))
        font = QFont()
        font.setPointSize(10)
        self.groupBox_3.setFont(font)
        self.groupBox_3.setStyleSheet(u"")
        self.verticalLayout_7 = QVBoxLayout(self.groupBox_3)
        self.verticalLayout_7.setSpacing(3)
        self.verticalLayout_7.setObjectName(u"verticalLayout_7")
        self.verticalLayout_7.setContentsMargins(0, 0, 0, 0)
        self.pushButton_ai_seg = QPushButton(self.groupBox_3)
        self.pushButton_ai_seg.setObjectName(u"pushButton_ai_seg")
        self.pushButton_ai_seg.setMinimumSize(QSize(0, 0))
        self.pushButton_ai_seg.setMaximumSize(QSize(16777215, 24))

        self.verticalLayout_7.addWidget(self.pushButton_ai_seg)

        self.checkBox_sparse = QCheckBox(self.groupBox_3)
        self.checkBox_sparse.setObjectName(u"checkBox_sparse")

        self.verticalLayout_7.addWidget(self.checkBox_sparse)

        self.horizontalLayout_46 = QHBoxLayout()
        self.horizontalLayout_46.setObjectName(u"horizontalLayout_46")
        self.label_30 = QLabel(self.groupBox_3)
        self.label_30.setObjectName(u"label_30")
        self.label_30.setMaximumSize(QSize(16777215, 16777215))
        self.label_30.setAlignment(Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignTrailing|Qt.AlignmentFlag.AlignVCenter)

        self.horizontalLayout_46.addWidget(self.label_30)

        self.comboBox_gpu = QComboBox(self.groupBox_3)
        self.comboBox_gpu.setObjectName(u"comboBox_gpu")
        self.comboBox_gpu.setMinimumSize(QSize(140, 0))
        self.comboBox_gpu.setMaximumSize(QSize(120, 16777215))

        self.horizontalLayout_46.addWidget(self.comboBox_gpu)


        self.verticalLayout_7.addLayout(self.horizontalLayout_46)

        self.horizontalLayout_26 = QHBoxLayout()
        self.horizontalLayout_26.setSpacing(3)
        self.horizontalLayout_26.setObjectName(u"horizontalLayout_26")
        self.label_4 = QLabel(self.groupBox_3)
        self.label_4.setObjectName(u"label_4")
        font1 = QFont()
        font1.setPointSize(9)
        self.label_4.setFont(font1)
        self.label_4.setStyleSheet(u"border:none")
        self.label_4.setLineWidth(0)
        self.label_4.setAlignment(Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignTrailing|Qt.AlignmentFlag.AlignVCenter)

        self.horizontalLayout_26.addWidget(self.label_4)

        self.comboBox_corrector = QComboBox(self.groupBox_3)
        self.comboBox_corrector.addItem("")
        self.comboBox_corrector.addItem("")
        self.comboBox_corrector.addItem("")
        self.comboBox_corrector.addItem("")
        self.comboBox_corrector.setObjectName(u"comboBox_corrector")
        self.comboBox_corrector.setMinimumSize(QSize(140, 0))
        self.comboBox_corrector.setMaximumSize(QSize(120, 24))
        self.comboBox_corrector.setFont(font1)

        self.horizontalLayout_26.addWidget(self.comboBox_corrector)


        self.verticalLayout_7.addLayout(self.horizontalLayout_26)

        self.horizontalLayout_25 = QHBoxLayout()
        self.horizontalLayout_25.setObjectName(u"horizontalLayout_25")
        self.label_3 = QLabel(self.groupBox_3)
        self.label_3.setObjectName(u"label_3")
        self.label_3.setMaximumSize(QSize(16777215, 16777215))
        self.label_3.setFont(font1)
        self.label_3.setAlignment(Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignTrailing|Qt.AlignmentFlag.AlignVCenter)

        self.horizontalLayout_25.addWidget(self.label_3)

        self.spinBox_Interp_step = QSpinBox(self.groupBox_3)
        self.spinBox_Interp_step.setObjectName(u"spinBox_Interp_step")
        self.spinBox_Interp_step.setMinimumSize(QSize(0, 0))
        self.spinBox_Interp_step.setMinimum(2)
        self.spinBox_Interp_step.setValue(10)

        self.horizontalLayout_25.addWidget(self.spinBox_Interp_step)


        self.verticalLayout_7.addLayout(self.horizontalLayout_25)

        self.pushButton_Switch_Interp = QPushButton(self.groupBox_3)
        self.pushButton_Switch_Interp.setObjectName(u"pushButton_Switch_Interp")
        self.pushButton_Switch_Interp.setMinimumSize(QSize(0, 0))
        self.pushButton_Switch_Interp.setMaximumSize(QSize(16777215, 24))
        self.pushButton_Switch_Interp.setIconSize(QSize(55, 24))

        self.verticalLayout_7.addWidget(self.pushButton_Switch_Interp)

        self.pushButton_Restart_Interp = QPushButton(self.groupBox_3)
        self.pushButton_Restart_Interp.setObjectName(u"pushButton_Restart_Interp")
        self.pushButton_Restart_Interp.setMinimumSize(QSize(0, 0))
        self.pushButton_Restart_Interp.setMaximumSize(QSize(16777215, 24))

        self.verticalLayout_7.addWidget(self.pushButton_Restart_Interp)

        self.checkBox_keep_fluid = QCheckBox(self.groupBox_3)
        self.checkBox_keep_fluid.setObjectName(u"checkBox_keep_fluid")
        self.checkBox_keep_fluid.setChecked(False)

        self.verticalLayout_7.addWidget(self.checkBox_keep_fluid)


        self.verticalLayout.addWidget(self.groupBox_3)

        self.groupBox = QGroupBox(LayerSegSideControlPanel)
        self.groupBox.setObjectName(u"groupBox")
        self.groupBox.setMinimumSize(QSize(0, 0))
        self.groupBox.setMaximumSize(QSize(16777215, 350))
        self.groupBox.setFont(font)
        self.groupBox.setStyleSheet(u"")
        self.horizontalLayout_22 = QHBoxLayout(self.groupBox)
        self.horizontalLayout_22.setSpacing(3)
        self.horizontalLayout_22.setObjectName(u"horizontalLayout_22")
        self.horizontalLayout_22.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_5 = QVBoxLayout()
        self.verticalLayout_5.setSpacing(0)
        self.verticalLayout_5.setObjectName(u"verticalLayout_5")
        self.verticalLayout_5.setContentsMargins(-1, -1, -1, 0)
        self.horizontalLayout_40 = QHBoxLayout()
        self.horizontalLayout_40.setSpacing(1)
        self.horizontalLayout_40.setObjectName(u"horizontalLayout_40")
        self.checkBox_pvd = QCheckBox(self.groupBox)
        self.checkBox_pvd.setObjectName(u"checkBox_pvd")
        self.checkBox_pvd.setMinimumSize(QSize(115, 0))
        self.checkBox_pvd.setMaximumSize(QSize(115, 16777215))
        self.checkBox_pvd.setFont(font)
        self.checkBox_pvd.setStyleSheet(u"")
        self.checkBox_pvd.setChecked(False)

        self.horizontalLayout_40.addWidget(self.checkBox_pvd)

        self.comboBox_pvd_line = QComboBox(self.groupBox)
        self.comboBox_pvd_line.addItem("")
        self.comboBox_pvd_line.addItem("")
        self.comboBox_pvd_line.setObjectName(u"comboBox_pvd_line")
        self.comboBox_pvd_line.setMinimumSize(QSize(24, 0))
        self.comboBox_pvd_line.setMaximumSize(QSize(45, 16777215))
        self.comboBox_pvd_line.setFont(font)

        self.horizontalLayout_40.addWidget(self.comboBox_pvd_line)

        self.horizontalLayout_40.setStretch(1, 1)

        self.verticalLayout_5.addLayout(self.horizontalLayout_40)

        self.horizontalLayout_2 = QHBoxLayout()
        self.horizontalLayout_2.setSpacing(1)
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.checkBox_ilm = QCheckBox(self.groupBox)
        self.checkBox_ilm.setObjectName(u"checkBox_ilm")
        self.checkBox_ilm.setMinimumSize(QSize(115, 0))
        self.checkBox_ilm.setMaximumSize(QSize(115, 16777215))
        self.checkBox_ilm.setFont(font)
        self.checkBox_ilm.setStyleSheet(u"")
        self.checkBox_ilm.setChecked(True)

        self.horizontalLayout_2.addWidget(self.checkBox_ilm)

        self.comboBox_ilm_line = QComboBox(self.groupBox)
        self.comboBox_ilm_line.addItem("")
        self.comboBox_ilm_line.addItem("")
        self.comboBox_ilm_line.setObjectName(u"comboBox_ilm_line")
        self.comboBox_ilm_line.setMinimumSize(QSize(24, 0))
        self.comboBox_ilm_line.setMaximumSize(QSize(45, 16777215))
        self.comboBox_ilm_line.setFont(font)

        self.horizontalLayout_2.addWidget(self.comboBox_ilm_line)

        self.horizontalLayout_2.setStretch(1, 1)

        self.verticalLayout_5.addLayout(self.horizontalLayout_2)

        self.horizontalLayout_4 = QHBoxLayout()
        self.horizontalLayout_4.setSpacing(1)
        self.horizontalLayout_4.setObjectName(u"horizontalLayout_4")
        self.checkBox_nflgcl = QCheckBox(self.groupBox)
        self.checkBox_nflgcl.setObjectName(u"checkBox_nflgcl")
        self.checkBox_nflgcl.setMinimumSize(QSize(115, 0))
        self.checkBox_nflgcl.setMaximumSize(QSize(115, 16777215))
        self.checkBox_nflgcl.setFont(font)
        self.checkBox_nflgcl.setStyleSheet(u"")
        self.checkBox_nflgcl.setChecked(True)

        self.horizontalLayout_4.addWidget(self.checkBox_nflgcl)

        self.comboBox_nflgcl_line = QComboBox(self.groupBox)
        self.comboBox_nflgcl_line.addItem("")
        self.comboBox_nflgcl_line.addItem("")
        self.comboBox_nflgcl_line.setObjectName(u"comboBox_nflgcl_line")
        self.comboBox_nflgcl_line.setMinimumSize(QSize(24, 0))
        self.comboBox_nflgcl_line.setMaximumSize(QSize(45, 16777215))
        self.comboBox_nflgcl_line.setFont(font)

        self.horizontalLayout_4.addWidget(self.comboBox_nflgcl_line)

        self.horizontalLayout_4.setStretch(0, 2)
        self.horizontalLayout_4.setStretch(1, 1)

        self.verticalLayout_5.addLayout(self.horizontalLayout_4)

        self.horizontalLayout_5 = QHBoxLayout()
        self.horizontalLayout_5.setSpacing(1)
        self.horizontalLayout_5.setObjectName(u"horizontalLayout_5")
        self.checkBox_gclipl = QCheckBox(self.groupBox)
        self.checkBox_gclipl.setObjectName(u"checkBox_gclipl")
        self.checkBox_gclipl.setMinimumSize(QSize(115, 0))
        self.checkBox_gclipl.setMaximumSize(QSize(115, 16777215))
        self.checkBox_gclipl.setFont(font)
        self.checkBox_gclipl.setStyleSheet(u"")

        self.horizontalLayout_5.addWidget(self.checkBox_gclipl)

        self.comboBox_gclipl_line = QComboBox(self.groupBox)
        self.comboBox_gclipl_line.addItem("")
        self.comboBox_gclipl_line.addItem("")
        self.comboBox_gclipl_line.setObjectName(u"comboBox_gclipl_line")
        self.comboBox_gclipl_line.setMinimumSize(QSize(24, 0))
        self.comboBox_gclipl_line.setMaximumSize(QSize(45, 16777215))
        self.comboBox_gclipl_line.setFont(font)

        self.horizontalLayout_5.addWidget(self.comboBox_gclipl_line)

        self.horizontalLayout_5.setStretch(0, 2)
        self.horizontalLayout_5.setStretch(1, 1)

        self.verticalLayout_5.addLayout(self.horizontalLayout_5)

        self.horizontalLayout_6 = QHBoxLayout()
        self.horizontalLayout_6.setSpacing(1)
        self.horizontalLayout_6.setObjectName(u"horizontalLayout_6")
        self.checkBox_iplinl = QCheckBox(self.groupBox)
        self.checkBox_iplinl.setObjectName(u"checkBox_iplinl")
        self.checkBox_iplinl.setMinimumSize(QSize(115, 0))
        self.checkBox_iplinl.setMaximumSize(QSize(115, 16777215))
        self.checkBox_iplinl.setFont(font)
        self.checkBox_iplinl.setStyleSheet(u"")
        self.checkBox_iplinl.setChecked(True)

        self.horizontalLayout_6.addWidget(self.checkBox_iplinl)

        self.comboBox_iplinl_line = QComboBox(self.groupBox)
        self.comboBox_iplinl_line.addItem("")
        self.comboBox_iplinl_line.addItem("")
        self.comboBox_iplinl_line.setObjectName(u"comboBox_iplinl_line")
        self.comboBox_iplinl_line.setMinimumSize(QSize(24, 0))
        self.comboBox_iplinl_line.setMaximumSize(QSize(45, 16777215))
        self.comboBox_iplinl_line.setFont(font)

        self.horizontalLayout_6.addWidget(self.comboBox_iplinl_line)

        self.horizontalLayout_6.setStretch(0, 2)
        self.horizontalLayout_6.setStretch(1, 1)

        self.verticalLayout_5.addLayout(self.horizontalLayout_6)

        self.horizontalLayout_7 = QHBoxLayout()
        self.horizontalLayout_7.setSpacing(1)
        self.horizontalLayout_7.setObjectName(u"horizontalLayout_7")
        self.checkBox_inlopl = QCheckBox(self.groupBox)
        self.checkBox_inlopl.setObjectName(u"checkBox_inlopl")
        self.checkBox_inlopl.setMinimumSize(QSize(115, 0))
        self.checkBox_inlopl.setMaximumSize(QSize(115, 16777215))
        self.checkBox_inlopl.setFont(font)
        self.checkBox_inlopl.setStyleSheet(u"")
        self.checkBox_inlopl.setChecked(True)

        self.horizontalLayout_7.addWidget(self.checkBox_inlopl)

        self.comboBox_inlopl_line = QComboBox(self.groupBox)
        self.comboBox_inlopl_line.addItem("")
        self.comboBox_inlopl_line.addItem("")
        self.comboBox_inlopl_line.setObjectName(u"comboBox_inlopl_line")
        self.comboBox_inlopl_line.setMinimumSize(QSize(24, 0))
        self.comboBox_inlopl_line.setMaximumSize(QSize(45, 16777215))
        self.comboBox_inlopl_line.setFont(font)

        self.horizontalLayout_7.addWidget(self.comboBox_inlopl_line)

        self.horizontalLayout_7.setStretch(0, 2)
        self.horizontalLayout_7.setStretch(1, 1)

        self.verticalLayout_5.addLayout(self.horizontalLayout_7)

        self.horizontalLayout_8 = QHBoxLayout()
        self.horizontalLayout_8.setSpacing(1)
        self.horizontalLayout_8.setObjectName(u"horizontalLayout_8")
        self.checkBox_oplonl = QCheckBox(self.groupBox)
        self.checkBox_oplonl.setObjectName(u"checkBox_oplonl")
        self.checkBox_oplonl.setMinimumSize(QSize(115, 0))
        self.checkBox_oplonl.setMaximumSize(QSize(115, 16777215))
        self.checkBox_oplonl.setFont(font)
        self.checkBox_oplonl.setStyleSheet(u"")
        self.checkBox_oplonl.setChecked(True)

        self.horizontalLayout_8.addWidget(self.checkBox_oplonl)

        self.comboBox_oplonl_line = QComboBox(self.groupBox)
        self.comboBox_oplonl_line.addItem("")
        self.comboBox_oplonl_line.addItem("")
        self.comboBox_oplonl_line.setObjectName(u"comboBox_oplonl_line")
        self.comboBox_oplonl_line.setMinimumSize(QSize(24, 0))
        self.comboBox_oplonl_line.setMaximumSize(QSize(45, 16777215))
        self.comboBox_oplonl_line.setFont(font)

        self.horizontalLayout_8.addWidget(self.comboBox_oplonl_line)

        self.horizontalLayout_8.setStretch(0, 2)
        self.horizontalLayout_8.setStretch(1, 1)

        self.verticalLayout_5.addLayout(self.horizontalLayout_8)

        self.horizontalLayout_9 = QHBoxLayout()
        self.horizontalLayout_9.setSpacing(1)
        self.horizontalLayout_9.setObjectName(u"horizontalLayout_9")
        self.checkBox_elm = QCheckBox(self.groupBox)
        self.checkBox_elm.setObjectName(u"checkBox_elm")
        self.checkBox_elm.setMinimumSize(QSize(115, 0))
        self.checkBox_elm.setMaximumSize(QSize(115, 16777215))
        self.checkBox_elm.setFont(font)
        self.checkBox_elm.setStyleSheet(u"")

        self.horizontalLayout_9.addWidget(self.checkBox_elm)

        self.comboBox_elm_line = QComboBox(self.groupBox)
        self.comboBox_elm_line.addItem("")
        self.comboBox_elm_line.addItem("")
        self.comboBox_elm_line.setObjectName(u"comboBox_elm_line")
        self.comboBox_elm_line.setMinimumSize(QSize(24, 0))
        self.comboBox_elm_line.setMaximumSize(QSize(45, 16777215))
        self.comboBox_elm_line.setFont(font)

        self.horizontalLayout_9.addWidget(self.comboBox_elm_line)

        self.horizontalLayout_9.setStretch(0, 2)
        self.horizontalLayout_9.setStretch(1, 1)

        self.verticalLayout_5.addLayout(self.horizontalLayout_9)

        self.horizontalLayout_10 = QHBoxLayout()
        self.horizontalLayout_10.setSpacing(1)
        self.horizontalLayout_10.setObjectName(u"horizontalLayout_10")
        self.checkBox_ez = QCheckBox(self.groupBox)
        self.checkBox_ez.setObjectName(u"checkBox_ez")
        self.checkBox_ez.setMinimumSize(QSize(115, 0))
        self.checkBox_ez.setMaximumSize(QSize(115, 16777215))
        self.checkBox_ez.setFont(font)
        self.checkBox_ez.setStyleSheet(u"")
        self.checkBox_ez.setChecked(True)

        self.horizontalLayout_10.addWidget(self.checkBox_ez)

        self.comboBox_ez_line = QComboBox(self.groupBox)
        self.comboBox_ez_line.addItem("")
        self.comboBox_ez_line.addItem("")
        self.comboBox_ez_line.setObjectName(u"comboBox_ez_line")
        self.comboBox_ez_line.setMinimumSize(QSize(24, 0))
        self.comboBox_ez_line.setMaximumSize(QSize(45, 16777215))
        self.comboBox_ez_line.setFont(font)

        self.horizontalLayout_10.addWidget(self.comboBox_ez_line)

        self.horizontalLayout_10.setStretch(0, 2)
        self.horizontalLayout_10.setStretch(1, 1)

        self.verticalLayout_5.addLayout(self.horizontalLayout_10)

        self.horizontalLayout_11 = QHBoxLayout()
        self.horizontalLayout_11.setSpacing(1)
        self.horizontalLayout_11.setObjectName(u"horizontalLayout_11")
        self.checkBox_eziz = QCheckBox(self.groupBox)
        self.checkBox_eziz.setObjectName(u"checkBox_eziz")
        self.checkBox_eziz.setMinimumSize(QSize(115, 0))
        self.checkBox_eziz.setMaximumSize(QSize(115, 16777215))
        self.checkBox_eziz.setFont(font)
        self.checkBox_eziz.setStyleSheet(u"")
        self.checkBox_eziz.setChecked(True)

        self.horizontalLayout_11.addWidget(self.checkBox_eziz)

        self.comboBox_eziz_line = QComboBox(self.groupBox)
        self.comboBox_eziz_line.addItem("")
        self.comboBox_eziz_line.addItem("")
        self.comboBox_eziz_line.setObjectName(u"comboBox_eziz_line")
        self.comboBox_eziz_line.setMinimumSize(QSize(24, 0))
        self.comboBox_eziz_line.setMaximumSize(QSize(45, 16777215))
        self.comboBox_eziz_line.setFont(font)

        self.horizontalLayout_11.addWidget(self.comboBox_eziz_line)

        self.horizontalLayout_11.setStretch(0, 2)
        self.horizontalLayout_11.setStretch(1, 1)

        self.verticalLayout_5.addLayout(self.horizontalLayout_11)

        self.horizontalLayout_12 = QHBoxLayout()
        self.horizontalLayout_12.setSpacing(1)
        self.horizontalLayout_12.setObjectName(u"horizontalLayout_12")
        self.checkBox_izrpe = QCheckBox(self.groupBox)
        self.checkBox_izrpe.setObjectName(u"checkBox_izrpe")
        self.checkBox_izrpe.setMinimumSize(QSize(115, 0))
        self.checkBox_izrpe.setMaximumSize(QSize(115, 16777215))
        self.checkBox_izrpe.setFont(font)

        self.horizontalLayout_12.addWidget(self.checkBox_izrpe)

        self.comboBox_izrpe_line = QComboBox(self.groupBox)
        self.comboBox_izrpe_line.addItem("")
        self.comboBox_izrpe_line.addItem("")
        self.comboBox_izrpe_line.setObjectName(u"comboBox_izrpe_line")
        self.comboBox_izrpe_line.setMinimumSize(QSize(24, 0))
        self.comboBox_izrpe_line.setMaximumSize(QSize(45, 16777215))
        self.comboBox_izrpe_line.setFont(font)

        self.horizontalLayout_12.addWidget(self.comboBox_izrpe_line)

        self.horizontalLayout_12.setStretch(0, 2)
        self.horizontalLayout_12.setStretch(1, 1)

        self.verticalLayout_5.addLayout(self.horizontalLayout_12)

        self.horizontalLayout_13 = QHBoxLayout()
        self.horizontalLayout_13.setSpacing(1)
        self.horizontalLayout_13.setObjectName(u"horizontalLayout_13")
        self.checkBox_rpebm = QCheckBox(self.groupBox)
        self.checkBox_rpebm.setObjectName(u"checkBox_rpebm")
        self.checkBox_rpebm.setMinimumSize(QSize(115, 0))
        self.checkBox_rpebm.setMaximumSize(QSize(115, 16777215))
        self.checkBox_rpebm.setFont(font)
        self.checkBox_rpebm.setChecked(True)

        self.horizontalLayout_13.addWidget(self.checkBox_rpebm)

        self.comboBox_rpebm_line = QComboBox(self.groupBox)
        self.comboBox_rpebm_line.addItem("")
        self.comboBox_rpebm_line.addItem("")
        self.comboBox_rpebm_line.setObjectName(u"comboBox_rpebm_line")
        self.comboBox_rpebm_line.setMinimumSize(QSize(24, 0))
        self.comboBox_rpebm_line.setMaximumSize(QSize(45, 16777215))
        self.comboBox_rpebm_line.setFont(font)

        self.horizontalLayout_13.addWidget(self.comboBox_rpebm_line)

        self.horizontalLayout_13.setStretch(0, 2)
        self.horizontalLayout_13.setStretch(1, 1)

        self.verticalLayout_5.addLayout(self.horizontalLayout_13)

        self.horizontalLayout_14 = QHBoxLayout()
        self.horizontalLayout_14.setSpacing(1)
        self.horizontalLayout_14.setObjectName(u"horizontalLayout_14")
        self.checkBox_sathal = QCheckBox(self.groupBox)
        self.checkBox_sathal.setObjectName(u"checkBox_sathal")
        self.checkBox_sathal.setMinimumSize(QSize(115, 0))
        self.checkBox_sathal.setMaximumSize(QSize(115, 16777215))
        self.checkBox_sathal.setFont(font)
        self.checkBox_sathal.setChecked(False)

        self.horizontalLayout_14.addWidget(self.checkBox_sathal)

        self.comboBox_sathal_line = QComboBox(self.groupBox)
        self.comboBox_sathal_line.addItem("")
        self.comboBox_sathal_line.addItem("")
        self.comboBox_sathal_line.setObjectName(u"comboBox_sathal_line")
        self.comboBox_sathal_line.setMinimumSize(QSize(24, 0))
        self.comboBox_sathal_line.setMaximumSize(QSize(45, 16777215))
        self.comboBox_sathal_line.setFont(font)

        self.horizontalLayout_14.addWidget(self.comboBox_sathal_line)

        self.horizontalLayout_14.setStretch(0, 2)
        self.horizontalLayout_14.setStretch(1, 1)

        self.verticalLayout_5.addLayout(self.horizontalLayout_14)

        self.horizontalLayout_21 = QHBoxLayout()
        self.horizontalLayout_21.setSpacing(1)
        self.horizontalLayout_21.setObjectName(u"horizontalLayout_21")
        self.checkBox_choroid = QCheckBox(self.groupBox)
        self.checkBox_choroid.setObjectName(u"checkBox_choroid")
        self.checkBox_choroid.setMinimumSize(QSize(115, 0))
        self.checkBox_choroid.setMaximumSize(QSize(115, 16777215))
        self.checkBox_choroid.setFont(font)
        self.checkBox_choroid.setChecked(True)

        self.horizontalLayout_21.addWidget(self.checkBox_choroid)

        self.comboBox_choroid_line = QComboBox(self.groupBox)
        self.comboBox_choroid_line.addItem("")
        self.comboBox_choroid_line.addItem("")
        self.comboBox_choroid_line.setObjectName(u"comboBox_choroid_line")
        self.comboBox_choroid_line.setMinimumSize(QSize(24, 0))
        self.comboBox_choroid_line.setMaximumSize(QSize(45, 16777215))
        self.comboBox_choroid_line.setFont(font)

        self.horizontalLayout_21.addWidget(self.comboBox_choroid_line)

        self.horizontalLayout_21.setStretch(0, 2)
        self.horizontalLayout_21.setStretch(1, 1)

        self.verticalLayout_5.addLayout(self.horizontalLayout_21)

        self.checkBox_all = QCheckBox(self.groupBox)
        self.checkBox_all.setObjectName(u"checkBox_all")
        self.checkBox_all.setFont(font)

        self.verticalLayout_5.addWidget(self.checkBox_all)

        self.verticalSpacer = QSpacerItem(20, 0, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.verticalLayout_5.addItem(self.verticalSpacer)


        self.horizontalLayout_22.addLayout(self.verticalLayout_5)


        self.verticalLayout.addWidget(self.groupBox)

        self.verticalSpacer_2 = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.verticalLayout.addItem(self.verticalSpacer_2)


        self.retranslateUi(LayerSegSideControlPanel)

        QMetaObject.connectSlotsByName(LayerSegSideControlPanel)
    # setupUi

    def retranslateUi(self, LayerSegSideControlPanel):
        LayerSegSideControlPanel.setWindowTitle(QCoreApplication.translate("LayerSegSideControlPanel", u"Form", None))
        self.groupBox_3.setTitle("")
        self.pushButton_ai_seg.setText(QCoreApplication.translate("LayerSegSideControlPanel", u"Run AI Segmentation", None))
#if QT_CONFIG(tooltip)
        self.checkBox_sparse.setToolTip(QCoreApplication.translate("LayerSegSideControlPanel", u"<html><head/><body><p>Enable sparse mode to perform segmentation. This mode is suitable for sparse sanning OCT data. </p></body></html>", None))
#endif // QT_CONFIG(tooltip)
        self.checkBox_sparse.setText(QCoreApplication.translate("LayerSegSideControlPanel", u"Sparse data mode", None))
        self.label_30.setText(QCoreApplication.translate("LayerSegSideControlPanel", u"Device:", None))
        self.label_4.setText(QCoreApplication.translate("LayerSegSideControlPanel", u"Corrector:", None))
        self.comboBox_corrector.setItemText(0, QCoreApplication.translate("LayerSegSideControlPanel", u"None", None))
        self.comboBox_corrector.setItemText(1, QCoreApplication.translate("LayerSegSideControlPanel", u"Liner", None))
        self.comboBox_corrector.setItemText(2, QCoreApplication.translate("LayerSegSideControlPanel", u"Livewire", None))
        self.comboBox_corrector.setItemText(3, QCoreApplication.translate("LayerSegSideControlPanel", u"Anchors", None))

        self.label_3.setText(QCoreApplication.translate("LayerSegSideControlPanel", u"Interpolation Step:", None))
        self.spinBox_Interp_step.setPrefix("")
        self.pushButton_Switch_Interp.setText(QCoreApplication.translate("LayerSegSideControlPanel", u"Interpolation", None))
        self.pushButton_Restart_Interp.setText(QCoreApplication.translate("LayerSegSideControlPanel", u"Restart Interplation", None))
        self.checkBox_keep_fluid.setText(QCoreApplication.translate("LayerSegSideControlPanel", u"Keep Fluid volume", None))
        self.groupBox.setTitle(QCoreApplication.translate("LayerSegSideControlPanel", u"Layer boundaries", None))
        self.checkBox_pvd.setText(QCoreApplication.translate("LayerSegSideControlPanel", u"0. PVD", None))
        self.comboBox_pvd_line.setItemText(0, QCoreApplication.translate("LayerSegSideControlPanel", u"--", None))
        self.comboBox_pvd_line.setItemText(1, QCoreApplication.translate("LayerSegSideControlPanel", u"-", None))

        self.checkBox_ilm.setText(QCoreApplication.translate("LayerSegSideControlPanel", u"1. ILM", None))
        self.comboBox_ilm_line.setItemText(0, QCoreApplication.translate("LayerSegSideControlPanel", u"--", None))
        self.comboBox_ilm_line.setItemText(1, QCoreApplication.translate("LayerSegSideControlPanel", u"-", None))

        self.checkBox_nflgcl.setText(QCoreApplication.translate("LayerSegSideControlPanel", u"2. NFL/GCL", None))
        self.comboBox_nflgcl_line.setItemText(0, QCoreApplication.translate("LayerSegSideControlPanel", u"--", None))
        self.comboBox_nflgcl_line.setItemText(1, QCoreApplication.translate("LayerSegSideControlPanel", u"-", None))

        self.checkBox_gclipl.setText(QCoreApplication.translate("LayerSegSideControlPanel", u"3. GCL/IPL", None))
        self.comboBox_gclipl_line.setItemText(0, QCoreApplication.translate("LayerSegSideControlPanel", u"--", None))
        self.comboBox_gclipl_line.setItemText(1, QCoreApplication.translate("LayerSegSideControlPanel", u"-", None))

        self.checkBox_iplinl.setText(QCoreApplication.translate("LayerSegSideControlPanel", u"4. IPL/INL", None))
        self.comboBox_iplinl_line.setItemText(0, QCoreApplication.translate("LayerSegSideControlPanel", u"--", None))
        self.comboBox_iplinl_line.setItemText(1, QCoreApplication.translate("LayerSegSideControlPanel", u"-", None))

        self.checkBox_inlopl.setText(QCoreApplication.translate("LayerSegSideControlPanel", u"5. INL/OPL", None))
        self.comboBox_inlopl_line.setItemText(0, QCoreApplication.translate("LayerSegSideControlPanel", u"--", None))
        self.comboBox_inlopl_line.setItemText(1, QCoreApplication.translate("LayerSegSideControlPanel", u"-", None))

        self.checkBox_oplonl.setText(QCoreApplication.translate("LayerSegSideControlPanel", u"6. OPL/ONL", None))
        self.comboBox_oplonl_line.setItemText(0, QCoreApplication.translate("LayerSegSideControlPanel", u"--", None))
        self.comboBox_oplonl_line.setItemText(1, QCoreApplication.translate("LayerSegSideControlPanel", u"-", None))

        self.checkBox_elm.setText(QCoreApplication.translate("LayerSegSideControlPanel", u"7. ELM", None))
        self.comboBox_elm_line.setItemText(0, QCoreApplication.translate("LayerSegSideControlPanel", u"--", None))
        self.comboBox_elm_line.setItemText(1, QCoreApplication.translate("LayerSegSideControlPanel", u"-", None))

        self.checkBox_ez.setText(QCoreApplication.translate("LayerSegSideControlPanel", u"8. EZ", None))
        self.comboBox_ez_line.setItemText(0, QCoreApplication.translate("LayerSegSideControlPanel", u"--", None))
        self.comboBox_ez_line.setItemText(1, QCoreApplication.translate("LayerSegSideControlPanel", u"-", None))

        self.checkBox_eziz.setText(QCoreApplication.translate("LayerSegSideControlPanel", u"9. EZ/IZ", None))
        self.comboBox_eziz_line.setItemText(0, QCoreApplication.translate("LayerSegSideControlPanel", u"--", None))
        self.comboBox_eziz_line.setItemText(1, QCoreApplication.translate("LayerSegSideControlPanel", u"-", None))

        self.checkBox_izrpe.setText(QCoreApplication.translate("LayerSegSideControlPanel", u"10. IZ/RPE", None))
        self.comboBox_izrpe_line.setItemText(0, QCoreApplication.translate("LayerSegSideControlPanel", u"--", None))
        self.comboBox_izrpe_line.setItemText(1, QCoreApplication.translate("LayerSegSideControlPanel", u"-", None))

        self.checkBox_rpebm.setText(QCoreApplication.translate("LayerSegSideControlPanel", u"11. RPE/BM", None))
        self.comboBox_rpebm_line.setItemText(0, QCoreApplication.translate("LayerSegSideControlPanel", u"--", None))
        self.comboBox_rpebm_line.setItemText(1, QCoreApplication.translate("LayerSegSideControlPanel", u"-", None))

        self.checkBox_sathal.setText(QCoreApplication.translate("LayerSegSideControlPanel", u"12. SAT/HAL", None))
        self.comboBox_sathal_line.setItemText(0, QCoreApplication.translate("LayerSegSideControlPanel", u"--", None))
        self.comboBox_sathal_line.setItemText(1, QCoreApplication.translate("LayerSegSideControlPanel", u"-", None))

        self.checkBox_choroid.setText(QCoreApplication.translate("LayerSegSideControlPanel", u"13. CHOROID", None))
        self.comboBox_choroid_line.setItemText(0, QCoreApplication.translate("LayerSegSideControlPanel", u"--", None))
        self.comboBox_choroid_line.setItemText(1, QCoreApplication.translate("LayerSegSideControlPanel", u"-", None))

        self.checkBox_all.setText(QCoreApplication.translate("LayerSegSideControlPanel", u"Select All", None))
    # retranslateUi

