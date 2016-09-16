__author__ = 'iped'
import os
os.environ['ETS_TOOLKIT'] = 'qt4'
from pyface.qt import QtGui, QtCore
from view.pyloc_view import PyLocControl

if __name__ == '__main__':
    controller = PyLocControl()
    #controller = PyLocControl('/Users/iped/PycharmProjects/voxTool/R1170J_CT_combined.nii.gz')
    controller.exec_()
