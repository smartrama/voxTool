__author__ = 'iped'
import os
os.environ['ETS_TOOLKIT'] = 'qt4'
from pyface.qt import QtGui, QtCore
from view.pyloc import PylocControl
import yaml

if __name__ == '__main__':
    config = yaml.load(open(os.path.join(os.path.dirname(__file__), 'config.yml')))
    controller = PylocControl(config)
    #controller = PyLocControl('/Users/iped/PycharmProjects/voxTool/R1170J_CT_combined.nii.gz')
    controller.exec_()
