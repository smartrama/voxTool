import os
os.environ['ETS_TOOLKIT'] = 'qt4'

from pyface.qt import QtGui, QtCore
from pointcloud_viewer import PointCloudWidget
from model.pointcloud import CT
from mayavi.core.ui.api import MayaviScene, MlabSceneModel, \
        SceneEditor

from pointcloud_viewer import PointCloudViewer

__author__ = 'iped'


class PyLoc(QtGui.QWidget):

    def __init__(self, parent=None):
        QtGui.QWidget.__init__(self, parent)
        self.submission = ElectrodeSubmissionLayout(self)
        self.cloud_widget = PointCloudWidget()
        self.task_bar = TaskBarLayout()

        layout = QtGui.QVBoxLayout(self)
        splitter = QtGui.QSplitter()
        splitter.addWidget(self.submission)
        splitter.addWidget(self.cloud_widget)

        layout.addWidget(splitter)
        layout.addLayout(self.task_bar)

        self.add_callbacks()



    @staticmethod
    def launch():
        app = QtGui.QApplication.instance()
        pyloc = PyLoc()
        pyloc.show()
        #pyloc.cloud_widget.load_ct(CT('/Users/iped/PycharmProjects/voxTool/sandbox/R1002P_CT_combined.img'))
        #CT.THRESHOLD=99.99
        #thresholded = CT('/Users/iped/PycharmProjects/voxTool/sandbox/R1002P_CT_combined.img')
        #thresholded.unselected_points.type = thresholded.unselected_points.TYPES.SELECTED
        #pyloc.cloud_widget.load_ct(thresholded)
        window = QtGui.QMainWindow()
        window.setCentralWidget(pyloc)
        window.show()
        app.exec_()

    def add_callbacks(self):
        self.task_bar.load_scan_button.clicked.connect(self.load_scan)

    def load_scan(self):
        file = QtGui.QFileDialog.getOpenFileName(self, 'Select Scan', '.', '(*.img; *.nii.gz)')
        if file:
            self.cloud_widget.load_ct(CT(file))


class ElectrodeSubmissionLayout(QtGui.QFrame):

    def __init__(self, parent=None):
        super(ElectrodeSubmissionLayout, self).__init__(parent)

        layout = QtGui.QVBoxLayout(self)
        self.input_box = QtGui.QLineEdit()
        layout.addWidget(self.input_box)

        self.submit_button = QtGui.QPushButton("Submit Electrode")
        layout.addWidget(self.submit_button)

        title = QtGui.QLabel("Electrodes")
        layout.addWidget(title)

        self.list_widget = QtGui.QListView()
        layout.addWidget(self.list_widget)

        self.modify_button = QtGui.QPushButton("Modify Electrode")
        layout.addWidget(self.modify_button)


class TaskBarLayout(QtGui.QHBoxLayout):

    def __init__(self, parent=None):
        super(TaskBarLayout, self).__init__(parent)
        self.load_scan_button = QtGui.QPushButton("Load Scan")
        self.load_coord_button = QtGui.QPushButton("Load Coordinates")
        self.save_coord_button = QtGui.QPushButton("Save Coordinates")
        self.addWidget(self.load_scan_button)
        self.addWidget(self.load_coord_button)
        self.addWidget(self.save_coord_button)


if __name__ == '__main__':
    PyLoc.launch()