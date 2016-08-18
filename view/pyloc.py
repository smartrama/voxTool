import os
os.environ['ETS_TOOLKIT'] = 'qt4'

from pyface.qt import QtGui, QtCore
from pointcloud_viewer import PointCloudWidget
from model.pointcloud import CT
from mayavi.core.ui.api import MayaviScene, MlabSceneModel, \
        SceneEditor

from pointcloud_viewer import PointCloudViewer

from slice_viewer import SliceViewWidget

__author__ = 'iped'


class PyLoc(QtGui.QWidget):

    def __init__(self, parent=None):
        QtGui.QWidget.__init__(self, parent)
        self.submission = ElectrodeSubmissionLayout(self)
        self.cloud_widget = PointCloudWidget(self)
        self.task_bar = TaskBarLayout()
        self.slice_viewers = [SliceViewWidget(self)]

        layout = QtGui.QVBoxLayout(self)
        splitter = QtGui.QSplitter()
        splitter.addWidget(self.submission)
        splitter.addWidget(self.cloud_widget)
        splitter.addWidget(self.slice_viewers[0])
        splitter.setSizes([100, 400, 200])

        layout.addWidget(splitter)
        layout.addLayout(self.task_bar)

        self.add_callbacks()

        self.ct = None



    @staticmethod
    def launch():
        app = QtGui.QApplication.instance()
        pyloc = PyLoc()
        pyloc.show()
        #pyloc.cloud_widget.load_ct(CT('/Users/iped/PycharmProjects/voxTool/sandbox/R1002P_CT_combined.img'))
        #CT.THRESHOLD=99.99
        #thresholded = CT('/Users/iped/PycharmProjects/voxTool/sandbox/R1002P_CT_combined.img')
        #thresholded.all_points.type = thresholded.all_points.TYPES.SELECTED
        #pyloc.cloud_widget.load_ct(thresholded)
        window = QtGui.QMainWindow()
        window.setCentralWidget(pyloc)
        window.show()
        app.exec_()

    def add_callbacks(self):
        self.task_bar.load_scan_button.clicked.connect(self.load_scan)
        self.submission.submit_button.clicked.connect(self.select_electrode)
        self.task_bar.clean_button.clicked.connect(self.clean_scan)

    def load_scan(self):
        file = QtGui.QFileDialog.getOpenFileName(self, 'Select Scan', '.', '(*.img; *.nii.gz)')
        if file:
            self.ct = CT(file)
            self.cloud_widget.load_ct(self.ct)
            self.slice_viewers[0].set_ct(self.ct)

    def select_electrode(self):
        electrode_name = self.submission.input_box.text()
        self.ct.confirm_selected_electrode(electrode_name)
        self.cloud_widget.update()

    def clean_scan(self):
        self.ct.remove_isolated_points()
        self.cloud_widget.update()

    def notify_slice_viewers(self, coordinate):
        for viewer in self.slice_viewers:
            viewer.set_coordinate(coordinate)
            viewer.update()


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
        self.clean_button = QtGui.QPushButton("Clean scan")
        self.addWidget(self.load_scan_button)
        self.addWidget(self.load_coord_button)
        self.addWidget(self.save_coord_button)
        self.addWidget(self.clean_button)


if __name__ == '__main__':
    PyLoc.launch()