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

class PyLocControl(object):

    def __init__(self):
        self.app = QtGui.QApplication.instance()
        self.view = PyLocView(self)
        self.view.show()

        window = QtGui.QMainWindow()
        window.setCentralWidget(self.view)
        window.show()

        self.ct = None

    def exec_(self):
        self.app.exec_()

    def choose_ct_scan(self):
        file = QtGui.QFileDialog.getOpenFileName(None, 'Select Scan', '.', '(*.img; *.nii.gz)')
        if file:
            self.load_ct_scan(file)

    def load_ct_scan(self, filename):
        self.ct = CT(filename)
        # TODO: cloud widget, slice viewers

    def clean_ct_scan(self):
        self.ct.remove_isolated_points()
        # TODO: Update cloud widget

    def update_cloud(self):
        self.view.update_cloud(self.ct.get_coordinates())
        self.view.update_slices(self.ct.get_slices())
        # TODO: CT.get_coordinates/slices, view.update_cloud/slices


class PyLocView(QtGui.QWidget):

    def __init__(self, controller, parent=None):
        QtGui.QWidget.__init__(self, parent)
        self.controller = controller
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

    def add_callbacks(self):
        self.task_bar.load_scan_button.clicked.connect(self.load_scan)
        self.submission.submit_button.clicked.connect(self.add_electrode)
        self.task_bar.clean_button.clicked.connect(self.clean_scan)

    def add_grid(self, grid_name):
        self.ct.add_grid(grid_name)
        self.cloud_widget.add_grid(self.ct.grids[grid_name])

    def add_electrode(self):
        grid_name = self.submission.grid_name.text()
        if not self.ct.contains_grid(grid_name):
            self.add_grid(grid_name)
        electrode_number = self.submission.electrode_number.text()
        self.ct.add_selection_to_grid(grid_name, electrode_number)
        self.cloud_widget.update()

    def notify_slice_viewers(self, coordinate):
        for viewer in self.slice_viewers:
            viewer.set_coordinate(coordinate)
            viewer.update()


class ElectrodeSubmissionLayout(QtGui.QFrame):

    def __init__(self, parent=None):
        super(ElectrodeSubmissionLayout, self).__init__(parent)

        layout = QtGui.QVBoxLayout(self)

        text_layout = QtGui.QHBoxLayout()
        self.grid_name = QtGui.QLineEdit()
        text_layout.addWidget(self.grid_name)
        self.electrode_number = QtGui.QLineEdit()
        text_layout.addWidget(self.electrode_number)
        layout.addLayout(text_layout)


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
    PyLocView.launch()