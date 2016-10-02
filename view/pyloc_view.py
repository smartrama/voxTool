import os

os.environ['ETS_TOOLKIT'] = 'qt4'

from model.pointcloud import CT, Grid

import numpy as np
from mayavi import mlab
from traits.api import HasTraits, Instance, on_trait_change
from traitsui.api import View, Item
from mayavi.core.ui.api import MayaviScene, MlabSceneModel, \
    SceneEditor
from pyface.qt import QtGui, QtCore
import random

from slice_viewer import SliceViewWidget

__author__ = 'iped'


class PyLocControl(object):
    def __init__(self, ct_filename=None):

        self.app = QtGui.QApplication.instance()
        self.view = PyLocView(self)
        self.view.show()

        self.window = QtGui.QMainWindow()
        self.window.setCentralWidget(self.view)
        self.window.show()

        self.ct = None

        if ct_filename:
            self.load_ct_scan(ct_filename)

    def exec_(self):
        self.app.exec_()

    def choose_ct_scan(self):
        file = QtGui.QFileDialog.getOpenFileName(None, 'Select Scan', '.', '(*.img; *.nii.gz)')
        if file:
            self.load_ct_scan(file)

    def load_ct_scan(self, filename):
        self.ct = CT(filename)
        self.view.clear()
        self.view.add_cloud(self.ct.all_points, callback=self.select_coordinate)
        self.view.add_cloud(self.ct.selected_points)
        self.view.set_slice_image(self.ct.data)

    def clean_ct_scan(self):
        self.ct.remove_isolated_points()
        self.view.update_cloud(self.ct.all_points.label)

    def update_cloud(self, cloud):
        self.view.update_cloud(cloud)

    def select_coordinate(self, coordinate):
        centered_coordinate = self.ct.select_weighted_center(coordinate, radius=1)
        self.view.update_cloud(self.ct.selected_points.label)
        if np.isnan(centered_coordinate).all():
            return
        self.view.update_slices(centered_coordinate)

    def save_coord(self, f_handler):
        f = open(f_handler, "w")

        coord_txt = ''

        for grid_label in self.ct.grids.keys():
            for coord in sorted(self.ct.grids[grid_label].electrodes.keys(),
                                key=lambda x: self.ct.grids[grid_label].electrodes[x].label):
                c = self.ct.grids[grid_label].electrodes[coord].point_cloud.get_center()
                electrode_label = self.ct.grids[grid_label].electrodes[coord].label
                coord_txt += '{},{},{},{},{}\n'.format(grid_label, electrode_label, c[0], c[1], c[2])
        f.write(coord_txt)
        f.close()

    def save_coord_csv(self):
        f = QtGui.QFileDialog.getOpenFileName(None, 'Select Scan', '.', '(*.csv)')
        self.save_coord(f)

    GRID_PRIORITY = 1

    def add_grid(self, grid):
        self.ct.grids[grid.label] = grid
        self.view.add_cloud(grid, self.GRID_PRIORITY)

    def add_electrode(self, electrode, grid_label, grid_coordinates):
        self.ct.grids[grid_label].add_electrode(electrode, grid_coordinates)
        self.view.update_cloud(grid_label)
        self.view.update_list(self.ct.grids.values())

    def add_selected_electrode(self):
        grid_label = self.get_grid_label()
        electrode_label = self.get_electrode_label()
        grid_coordinates = self.get_grid_coordinates()
        grid_dimensions = self.get_grid_dimensions()
        if not self.ct.contains_grid(grid_label):
            self.add_grid(Grid(grid_label, dimensions=grid_dimensions))
        electrode = self.ct.create_electrode_from_selection(electrode_label, 10)
        self.add_electrode(electrode, grid_label, grid_coordinates)
        self.view.submission_layout.contact_edit.setText(str(int(electrode_label) + 1))
        if grid_dimensions[1] == 1:
            self.view.submission_layout.coordinates_x_edit.setText(
                str(grid_coordinates[0] + 1)
            )
        else:
            self.view.submission_layout.coordinates_y_edit.setText(
                str(grid_coordinates[1] + 1)
            )

    def key_pressed(self, e):
        self.add_selected_electrode()

    def get_grid_label(self):
        return str(self.view.submission_layout.lead_edit.text())

    def get_electrode_label(self):
        return str(self.view.submission_layout.contact_edit.text())

    def get_grid_coordinates(self):
        return (int(self.view.submission_layout.coordinates_x_edit.text()),
                int(self.view.submission_layout.coordinates_y_edit.text()))

    def get_grid_dimensions(self):
        return (int(self.view.submission_layout.dimensions_x_edit.text()),
                int(self.view.submission_layout.dimensions_y_edit.text()))

    def update_electrode_lead(self):
        self.view.submission_layout.contact_grid_label.setText(
            self.view.submission_layout.lead_edit.text()
        )

    def interpolate_main(self):
        # Interpolate over all grids/strips
        self.ct.interpolate_all()
        for grid_label in self.ct.grids.keys():
            self.view.update_cloud(grid_label)
        self.view.update_list(self.ct.grids.values())





class PyLocView(QtGui.QWidget):
    def __init__(self, controller, parent=None):
        QtGui.QWidget.__init__(self, parent)
        self.controller = controller
        self.submission_layout = ElectrodeSubmissionLayout(self)
        self.cloud_widget = PointCloudWidget(self)
        self.task_bar = TaskBarLayout()
        self.ct_slice_viewer = SliceViewWidget(self)

        layout = QtGui.QVBoxLayout(self)
        splitter = QtGui.QSplitter()
        splitter.addWidget(self.submission_layout)
        splitter.addWidget(self.cloud_widget)
        splitter.addWidget(self.ct_slice_viewer)
        splitter.setSizes([100, 400, 200])

        layout.addWidget(splitter)
        layout.addLayout(self.task_bar)

        self.add_callbacks()

        self.ct = None

    def add_callbacks(self):
        self.task_bar.load_scan_button.clicked.connect(self.controller.choose_ct_scan)
        self.task_bar.save_coord_button.clicked.connect(self.controller.save_coord_csv)
        self.submission_layout.submit_button.clicked.connect(self.controller.add_selected_electrode)
        self.submission_layout.interpolate_button.clicked.connect(self.controller.interpolate_main)
        self.task_bar.clean_button.clicked.connect(self.controller.clean_ct_scan)
        self.submission_layout.lead_edit.textChanged.connect(self.controller.update_electrode_lead)
        self.cloud_widget.viewer.scene.interactor.add_observer('KeyPresEvent', self.controller.key_pressed)

    def update_slices(self, coordinate):
        self.ct_slice_viewer.set_coordinate(coordinate)
        self.ct_slice_viewer.update()

    def update_cloud(self, cloud_label):
        self.cloud_widget.update_cloud(cloud_label)

    def update_list(self, grids):
        self.submission_layout.list_widget.clear()

        for grid in sorted(grids, key=lambda grid: grid.label):
            for coordinates, lead in sorted(grid.electrodes.items()):
                self.submission_layout.list_widget.addItem(
                    '{}{} {} ({},{})'.format(grid.label, lead.label, lead.type, *coordinates)
                )

    def add_cloud(self, cloud, priority=0, callback=None):
        self.cloud_widget.add_cloud(cloud, priority, callback)

    def remove_cloud(self, cloud_label):
        self.cloud_widget.remove_cloud(cloud_label)

    def set_slice_image(self, image):
        self.ct_slice_viewer.set_image(image)

    def clear(self):
        self.cloud_widget.clear()


class ElectrodeSubmissionLayout(QtGui.QFrame):
    def __init__(self, parent=None):
        super(ElectrodeSubmissionLayout, self).__init__(parent)

        layout = QtGui.QVBoxLayout(self)

        top_layout = QtGui.QVBoxLayout()
        contact_label = QtGui.QLabel("Lead Information")
        top_layout.addWidget(contact_label)

        lead_layout = QtGui.QHBoxLayout()
        self.lead_edit = QtGui.QLineEdit()
        lead_layout.addWidget(self.lead_edit)
        dimensions_layout = QtGui.QHBoxLayout()
        dimensions_layout.setSpacing(0)
        dimensions_layout.addWidget(QtGui.QLabel("("))
        self.dimensions_x_edit = QtGui.QLineEdit()
        self.dimensions_x_edit.setMinimumWidth(30)
        self.dimensions_x_edit.setMaximumWidth(30)
        dimensions_layout.addWidget(self.dimensions_x_edit)
        dimensions_layout.addWidget(QtGui.QLabel(","))
        self.dimensions_y_edit = QtGui.QLineEdit()
        self.dimensions_y_edit.setMinimumWidth(30)
        self.dimensions_y_edit.setMaximumWidth(30)
        dimensions_layout.addWidget(self.dimensions_y_edit)
        dimensions_layout.addWidget(QtGui.QLabel(")"))
        lead_layout.addLayout(dimensions_layout)

        top_layout.addLayout(lead_layout)

        layout.addLayout(top_layout)

        contact_label = QtGui.QLabel("Contact Information")
        top_layout.addWidget(contact_label)

        contact_layout = QtGui.QHBoxLayout()
        self.contact_grid_label = QtGui.QLabel("---")
        contact_layout.addWidget(self.contact_grid_label)
        self.contact_edit = QtGui.QLineEdit()
        contact_layout.addWidget(self.contact_edit)
        coordinates_layout = QtGui.QHBoxLayout()
        coordinates_layout.setSpacing(0)
        coordinates_layout.addWidget(QtGui.QLabel("("))
        self.coordinates_x_edit = QtGui.QLineEdit()
        self.coordinates_x_edit.setMinimumWidth(30)
        self.coordinates_x_edit.setMaximumWidth(30)
        coordinates_layout.addWidget(self.coordinates_x_edit)
        coordinates_layout.addWidget(QtGui.QLabel(","))
        self.coordinates_y_edit = QtGui.QLineEdit()
        self.coordinates_y_edit.setMinimumWidth(30)
        self.coordinates_y_edit.setMaximumWidth(30)
        coordinates_layout.addWidget(self.coordinates_y_edit)
        coordinates_layout.addWidget(QtGui.QLabel(")"))
        contact_layout.addLayout(coordinates_layout)

        top_layout.addLayout(contact_layout)

        self.submit_button = QtGui.QPushButton("Submit Electrode")
        layout.addWidget(self.submit_button)

        title = QtGui.QLabel("Electrodes")
        layout.addWidget(title)

        self.list_widget = QtGui.QListWidget()
        layout.addWidget(self.list_widget)

        self.interpolate_button = QtGui.QPushButton("Interpolate!")
        layout.addWidget(self.interpolate_button)

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


class PointCloudWidget(QtGui.QWidget):
    def __init__(self, controller, parent=None):
        QtGui.QWidget.__init__(self, parent)
        layout = QtGui.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.viewer = CloudViewer()

        self.ui = self.viewer.edit_traits(parent=self,
                                          kind='subpanel').control
        layout.addWidget(self.ui)
        self.controller = controller
        self.clouds = {}

    def update_cloud(self, cloud_label):
        self.viewer.update_cloud(cloud_label)

    def add_cloud(self, cloud, priority=0, callback=None):
        self.viewer.add_cloud(cloud, priority, callback)

    def remove_cloud(self, cloud_label):
        self.viewer.remove_cloud(cloud_label)

    def clear(self):
        self.viewer.clear_views()


class CloudViewer(HasTraits):
    BACKGROUND_COLOR = (.1, .1, .1)

    scene = Instance(MlabSceneModel, ())

    def __init__(self):
        super(CloudViewer, self).__init__()
        self.figure = self.scene.mlab.gcf()
        mlab.figure(self.figure, bgcolor=self.BACKGROUND_COLOR)
        self.clouds = {}
        self.picker = None

    def clear_views(self):
        for cloud_label in self.clouds.keys():
            self.remove_cloud(cloud_label)

    def update_cloud(self, cloud_label):
        self.clouds[cloud_label].update()

    def add_cloud(self, cloud, priority=0, callback=None):
        self.clouds[cloud.label] = CloudView(cloud, priority, callback)
        self.clouds[cloud.label].plot()
        self.picker.add_cloud(cloud.label, self.clouds[cloud.label])

    def remove_cloud(self, cloud_label):
        self.clouds[cloud_label].unplot()
        del self.clouds[cloud_label]
        self.picker.remove_cloud(cloud_label)

    @on_trait_change('scene.activated')
    def plot(self):
        self.picker = CloudPicker(self.figure)
        all_views = sorted(self.clouds.values(), key=lambda v: v.priority)
        for cloud_view in all_views:
            cloud_view.plot()

    def update_all(self):
        mlab.figure(self.figure, bgcolor=self.BACKGROUND_COLOR)
        views = sorted(self.views, key=lambda view: view.priority)
        for view in views:
            view.update()

    view = View(Item('scene', editor=SceneEditor(scene_class=MayaviScene),
                     height=250, width=300, show_label=False),
                resizable=True  # We need this to resize with the parent widget
                )


class CloudPicker(object):
    def __init__(self, figure):
        self.figure = figure
        self.clouds = {}

        self.picker = figure.on_mouse_pick(self.callback)
        self.picker.tolerance = 0.01

    def add_cloud(self, label, cloud):
        self.clouds[label] = cloud

    def remove_cloud(self, label):
        del self.clouds[label]

    def callback(self, picker):
        found = False
        views_in = []
        for cloud in sorted(self.clouds.values(), key=lambda cloud: cloud.priority):
            if cloud.contains(picker):
                views_in.append(cloud)
        for cloud in views_in:
            blocked = cloud.callback(picker)
            if blocked:
                return True
        return found


class CloudView(object):
    def get_colormap(self):
        if self.point_cloud.label == '_ct':
            return 'bone'
        elif self.point_cloud.label == '_selected':
            return 'summer'
        else:
            return 'spectral'

    def get_color(self):
        _, y, _ = self.point_cloud.xyz
        if self.point_cloud.label == '_ct':
            return ((np.array(y) - float(min(y))) / float(max(y))) * .6 + .4
        elif self.point_cloud.label == '_proposed_electrode':
            return np.ones(y.shape) * .7
        elif self.point_cloud.label == '_missing_electrode':
            return np.ones(y.shape) * .4
        elif self.point_cloud.label == '_selected':
            return np.ones(y.shape) * .2
        else:
            seeded_random = random.Random(self.point_cloud.label)
            return np.ones(y.shape) * seeded_random.random()

    def __init__(self, point_cloud, priority=0, callback=None):
        self.point_cloud = point_cloud
        self.priority = priority
        self._callback = callback if callback else lambda *_: None
        self._plot = None
        self._glyph_points = None

    def plot(self):
        x, y, z = self.point_cloud.xyz
        self._plot = mlab.points3d(x, y, z, self.get_color(), mask_points=10,
                                   mode='cube', resolution=3,
                                   colormap=self.get_colormap(),
                                   vmax=1, vmin=0,
                                   scale_mode='none', scale_factor=1)

    def unplot(self):
        self._plot.mlab_source.reset(x=[], y=[], z=[], scalars=[])

    def update(self):
        x, y, z = self.point_cloud.xyz
        self._plot.mlab_source.reset(x=x, y=y, z=z, scalars=self.get_color())

    def contains(self, picker):
        return self._plot and picker.pick_position in self.point_cloud

    def callback(self, picker):
        return self._callback(np.array(picker.pick_position))


class Popup(QtGui.QDialog):
    def __init__(self, parent=None):
        super(Popup, self).__init__(parent)
        self.resize(40, 100)

    def showEvent(self, event):
        geom = self.frameGeometry()
        geom.moveCenter(QtGui.QCursor.pos())
        self.setGeometry(geom)
        super(Popup, self).showEvent(event)

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Escape:
            self.hide()
            event.accept()
        else:
            super(Popup, self).keyPressEvent(event)


if __name__ == '__main__':
    PyLocView.launch()
