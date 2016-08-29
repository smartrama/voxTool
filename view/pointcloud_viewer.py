import numpy as np
from model.pointcloud import PointCloud
from mayavi import mlab
from traits.api import HasTraits, Instance, on_trait_change
from traitsui.api import View, Item
from mayavi.core.ui.api import MayaviScene, MlabSceneModel, \
        SceneEditor
from pyface.qt import QtGui, QtCore
import random

__author__ = 'iped'

class PointCloudController(object):

    def __init__(self, parent):
        self.parent = parent
        self.view = PointCloudWidget(self)




class PointCloudWidget(QtGui.QWidget):

    def __init__(self, controller, parent=None ):
        QtGui.QWidget.__init__(self, parent)
        layout = QtGui.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.viewer = PointCloudViewer()

        self.ui = self.viewer.edit_traits(parent=self,
                                          kind='subpanel').control
        layout.addWidget(self.ui)
        #self.ui.setParent(self)
        self.ct = None
        self.controller = controller

    def update(self):
        super(PointCloudWidget, self).update()
        self.viewer.update()

    def reload_ct(self, ct):
        self.viewer.clear_views()
        self.load_ct(ct)

    def load_ct(self, ct):
        ct_view = PointCloudView(ct.all_points, -1, self.ct_callback, True)
        selected_view = PointCloudView(ct.selected_points, 10, self.selected_callback)
        self.viewer.add_view(ct_view)
        self.viewer.add_view(selected_view)
        self.viewer.plot()
        self.ct = ct

    def selected_callback(self, coordinate):
        popup = Popup()
        popup.show()
        popup.raise_()

    def add_view(self, view):
        self.viewer.add_view(view)
        self.viewer.update()

    def add_grid(self, grid):
        self.viewer.add_view(PointCloudView(grid))

    def ct_callback(self, coordinate):
        centered_coordinate = self.ct.select_centered_points_near(coordinate)
        self.viewer.update()
        self.controller.notify_slice_viewers(centered_coordinate)


class PointCloudViewer(HasTraits):
    BACKGROUND_COLOR = (.1, .1, .1)

    scene = Instance(MlabSceneModel, ())

    def __init__(self, figure=None):
        super(PointCloudViewer, self).__init__()
        self.views = []
        self.figure = self.scene.mlab.gcf()
        mlab.figure(self.figure, bgcolor=self.BACKGROUND_COLOR)
        self.plotted = False
        self.ct = None
        self.picker = None

    def clear_views(self):
        self.views = []
        self.figure.clf()

    def add_view(self, view):
        self.views.append(view)
        self.picker.point_cloud_views = self.views

    def plot(self):
        if not self.picker:
            self.picker =  MultiPlotPicker(self.figure, self.views)
        all_views = sorted(self.views, key=lambda v: v.priority)

        for point_cloud_view in all_views:
            point_cloud_view.plot()
        self.plotted = True

    @on_trait_change('scene.activated')
    def update(self):
        if not self.plotted:
            self.plot()
        else:
            self.picker.point_cloud_views = self.views
            mlab.figure(self.figure, bgcolor=self.BACKGROUND_COLOR)
            views = sorted(self.views, key=lambda view: view.priority)
            for view in views:
                view.update()

    view = View(Item('scene', editor=SceneEditor(scene_class=MayaviScene),
                     height=250, width=300, show_label=False),
                resizable=True  # We need this to resize with the parent widget
                )


class MultiPlotPicker(object):

    def __init__(self, figure, point_cloud_views):
        self.figure = figure
        self.views = point_cloud_views

        self.picker = figure.on_mouse_pick(self.picker_callback)
        self.picker.tolerance = 0.01

    def picker_callback(self, picker):
        found = False
        views_in = []
        for view in self.views:
            if view.contains_picker(picker):
                views_in.append(view)
        for view in views_in:
            blocked = view.picker_callback(picker)
            if blocked:
                return True
        return found


class PointCloudView(object):

    def choose_colormap(self):
        if self.point_cloud.label == '_ct':
            return 'bone'
        elif self.point_cloud.label == '_selected':
            return 'summer'
        else:
            return 'spectral'

    def choose_color(self):
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

    def __init__(self, point_cloud, priority=0, picker_callback=lambda *_:None, never_update=False):
        self.point_cloud = point_cloud
        self._custom_picker_callback = picker_callback
        self.color = None
        self.colormap = self.choose_colormap()
        self._plot = None
        self._glyph_points = None
        self.never_update = never_update
        self.priority = priority

    def plot(self):
        x, y, z = self.point_cloud.xyz
        color = self.choose_color()
        if len(x) > 0:
            self._plot = mlab.points3d(x, y, z,
                                       color, mode='cube', resolution=3,
                                       colormap=self.colormap, vmax=1, vmin=0,
                                       scale_mode='none', scale_factor=1)


    def update(self):
        if not self._plot:
            self.plot()
        elif not self.never_update:
            x, y, z = self.point_cloud.xyz
            self._plot.mlab_source.reset(x=x, y=y, z=z, scalars=self.choose_color())

    def contains_picker(self, picker):
        return self._plot and picker.pick_position in self.point_cloud

    def picker_callback(self, picker):
        return self._custom_picker_callback(np.array(picker.pick_position))


class Popup(QtGui.QDialog):

    def __init__(self, parent=None):
        super(Popup, self).__init__(parent)
        self.resize(40,100)

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
