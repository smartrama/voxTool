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

class PointCloudWidget(QtGui.QWidget):

    def __init__(self, parent=None):
        QtGui.QWidget.__init__(self, parent)
        layout = QtGui.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.viewer = PointCloudViewer()

        self.ui = self.viewer.edit_traits(parent=self,
                                          kind='subpanel').control
        layout.addWidget(self.ui)
        self.ui.setParent(self)
        self.ct = None

    def load_ct(self, ct):
        self.ct_view = PointCloudView(ct.unselected_points, self.ct_callback)
        self.selected_view = PointCloudView(ct.selected_points, self.selected_callback)
        self.viewer.add_views(self.ct_view, self.selected_view)
        self.viewer.plot()
        self.ct = ct

    def ct_callback(self, coordinate):
        #selected_point = self.ct.unselected_points[index]
        self.ct.select_points_near(coordinate)
        self.viewer.update()

    def proposed_callback(self, picker):
        pass

    def missing_callback(self, picker):
        pass

    def confirmed_callback(self, picker):
        pass

    def selected_callback(self, picker):
        pass

class PointCloudViewer(HasTraits):
    scene = Instance(MlabSceneModel, ())

    def __init__(self, figure=None):
        super(PointCloudViewer, self).__init__()
        self.point_cloud_views = []
        self.point_cloud_groups = []
        self.figure = self.scene.mlab.gcf()
        self.picker = None
        self.plotted = False

    def add_views(self, *point_clouds):
        self.point_cloud_views.extend(point_clouds)
        self.refresh_picker_callback()

    def add_groups(self, *groups):
        self.point_cloud_groups.extend(groups)

    def refresh_picker_callback(self):
        if not self.picker:
            self.picker = MultiPlotPicker(self.figure, self.point_cloud_views, True)
        else:
            self.picker.point_cloud_views = self.point_cloud_views


    def plot(self):
        for point_cloud_view in self.point_cloud_views:
            point_cloud_view.plot()
        for point_cloud_group in self.point_cloud_groups:
            for point_cloud_view in point_cloud_group:
                point_cloud_view.plot()
        self.plotted = True


    @on_trait_change('scene.activated')
    def update(self):
        if not self.plotted:
            self.plot()
        else:
            self.refresh_picker_callback()
            for point_cloud_view in self.point_cloud_views:
                point_cloud_view.update()

    view = View(Item('scene', editor=SceneEditor(scene_class=MayaviScene),
                     height=250, width=300, show_label=False),
                resizable=True  # We need this to resize with the parent widget
                )


class MultiPlotPicker(object):

    def __init__(self, figure, point_cloud_views, pass_through=True):
        self.figure = figure
        self.point_cloud_views = point_cloud_views
        self.pass_through = pass_through

        self.picker = figure.on_mouse_pick(self.picker_callback)
        self.picker.tolerance = 0.01

    def picker_callback(self, picker):
        found = False
        for point_cloud_view in self.point_cloud_views:
            blocked = point_cloud_view.picker_callback(picker)
            if blocked and not self.pass_through:
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
        if len(self.point_cloud.coordinates) > 0:
            y = self.point_cloud.coordinates[:,1]
            if self.point_cloud.label == '_ct':
                return ((np.array(y) - float(min(y))) / float(max(y)))
            elif self.point_cloud.label == '_proposed_electrode':
                return np.ones(y.shape) * .7
            elif self.point_cloud.label == '_missing_electrode':
                return np.ones(y.shape) * .4
            elif self.point_cloud.label == '_selected':
                return np.ones(y.shape) * .2
            else:
                seeded_random = random.Random(self.point_cloud.label)
                return seeded_random.random(), seeded_random.random(), seeded_random.random()
        else:
            return 0

    def __init__(self, point_cloud, picker_callback=lambda *_:None):
        self.point_cloud = point_cloud
        self._custom_picker_callback = picker_callback
        self.color = self.choose_color()
        self.colormap = self.choose_colormap()
        self._plot = None
        self._glyph_points = None

    def plot(self):
        x, y, z = self.point_cloud.xyz
        color = self.choose_color()
        if len(x) > 0:
            self._plot = mlab.points3d(x, y, z,
                                       color, mode='cube', resolution=3, colormap=self.colormap, scale_mode='none', scale_factor=1)


    def update(self):
        if not self._plot:
            self.plot()
        else:
            x, y, z = self.point_cloud.xyz
            self._plot.mlab_source.reset(x=x, y=y, z=z, scalars=self.choose_color())

    def picker_callback(self, picker):
        if self._plot and picker.actor in self._plot.actor.actors:
            self._custom_picker_callback(np.array(picker.pick_position))
