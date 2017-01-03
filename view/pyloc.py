import os

os.environ['ETS_TOOLKIT'] = 'qt4'

from pyface.qt import QtGui, QtCore
from model.scan import CT
from slice_viewer import SliceViewWidget
from mayavi.core.ui.api import MayaviScene, MlabSceneModel, \
    SceneEditor
from mayavi import mlab
from traits.api import HasTraits, Instance, on_trait_change
from traitsui.api import View, Item

import random
import numpy as np
import logging
import yaml
import re

from collections import OrderedDict

log = logging.getLogger()
log.setLevel(0)


class PylocControl(object):
    def __init__(self, config=None):

        if config == None:
            config = yaml.load(open("../model/config.yml"))

        self.app = QtGui.QApplication.instance()
        self.view = PylocWidget(self, config)
        self.view.show()

        self.window = QtGui.QMainWindow()
        self.window.setCentralWidget(self.view)
        self.window.show()

        self.lead_window = None

        self.ct = None
        self.config = config

        self.assign_callbacks()

        self.clicked_coordinate = np.zeros((3,))
        self.selected_coordinate = np.zeros((3,))

        self.selected_lead = None
        self.contact_label = ""
        self.lead_location = [0, 0]
        self.lead_group = 0

    def interpolate_selected_lead(self):
        self.selected_lead.interpolate()
        self.view.update_cloud('_leads')
        self.view.contact_panel.set_chosen_leads(self.ct.get_leads())

    def set_lead_location(self, lead_location, lead_group):
        self.lead_location = lead_location
        self.lead_group = lead_group

    def set_contact_label(self, label):
        self.contact_label = label
        self.lead_group = 0

    def set_selected_lead(self, lead_name):
        log.debug("Setting selected lead to {}".format(lead_name))
        try:
            self.selected_lead = self.ct.get_lead(lead_name)
        except KeyError:
            log.warn("Lead {} does not exist".format(lead_name))
        self.select_next_contact_label()

    def prompt_for_ct(self):
        file = QtGui.QFileDialog().getOpenFileName(None, 'Select Scan', '.', '(*.img; *.nii.gz)')
        if file:
            self.load_ct(file)
            self.view.task_bar.define_leads_button.setEnabled(True)

    def load_ct(self, filename):
        self.ct = CT(self.config)
        self.ct.load(filename)
        self.view.clear()
        self.view.add_cloud(self.ct, '_ct', callback=self.select_coordinate)
        self.view.add_cloud(self.ct, '_leads')
        self.view.add_cloud(self.ct, '_selected')
        self.view.set_slice_scan(self.ct.data)

    def exec_(self):
        self.app.exec_()

    def assign_callbacks(self):
        self.view.task_bar.load_scan_button.clicked.connect(self.prompt_for_ct)
        self.view.task_bar.define_leads_button.clicked.connect(self.define_leads)
        self.view.task_bar.save_coord_button.clicked.connect(self.save_coordinates)
        self.view.task_bar.load_coord_button.clicked.connect(self.load_coordinates)

    def save_coordinates(self):
        file = QtGui.QFileDialog().getSaveFileName(None, 'Select save file', '.', '(*.json)')
        if file:
            self.ct.to_json(file)

    def load_coordinates(self):
        file = QtGui.QFileDialog().getOpenFileName(None, 'Select voxel_coordinates.json', '.', '(*.json)')
        if file:
            self.ct.from_json(file)
            self.view.update_cloud('_leads')

    def define_leads(self):
        self.lead_window = QtGui.QMainWindow()
        lead_widget = LeadDefinitionWidget(self, self.config, self.view)
        self.lead_window.setCentralWidget(lead_widget)
        self.lead_window.show()
        self.lead_window.resize(200, lead_widget.height())

    def select_coordinate(self, coordinate, do_center=True):
        log.debug("Selecting near coordinate {}".format(coordinate))
        self.clicked_coordinate = coordinate
        self.selected_coordinate = coordinate
        radius = self.selected_lead.radius if not self.selected_lead is None else 10
        self.ct.select_points_near(coordinate, radius)
        if do_center:
            log.debug("Centering...")
            self.center_selection(self.config['selection_iterations'], radius)

        self.view.update_cloud('_selected')
        if not np.isnan(self.selected_coordinate).all():
            log.debug("Selected coordinate {}".format(self.selected_coordinate))
            self.view.update_slices(self.selected_coordinate)
            self.view.update_ras(self.selected_coordinate)
        else:
            log.debug("No coordinate selected")

    def center_selection(self, iterations, radius):
        for _ in range(iterations):
            self.selected_coordinate = self.ct.selection_center()
            self.ct.select_points_near(self.selected_coordinate, radius)

    def confirm(self, label):
        reply = QtGui.QMessageBox.question(None, 'Confirmation', label,
                                           QtGui.QMessageBox.Yes, QtGui.QMessageBox.No)
        return reply == QtGui.QMessageBox.Yes

    def add_selection(self):
        lead = self.selected_lead
        lead_label = lead.label
        contact_label = self.contact_label
        lead_location = self.lead_location[:]
        lead_group = self.lead_group

        if not self.ct.contact_exists(lead_label, contact_label) and \
                self.ct.lead_location_exists(lead_label, lead_location, lead_group):
            if not self.confirm("Lead location {} already exists. "
                                "Are you sure you want to duplicate?".format(lead_location)):
                return
        if self.config['zero_index_lead']:
            offset = 1
        else:
            offset = 0

        if lead_location[0] + offset > lead.dimensions[0] or \
                                lead_location[1] + offset > lead.dimensions[1]:
            if not self.confirm("Dimensions {} are outside of lead dimensions {}. "
                                "Are you sure you want to continue?".format(lead_location, lead.dimensions)):
                return

        self.ct.add_selection_to_lead(lead_label, contact_label, lead_location, self.lead_group)
        self.view.contact_panel.set_chosen_leads(self.ct.get_leads())
        self.ct.clear_selection()
        self.view.update_cloud('_leads')
        self.view.update_cloud('_selected')

        self.select_next_contact_label()

    def select_next_contact_label(self):
        lead = self.selected_lead
        lead_dimensions = lead.dimensions
        contacts = lead.contacts.values()

        if self.config['zero_index_lead']:
            offset = 1
        else:
            offset = 0

        increase_location = True
        if len(contacts) == 0:
            max_y = 1 - offset
            max_x = -offset
            max_label = '0'
        elif len(contacts) >= lead_dimensions[0] * lead_dimensions[1]:
            max_x, max_y = self.lead_location[0], self.lead_location[1]
            increase_location = False
        else:
            for c1, c2 in zip(contacts[:1:-1], contacts[-1::-1]):
                if c2.lead_group != c1.lead_group:
                    expected_x = 0
                    expected_y = 0
                else:
                    if c1.lead_location[0] - offset == lead.dimensions[0]:
                        expected_x = offset
                        expected_y = c1.lead_location[1] + 1
                    else:
                        expected_x = c1.lead_location[0] + 1
                        expected_y = c1.lead_location[1]
                if c2.lead_location[0] != expected_x or c2.lead_location[1] != expected_y:
                    max_y = c1.lead_location[1]
                    max_x = c1.lead_location[0]
                    max_label = contacts[-1].label
                    break
            else:
                max_y = contacts[-1].lead_location[1]
                max_x = contacts[-1].lead_location[0]
                max_label = contacts[-1].label

        if increase_location:
            increase_label = True
            if max_x + offset < lead_dimensions[0]:
                log.debug("Incrementing dimension 0")
                self.lead_location[0] = max_x + 1
                self.lead_location[1] = max_y
            elif max_y + offset == lead_dimensions[1]:
                log.debug("Not incrementing")
                increase_label = False
            else:
                log.debug("Incrementing dimension 1")
                self.lead_location[0] = 1 - offset
                self.lead_location[1] = max_y + 1

            self.view.update_lead_location(*self.lead_location)

            if increase_label and re.search(r"\d", max_label):
                num = int(re.findall(r"\d+", max_label)[-1])
                new_num = num + 1
                self.contact_label = max_label.replace(str(num), str(new_num))
                self.view.update_contact_label(self.contact_label)

    def set_leads(self, labels, lead_types, dimensions, radii, spacings):
        self.ct.set_leads(labels, lead_types, dimensions, radii, spacings)
        self.view.contact_panel.set_lead_labels(labels)

    def delete_contact(self, lead_label, contact_label):
        self.ct.get_lead(lead_label).remove_contact(contact_label)
        self.view.contact_panel.set_chosen_leads(self.ct.get_leads())
        self.view.update_cloud('_leads')


class PylocWidget(QtGui.QWidget):
    def __init__(self, controller, config, parent=None):
        QtGui.QWidget.__init__(self, parent)
        self.controller = controller
        self.cloud_widget = CloudWidget(self, config)
        self.task_bar = TaskBarLayout()
        self.slice_view = SliceViewWidget(self)
        self.contact_panel = ContactPanelWidget(controller, config, self)

        layout = QtGui.QVBoxLayout(self)
        splitter = QtGui.QSplitter()
        splitter.addWidget(self.contact_panel)
        splitter.addWidget(self.cloud_widget)
        splitter.addWidget(self.slice_view)

        layout.addWidget(splitter)
        layout.addLayout(self.task_bar)

    def clear(self):
        pass

    def update_cloud(self, label):
        self.cloud_widget.update_cloud(label)

    def update_slices(self, coordinates):
        self.slice_view.set_coordinate(coordinates)
        self.slice_view.update()

    def add_cloud(self, ct, label, callback=None):
        self.cloud_widget.add_cloud(ct, label, callback)

    def remove_cloud(self, label):
        self.cloud_widget.remove_cloud(label)

    def set_slice_scan(self, scan):
        self.slice_view.set_image(scan)

    def update_ras(self, coordinate):
        self.contact_panel.display_coordinate(coordinate)

    def update_contact_label(self, contact_label):
        self.contact_panel.set_contact_label(contact_label)

    def update_lead_location(self, x, y):
        self.contact_panel.set_lead_location(x, y)


class ContactPanelWidget(QtGui.QWidget):
    def __init__(self, controller, config, parent=None):
        super(ContactPanelWidget, self).__init__(parent)
        self.config = config
        self.controller = controller

        layout = QtGui.QVBoxLayout(self)

        lead_layout = QtGui.QHBoxLayout()
        layout.addLayout(lead_layout)

        self.label_dropdown = QtGui.QComboBox()
        self.label_dropdown.setMaximumWidth(75)
        self.add_labeled_widget(lead_layout,
                                "Label :", self.label_dropdown)
        self.contact_name = QtGui.QLineEdit()
        lead_layout.addWidget(self.contact_name)

        loc_layout = QtGui.QHBoxLayout()
        layout.addLayout(loc_layout)

        self.x_lead_loc = QtGui.QLineEdit()
        self.add_labeled_widget(loc_layout,
                                "Lead   x:", self.x_lead_loc)
        self.y_lead_loc = QtGui.QLineEdit()
        self.add_labeled_widget(loc_layout,
                                " y:", self.y_lead_loc)
        self.lead_group = QtGui.QLineEdit("0")
        self.add_labeled_widget(loc_layout,
                                " group:", self.lead_group)

        vox_layout = QtGui.QHBoxLayout()
        layout.addLayout(vox_layout)

        self.r_voxel = QtGui.QLineEdit()
        self.add_labeled_widget(vox_layout,
                                "R:", self.r_voxel)
        self.a_voxel = QtGui.QLineEdit()
        self.add_labeled_widget(vox_layout,
                                "A:", self.a_voxel)
        self.s_voxel = QtGui.QLineEdit()
        self.add_labeled_widget(vox_layout,
                                "S:", self.s_voxel)

        self.submit_button = QtGui.QPushButton("Submit")
        layout.addWidget(self.submit_button)

        contact_label = QtGui.QLabel("Contacts:")
        layout.addWidget(contact_label)

        self.contacts = []
        self.contact_list = QtGui.QListWidget()
        layout.addWidget(self.contact_list)

        self.interpolate_button = QtGui.QPushButton("Interpolate")
        layout.addWidget(self.interpolate_button)

        self.assign_callbacks()

    def display_coordinate(self, coordinate):
        self.r_voxel.setText("%.1f" % coordinate[0])
        self.a_voxel.setText("%.1f" % coordinate[1])
        self.s_voxel.setText("%.1f" % coordinate[2])

    def assign_callbacks(self):
        self.label_dropdown.currentIndexChanged.connect(self.lead_changed)
        self.contact_name.textChanged.connect(self.contact_changed)
        self.submit_button.clicked.connect(self.submit_pressed)
        self.x_lead_loc.textChanged.connect(self.lead_location_changed)
        self.y_lead_loc.textChanged.connect(self.lead_location_changed)
        self.lead_group.textChanged.connect(self.lead_location_changed)
        self.interpolate_button.clicked.connect(self.controller.interpolate_selected_lead)
        self.contact_list.currentItemChanged.connect(self.chosen_lead_selected)

    LEAD_LOC_REGEX = r'\((\d+\.?\d*),\s?(\d+\.?\d*),\s?(\d+\.?\d*)\)'

    def keyPressEvent(self, event):
        super(ContactPanelWidget, self).keyPressEvent(event)
        if event.key() == QtCore.Qt.Key_Delete:
            current_index = self.contact_list.currentIndex()
            try:
                lead, contact = self.contacts[current_index.row()]
                log.debug("Deleting contact {}{}".format(lead.label, contact.label))
                self.controller.delete_contact(lead.label, contact.label)
            except:
                log.error("Could not delete contact")

    def chosen_lead_selected(self):
        current_index = self.contact_list.currentIndex()
        _, current_contact = self.contacts[current_index.row()]
        log.debug("Selecting contact {}".format(current_contact.label))
        self.controller.select_coordinate(current_contact.center, False)

    def set_contact_label(self, label):
        self.contact_name.setText(label)

    def set_lead_location(self, x, y):
        self.x_lead_loc.setText(str(x))
        self.y_lead_loc.setText(str(y))

    def lead_location_changed(self):
        x = self.find_digit(self.x_lead_loc.text())
        self.x_lead_loc.setText(x)
        y = self.find_digit(self.y_lead_loc.text())
        self.y_lead_loc.setText(y)
        group = self.find_digit(self.lead_group.text())
        self.lead_group.setText(group)

        if len(x) > 0 and len(y) > 0 and len(group) > 0:
            self.controller.set_lead_location([int(x), int(y)], int(group))

    @staticmethod
    def find_digit(label):
        return re.sub(r"[^\d]", "", str(label))

    def lead_changed(self):
        self.controller.set_selected_lead(self.label_dropdown.currentText())
        self.lead_group.setText("0")
        self.lead_location_changed()

    def contact_changed(self):
        self.controller.set_contact_label(self.contact_name.text())

    def submit_pressed(self):
        self.controller.add_selection()

    def set_chosen_leads(self, leads):
        self.contact_list.clear()
        self.contacts = []
        for lead_name in sorted(leads.keys()):
            lead = leads[lead_name]
            for contact_name in sorted(lead.contacts.keys(), key=lambda x: int(''.join(re.findall('\d+', x)))):
                contact = lead.contacts[contact_name]
                self.add_contact(lead, contact)

    def add_contact(self, lead, contact):
        self.contact_list.addItem(
            QtGui.QListWidgetItem(self.config['lead_display'].format(lead=lead, contact=contact).strip())
        )
        self.contacts.append((lead, contact))

    def set_lead_labels(self, lead_labels):
        for lead_name in lead_labels:
            self.label_dropdown.addItem(lead_name)

    @staticmethod
    def add_labeled_widget(layout, label, widget):
        sub_layout = QtGui.QHBoxLayout()
        label_widget = QtGui.QLabel(label)
        sub_layout.addWidget(label_widget)
        sub_layout.addWidget(widget)
        layout.addLayout(sub_layout)


class LeadDefinitionWidget(QtGui.QWidget):
    instance = None

    def __init__(self, controller, config, parent=None):
        super(LeadDefinitionWidget, self).__init__(parent)
        self.config = config
        self.controller = controller

        layout = QtGui.QVBoxLayout(self)

        self.label_edit = QtGui.QLineEdit()
        self.add_labeled_widget(layout,
                                "Lead Name: ", self.label_edit)

        size_layout = QtGui.QHBoxLayout()
        size_layout.addWidget(QtGui.QLabel("Dimensions: "))
        self.x_size_edit = QtGui.QLineEdit()
        self.y_size_edit = QtGui.QLineEdit()
        self.add_labeled_widget(size_layout, "x:", self.x_size_edit)
        self.add_labeled_widget(size_layout, "y:", self.y_size_edit)
        layout.addLayout(size_layout)

        self.type_box = QtGui.QComboBox()
        for label, electrode_type in config['lead_types'].items():
            self.type_box.addItem("{}: {name}".format(label, **electrode_type))

        self.add_labeled_widget(layout, "Type: ", self.type_box)

        self.submit_button = QtGui.QPushButton("Submit")
        self.submit_button.clicked.connect(self.add_current_lead)
        layout.addWidget(self.submit_button)

        self.leads_list = QtGui.QListWidget()
        layout.addWidget(self.leads_list)

        bottom_layout = QtGui.QHBoxLayout()
        self.delete_button = QtGui.QPushButton("Delete")
        self.close_button = QtGui.QPushButton("Confirm")
        self.close_button.clicked.connect(self.finish)

        bottom_layout.addWidget(self.delete_button)
        bottom_layout.addWidget(self.close_button)
        layout.addLayout(bottom_layout)

        self._leads = OrderedDict()

    @classmethod
    def launch(cls, controller, config, parent=None):
        window = QtGui.QMainWindow()
        widget = cls(controller, config, parent)
        window.setCentralWidget(widget)
        window.show()
        window.resize(200, cls.instance.height())
        return window

    def finish(self):
        leads = self._leads.values()
        labels = [lead['label'] for lead in leads]
        types = [lead['type'] for lead in leads]
        dimensions = [(lead['x'], lead['y']) for lead in leads]
        spacings = [self.config['lead_types'][lead_type]['spacing'] for lead_type in types]
        radii = [self.config['lead_types'][lead_type]['radius'] for lead_type in types]
        self.controller.set_leads(labels, types, dimensions, radii, spacings)
        self.close()
        self.controller.lead_window.close()

    def set_leads(self, leads):
        self._leads = OrderedDict(sorted(leads.items()))
        self.refresh()

    def refresh(self):
        self.leads_list.clear()
        for lead in self._leads.values():
            self.leads_list.addItem(
                QtGui.QListWidgetItem(
                    "{label} ({x} x {y}, {type})".format(**lead)
                )
            )

    def add_current_lead(self):
        x_str = str(self.x_size_edit.text())
        y_str = str(self.y_size_edit.text())

        if not x_str.isdigit() or not y_str.isdigit():
            return

        type_str = self.type_box.currentText()
        label_str = str(self.label_edit.text())
        self._leads[label_str] = dict(
            label=label_str,
            x=int(x_str),
            y=int(y_str),
            type=type_str[0]
        )
        self.refresh()

    @staticmethod
    def add_labeled_widget(layout, label, widget):
        sub_layout = QtGui.QHBoxLayout()
        label_widget = QtGui.QLabel(label)
        sub_layout.addWidget(label_widget)
        sub_layout.addWidget(widget)
        layout.addLayout(sub_layout)


class TaskBarLayout(QtGui.QHBoxLayout):
    def __init__(self, parent=None):
        super(TaskBarLayout, self).__init__(parent)
        self.load_scan_button = QtGui.QPushButton("Load Scan")
        self.define_leads_button = QtGui.QPushButton("Define Leads")
        self.define_leads_button.setEnabled(False)
        self.load_coord_button = QtGui.QPushButton("Load Coordinates")
        self.save_coord_button = QtGui.QPushButton("Save Coordinates")
        self.clean_button = QtGui.QPushButton("Clean scan")
        self.addWidget(self.load_scan_button)
        self.addWidget(self.define_leads_button)
        self.addWidget(self.load_coord_button)
        self.addWidget(self.save_coord_button)
        self.addWidget(self.clean_button)


class CloudWidget(QtGui.QWidget):
    def __init__(self, controller, config, parent=None):
        super(CloudWidget, self).__init__(parent)
        self.config = config
        layout = QtGui.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.viewer = CloudViewer(config)
        self.ui = self.viewer.edit_traits(parent=self,
                                          kind='subpanel').control

        layout.addWidget(self.ui)
        self.controller = controller

    def update_cloud(self, label):
        self.viewer.update_cloud(label)

    def add_cloud(self, ct, label, callback=None):
        self.viewer.add_cloud(ct, label, callback)

    def remove_cloud(self, label):
        self.viewer.remove_cloud(label)


class CloudViewer(HasTraits):
    BACKGROUND_COLOR = (.1, .1, .1)

    scene = Instance(MlabSceneModel, ())

    def __init__(self, config):
        super(CloudViewer, self).__init__()
        self.config = config
        self.figure = self.scene.mlab.gcf()
        mlab.figure(self.figure, bgcolor=self.BACKGROUND_COLOR)
        self.clouds = {}

    def update_cloud(self, label):
        self.clouds[label].update()

    def add_cloud(self, ct, label, callback=None):
        self.clouds[label] = CloudView(ct, label, self.config, callback)
        self.clouds[label].plot()

    def remove_cloud(self, label):
        self.clouds[label].unplot()
        del self.clouds[label]

    @on_trait_change('scene.activated')
    def plot(self):
        self.figure.on_mouse_pick(self.callback)
        for view in self.clouds.values():
            view.plot()

    def update_all(self):
        mlab.figure(self.figure, bgcolor=self.BACKGROUND_COLOR)
        for view in self.clouds.values():
            view.update()

    def callback(self, picker):
        found = False
        for cloud in self.clouds.values():
            if cloud.contains(picker):
                if cloud.callback(picker):
                    return True
                found = True
        return found

    view = View(Item('scene', editor=SceneEditor(scene_class=MayaviScene),
                     height=250, width=300, show_label=False),
                resizable=True)


class CloudView(object):
    def get_colormap(self, label):
        if label == '_ct':
            return self.config['colormaps']['ct']
        elif label == '_selected':
            return self.config['colormaps']['selected']
        else:
            return self.config['colormaps']['default']

    def __init__(self, ct, label, config, callback=None):
        self.ct = ct
        self.config = config
        self.label = label
        self.colormap = self.get_colormap(label)
        self._callback = callback if callback else lambda *_: None
        self._plot = None
        self._glyph_points = None

    def callback(self, picker):
        return self._callback(np.array(picker.pick_position))

    def get_colors(self, labels, x, y, z):
        colors = np.ones(len(x))
        if len(labels) == 0:
            return []
        min_y = float(min(y))
        max_y = float(max(y))
        for i, label in enumerate(labels):
            if label == '_ct':
                colors[i] = ((y[i] - min_y) / max_y) * \
                            (self.config['ct_max_color'] - self.config['ct_min_color']) \
                            + self.config['ct_min_color']
            elif label == '_selected':
                colors[i] = .2
            else:
                seeded_rand = random.Random(label)
                colors[i] = seeded_rand.random() * \
                            (self.config['lead_max_color'] - self.config['lead_min_color']) \
                            + self.config['lead_min_color']
        return colors

    def contains(self, picker):
        return True if self._plot else False  # and picker.pick_position in self.ct.xyz(self.label)

    def plot(self):
        labels, x, y, z = self.ct.xyz(self.label)
        self._plot = mlab.points3d(x, y, z,  # self.get_colors(labels, x, y, z),
                                   mode='cube', resolution=3,
                                   colormap=self.colormap,
                                   opacity=.5,
                                   vmax=1, vmin=0,
                                   scale_mode='none', scale_factor=1)
        self._plot.mlab_source.set(scalars=self.get_colors(labels, x, y, z))

    def unplot(self):
        self._plot.mlab_source.reset(x=[], y=[], z=[], scalars=[])

    def update(self):
        labels, x, y, z = self.ct.xyz(self.label)
        log.debug("Updating cloud {} with {} points".format(self.label, len(labels)))
        self._plot.mlab_source.reset(
            x=x, y=y, z=z, scalars=self.get_colors(labels, x, y, z))


if __name__ == '__main__':
    # controller = PylocControl(yaml.load(open(os.path.join(os.path.dirname(__file__) , "../config.yml"))))
    # controller = PyLocControl('/Users/iped/PycharmProjects/voxTool/R1170J_CT_combined.nii.gz')
    controller = PylocControl(yaml.load(open(os.path.join(os.path.dirname(__file__), "../config.yml"))))

    # controller.load_ct("../T01_R1248P_CT.nii.gz")
    controller.load_ct('/Volumes/rhino_mount/data10/RAM/subjects/R1226D/tal/images/combined/R1226D_CT_combined.nii.gz')
    controller.set_leads(
        #    ["sA", "sB", "dA", "dB"], ["S", "S", "D", "D"], ([[6, 1]] * 2) + ([[8, 1]] * 2), ([5] * 2) + ([5] * 2), [10] * 4
        ["GG"], ["G"], ([[4, 8]]), [5], [10]
        # ["dA", "dB", "dC"], ["D", "D", "G"], [[8, 1], [8, 1], [4, 4]], [5, 10, 10], [10, 20, 20]
    )
    controller.exec_()

if __name__ == 'x__main__':
    app = QtGui.QApplication.instance()
    x = LeadDefinitionWidget(None, yaml.load(open(os.path.join(os.path.dirname(__file__), "../model/config.yml"))))
    x.show()
    window = QtGui.QMainWindow()
    window.setCentralWidget(x)
    window.show()
    app.exec_()
