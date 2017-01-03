import nibabel as nib
import numpy as np
import scipy.spatial.distance
from collections import OrderedDict
import logging
import json

log = logging.getLogger()


class PylocModelException(Exception):
    pass


class Scan(object):
    def __init__(self):
        self.filename = None
        self.data = None

    def _load_scan(self, img_file):
        self.filename = img_file
        log.debug("Loading {}".format(img_file))
        img = nib.load(self.filename)
        self.data = img.get_data().squeeze()


class PointCloud(object):
    def __init__(self, coordinates):
        self.coordinates = np.array(coordinates)

    def __len__(self):
        return len(self.coordinates)

    def clear(self):
        self.coordinates = np.array([[], [], []]).T

    def set_coordinates(self, coordinates):
        self.coordinates = np.array([tuple(coord) for coord in coordinates])

    def get_coordinates(self, mask=None):
        if mask is None:
            return self.coordinates
        else:
            return self.coordinates[mask]


class PointMask(object):
    def __init__(self, label, point_cloud, mask=None):
        self.label = label
        self.point_cloud = point_cloud
        if mask is None:
            mask = np.zeros(len(point_cloud), bool)
        self.mask = mask
        self._bounds = self._calculate_bounds()

    def copy(self):
        return PointMask(self.label, self.point_cloud, self.mask.copy())

    def clear(self):
        self.mask = np.zeros(len(self.point_cloud), bool)

    def add_points(self, coordinates):
        all_coordinates = self.point_cloud.get_coordinates()
        for coordinate in coordinates:
            self.mask[all_coordinates == coordinate] = True

    def add_mask(self, point_mask):
        self.mask = np.logical_or(self.mask, point_mask.mask)
        self._bounds = self._calculate_bounds()

    def remove_mask(self, point_mask):
        to_keep = np.logical_not(point_mask.mask)

        before = np.count_nonzero(self.mask)
        self.mask = np.logical_and(self.mask, to_keep)
        after = np.count_nonzero(self.mask)

        log.debug("Removing {} points".format(after - before))
        self._bounds = self._calculate_bounds()

    def coordinates(self):
        return self.point_cloud.get_coordinates(self.mask)

    @staticmethod
    def combined(point_masks):
        indices = np.array([])
        coords = np.array([[], [], []]).T
        labels = []
        for mask in point_masks:
            mask_indices = np.where(mask.mask)[0]
            new_indices = mask_indices[np.logical_not(np.in1d(mask_indices, indices, True))]
            if len(new_indices) > 0:
                new_coords = np.array([mask.point_cloud.get_coordinates()[i, :] for i in new_indices])
                coords = np.concatenate([coords, new_coords], 0)
                labels.extend([mask.label for _ in new_indices])
                indices = np.union1d(indices, new_indices)
        return coords, labels

    def _calculate_bounds(self):
        if len(self.point_cloud) == 0 or not self.mask.any():
            return np.array([[0, 0, 0], [0, 0, 0]])
        coords = self.coordinates()
        x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
        return np.array([[min(x), min(y), min(z)], [max(x), max(y), max(z)]])

    @property
    def bounds(self):
        return self._bounds

    def __contains__(self, coordinate):
        if ((coordinate - self.bounds[0, :]) > -1.5).all() and \
                ((coordinate - self.bounds[1, :]) < 1.5).all():
            return True
        return False

    @staticmethod
    def proximity_mask(point_cloud, point, distance):
        vector_dist = point_cloud.get_coordinates() - point
        dists = np.sqrt(np.sum(np.square(vector_dist), 1))
        return PointMask('_proximity', point_cloud, dists < distance)

    def get_center(self):
        return np.mean(self.coordinates(), 0)


class Contact(object):
    def __init__(self, point_mask, contact_label,
                 lead_location, lead_group):
        self.point_mask = point_mask.copy()
        self.label = contact_label
        self.lead_location = lead_location
        self.lead_group = lead_group

    def __contains__(self, coordinate):
        return coordinate in self.point_mask

    def coordinates(self):
        return self.point_mask.coordinates()

    @property
    def center(self):
        return np.round(self.point_mask.get_center(), 1)

    @property
    def center_str(self):
        return '({:.1f}, {:.1f}, {:.1f})'.format(*self.point_mask.get_center())

    @property
    def lead_location_str(self):
        return '({}, {})'.format(*self.lead_location)


class Lead(object):
    def __init__(self, point_cloud, lead_label, lead_type='S',
                 dimensions=(1, 5), radius=4, spacing=10):
        self.point_cloud = point_cloud
        self.label = lead_label
        self.type_ = lead_type
        self.dimensions = dimensions
        self.radius = radius
        self.spacing = spacing
        self.contacts = OrderedDict()

    def has_lead_location(self, lead_location, lead_group):
        for contact in self.contacts.values():
            if np.all(contact.lead_location == lead_location):
                if contact.lead_group == lead_group:
                    return True
        return False

    def add_contact(self, point_mask, contact_label, lead_location, lead_group):
        contact = Contact(point_mask, contact_label, lead_location, lead_group)
        if contact_label in self.contacts:
            self.remove_contact(contact_label)
        self.contacts[contact_label] = contact

    def remove_contact(self, contact_label):
        del self.contacts[contact_label]

    def coordinates(self):
        masks = [contact.point_mask for contact in self.contacts.values()]
        coords, _ = PointMask.combined(masks)
        return coords

    def get_mask(self):
        masks = [contact.point_mask for contact in self.contacts.values()]
        full_mask = np.zeros(len(self.point_cloud.get_coordinates()), bool)
        for mask in masks:
            full_mask = np.logical_or(full_mask, mask.mask)
        return PointMask(self.label, self.point_cloud, full_mask)


class CT(Scan):
    DEFAULT_THRESHOLD = 99.96

    def __init__(self, config):
        super(CT, self).__init__()
        self.config = config
        self.threshold = self.DEFAULT_THRESHOLD
        self._points = PointCloud([])
        self._leads = {}
        self._selection = PointMask("_selected", self._points)
        self.selected_lead_label = ""

    def to_dict(self):
        leads = {}
        for lead in self._leads.values():
            contacts = []
            groups = set()
            for contact in lead.contacts.values():
                groups.add(contact.lead_group)
                contacts.append(dict(
                    name=lead.label + contact.label,
                    lead_group=contact.lead_group,
                    lead_loc=contact.lead_location,
                    coordinate_spaces=dict(
                        ct_voxel=dict(
                            raw=list(contact.center)
                        )
                    )
                ))
            leads[lead.label] = dict(
                contacts=contacts,
                n_groups=len(groups),
                dimensions=lead.dimensions,
                type=lead.type_
            )
        return dict(
            leads=leads,
            origin_ct=self.filename
        )

    def from_dict(self, input_dict):
        leads = input_dict['leads']
        labels = leads.keys()
        types = [leads[label]['type'] for label in labels]
        dimensions = [leads[label]['dimensions'] for label in labels]
        radii = [self.config['lead_types'][type_]['radius'] for type_ in types]
        spacings = [self.config['lead_types'][type_]['spacing'] for type_ in types]
        self.set_leads(labels, types, dimensions, radii, spacings)
        for i, lead_label in enumerate(labels):
            for contact in leads[lead_label]['contacts']:
                coordinates = contact['coordinate_spaces']['ct_voxel']['raw']
                point_mask = PointMask.proximity_mask(self._points, coordinates, radii[i])
                group = contact['lead_group']
                loc = contact['lead_loc']
                contact_label = contact['name'].replace(lead_label, '')
                self._leads[lead_label].add_contact(point_mask, contact_label, loc, group)

    def to_json(self, filename):
        json.dump(self.to_dict(), open(filename, 'w'))

    def from_json(self, filename):
        self.from_dict(json.load(open(filename)))

    def set_leads(self, labels, lead_types, dimensions, radii, spacings):
        self._leads.clear()
        for label, lead_type, dimension, radius, spacing in \
                zip(labels, lead_types, dimensions, radii, spacings):
            log.debug("Adding lead {}, ({} {} {})".format(label, lead_type, dimension, spacing))
            self._leads[label] = Lead(self._points, label, lead_type, dimension, radius, spacing)

    def get_lead(self, lead_name):
        return self._leads[lead_name]

    def get_leads(self):
        return self._leads

    def load(self, img_file, threshold=None):
        self._load_scan(img_file)
        self.set_threshold(self.threshold if threshold is None else threshold)

    @property
    def shape(self):
        return self.data.shape

    def set_threshold(self, threshold):
        logging.debug("Threshold is set to {} percentile".format(threshold))
        self.threshold = threshold
        if self.data is None:
            raise PylocModelException("Data is not loaded")
        threshold_value = np.percentile(self.data, self.threshold)
        logging.debug("Thresholding at an intensity of {}".format(threshold_value))
        mask = self.data >= threshold_value
        logging.debug("Getting super-threshold indices")
        indices = np.array(mask.nonzero()).T
        self._points.set_coordinates(indices)
        # TODO: pointcloud should notify listening masks
        self._selection = PointMask("_selection", self._points)

    def select_points(self, point_mask):
        self._selection.clear()
        self._selection.add_mask(point_mask)

    def select_points_near(self, point, nearby_range=10):
        self.select_points(PointMask.proximity_mask(self._points, point, nearby_range))

    def selection_center(self):
        return self._selection.get_center()

    def select_weighted_center(self, point, radius=10, iterations=1):
        self.select_points_near(point, radius)
        for _ in range(iterations):
            center = self._selection.get_center()
            self.select_points_near(center, radius)
        return self._selection.get_center()

    def contact_exists(self, lead_label, contact_label):
        return lead_label in self._leads and contact_label in self._leads[lead_label].contacts

    def lead_location_exists(self, lead_label, lead_location, lead_group):
        return self._leads[lead_label].has_lead_location(lead_location, lead_group)

    def add_selection_to_lead(self, lead_label, contact_label, lead_location,
                              lead_group):
        self._leads[lead_label].add_contact(self._selection, contact_label,
                                            lead_location, lead_group)

    def set_selected_lead(self, lead_label):
        self.selected_lead_label = lead_label

    def all_xyz(self):
        c = self._points.get_coordinates()
        label = ['_ct'] * len(c[:, 0])
        return label, c[:, 0], c[:, 1], c[:, 2],

    def lead_xyz(self):
        coords, labels = PointMask.combined([lead.get_mask() for lead in self._leads.values()])
        if len(coords) == 0:
            return [], [], [], []
        return labels, coords[:, 0], coords[:, 1], coords[:, 2]

    def selection_xyz(self):
        c = self._selection.coordinates()
        label = ['_selected'] * len(c[:, 0])
        return label, c[:, 0], c[:, 1], c[:, 2]

    def xyz(self, label):
        if label == '_ct':
            return self.all_xyz()
        if label == '_leads':
            return self.lead_xyz()
        if label == '_selected':
            return self.selection_xyz()

    def clear_selection(self):
        self._selection.clear()
