import nibabel as nib
import numpy as np
import scipy.spatial.distance

__author__ = 'iped'

class PointCloud(object):

    TYPES = lambda: None
    TYPES.CT = 1
    TYPES.PROPOSED_ELECTRODE = 2
    TYPES.MISSING_ELECTRODE = 3
    TYPES.CONFIRMED_ELECTRODE = 4
    TYPES.SELECTED = 5

    def __init__(self, label, coordinates):
        self.coordinates, self.label =\
            coordinates, label

    def clear(self):
        self.coordinates = np.array([[],[],[]]).T

    def add_coordinates(self, coordinates):
        coord_set = self.setify_coords(coordinates)
        self.coordinates = np.array(list(coord_set.union(self.setify())))

    def remove_coordinates(self, coordinates):
        coord_set = self.setify_coords(coordinates)
        self.coordinates = np.array(list(self.setify() - coord_set))

    @staticmethod
    def setify_coords(coordinates):
        return set([tuple(coord) for coord in coordinates])

    def setify(self):
        return set([tuple(coord) for coord in self.coordinates])

    def intersect(self, point_cloud):
        self_set = self.setify()
        other_set = point_cloud.setify()
        new_array = np.array(list(self_set.intersection(other_set)))
        return PointCloud(self.label, new_array)

    def union(self, point_cloud):
        self_set = self.setify()
        other_set = point_cloud.setify()
        new_array = np.array(list(self_set.union(other_set)))
        return PointCloud(self.label, new_array)

    def move_points_to(self, coordinates, point_cloud):
        self.remove_coordinates(coordinates)
        point_cloud.add_coordinates(coordinates)

    def get_points_in_range(self, location, nearby_range=20):
        vector_dist = self.coordinates - location
        dists = np.sqrt(np.sum(np.square(vector_dist), 1))
        return self.coordinates[dists < nearby_range]

    def remove_isolated_points(self):
        print 'Getting distances...'
        dists = scipy.spatial.distance.pdist(self.coordinates, 'cityblock')
        print 'Thresholding'
        mask = scipy.spatial.distance.squareform((dists <= 1)).any(0)
        print 'Removing {} points'.format(np.count_nonzero(mask == 0))
        self.coordinates = self.coordinates[mask, :]

    def get_center(self):
        return np.mean(self.coordinates, 0)

    @property
    def xyz(self):
        if len(self.coordinates.shape) > 1 and self.coordinates.shape[1] > 0:
            return self.coordinates[:, 0], self.coordinates[:, 1], self.coordinates[:, 2]
        else:
            return [], [], []

    def __getitem__(self, item):
        return self.coordinates[item]


class Electrode(object):

    def __init__(self, point_cloud, radius=4):
        self.point_cloud = point_cloud
        self.radius = radius


class Grid(object):

    def __init__(self, electrodes, dimensions, spacing):
        self.electrodes = electrodes
        self.dimensions = dimensions
        self.spacing = spacing


class CT(object):

    THRESHOLD = 99.96

    def __init__(self, img_file):
        self.img_file = img_file
        img = nib.load(self.img_file)
        self.data = img.get_data().squeeze()
        mask = self.data >= np.percentile(self.data, self.THRESHOLD)
        indices = np.array(mask.nonzero()).T
        self.all_points = \
            PointCloud('_ct', indices)
        self.selected_points = self.empty_cloud('_selected')
        self.proposed_electrodes = []
        self.missing_electrodes = []
        self.confirmed_electrodes = []

    def remove_isolated_points(self):
        self.all_points.remove_isolated_points()

    @property
    def point_clouds(self):
        return self.all_points, self.selected_points

    @property
    def point_cloud_groups(self):
        return self.proposed_electrodes, self.missing_electrodes, self.confirmed_electrodes


    @classmethod
    def empty_cloud(cls, label):
        return PointCloud(label, np.array([[], [], []]).T)

    def select_points(self, points):
        self.selected_points.clear()
        self.selected_points.add_coordinates(points)

    def select_points_near(self, point, nearby_range=10):
        self.select_points(self.all_points.get_points_in_range(point, nearby_range))

    def select_centered_points_near(self, point, nearby_range=10, iterations=1):
        self.select_points_near(point, nearby_range)
        for _ in range(iterations):
            centered_point = self.selected_points.get_center()
            self.select_points_near(centered_point, nearby_range)
        return self.selected_points.get_center()

    def confirm_selected_electrode(self, name):
        new_cloud = self.empty_cloud(name)
        new_cloud = new_cloud.union(self.selected_points)
        self.confirmed_electrodes.append(new_cloud)
        self.selected_points.clear()

