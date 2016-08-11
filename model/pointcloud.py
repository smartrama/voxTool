import nibabel as nib
import numpy as np

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

    def get_points_in_range(self, location, range=20):
        vector_dist = self.coordinates - location
        dists = np.sqrt(np.sum(np.square(vector_dist), 1))
        return self.coordinates[dists < range]

    @property
    def xyz(self):
        if len(self.coordinates.shape) > 1 and self.coordinates.shape[1] > 0:
            return self.coordinates[:, 0], self.coordinates[:, 1], self.coordinates[:, 2]
        else:
            return [], [], []

    def __getitem__(self, item):
        return self.coordinates[item]

class CT(object):

    THRESHOLD = 99.9

    def __init__(self, img_file):
        self.img_file = img_file
        img = nib.load(self.img_file)
        data = img.get_data()
        mask = data >= np.percentile(data, self.THRESHOLD)
        indices = np.array(mask.nonzero()).T
        self.unselected_points = \
            PointCloud('_ct', indices)
        self.selected_points = \
            PointCloud('_selected', np.array([[], [], []]).T)
        self.proposed_points = []
        self.missing_points = []
        self.confirmed_points = []

    @property
    def point_clouds(self):
        return [self.unselected_points, self.selected_points] + \
            self.proposed_points + self.missing_points + self.confirmed_points

    def select_points(self, points):
        self.selected_points.clear()
        self.unselected_points.move_points_to(points, self.selected_points)

    def select_points_near(self, point):
        self.unselected_points = self.unselected_points.union(self.selected_points)
        self.select_points(self.unselected_points.get_points_in_range(point))