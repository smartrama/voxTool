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

    @property
    def bounds(self):
        return self.calculate_bounds()

    def calculate_bounds(self):
        x, y, z = self.xyz
        return np.array([[min(x), min(y), min(z)], [max(x), max(y), max(z)]])

    def __contains__(self, coordinate):
        if ((coordinate - self.bounds[0,:]) > -1.5).all() and \
                ((coordinate - self.bounds[1,:]) < 1.5).all():
            return True
        return False

class Electrode(object):

    def __init__(self, point_cloud, electrode_number, grid_coordinate = (0,0), radius=4):
        self.point_cloud = point_cloud
        self.number = electrode_number
        self.radius = radius
        self.grid_coordinate = grid_coordinate
        #self.bounds = self.calculate_bounds()

    @property
    def xyz(self):
        return self.point_cloud.xyz

    @property
    def coordinates(self):
        return self.point_cloud.coordinates

    @property
    def bounds(self):
        return self.point_cloud.calculate_bounds()

    def __contains__(self, coordinate):
        return coordinate in self.point_cloud

class Grid(object):

    def __init__(self, grid_label, dimensions=(1,4), spacing=10):
        self.label = grid_label
        self.dimensions = dimensions
        self.electrodes = {}
        self.spacing = spacing

    def add_electrode(self, electrode, grid_coordinate):
        self.electrodes[grid_coordinate] = electrode

    @property
    def xyz(self):
        coordinates = np.concatenate([electrode.coordinates for electrode in self.electrodes.values()])
        return coordinates[:, 0], coordinates[:, 1], coordinates[:, 2]

    @property
    def coordinates(self):
        return np.concatenate([electrode.coordinates for electrode in self.electrodes.values()])

    def __contains__(self, coordinate):
        for electrode in self.electrodes.values():
            if coordinate in electrode:
               return True
        return False


class CT(object):

    THRESHOLD = 99.96

    def __init__(self, img_file):
        self.img_file = img_file
        img = nib.load(self.img_file)
        self.data = img.get_data().squeeze()
        mask = self.data >= np.percentile(self.data, self.THRESHOLD)
        indices = np.array(mask.nonzero()).T
        self.all_points = PointCloud('_ct', indices)
        self.selected_points = self.empty_cloud('_selected')
        self.proposed_electrodes = []
        self.missing_electrodes = []
        self.grids = {}

    def remove_isolated_points(self):
        self.all_points.remove_isolated_points()

    @property
    def point_clouds(self):
        return self.all_points, self.selected_points

    @property
    def electrodes(self):
        return self.proposed_electrodes + self.missing_electrodes

    @classmethod
    def empty_cloud(cls, label):
        return PointCloud(label, np.array([[], [], []]).T)

    def select_points(self, points):
        self.selected_points.clear()
        self.selected_points.add_coordinates(points)

    def select_points_near(self, point, nearby_range=10):
        self.select_points(self.all_points.get_points_in_range(point, nearby_range))

    def select_centered_points_near(self, point, radius=10, iterations=1):
        self.select_points_near(point, radius)
        for _ in range(iterations):
            centered_point = self.selected_points.get_center()
            self.select_points_near(centered_point, radius)
        return self.selected_points.get_center()

    def add_grid(self, grid_label, *args):
        self.grids[grid_label] = Grid(grid_label, *args)

    def contains_grid(self, grid_label):
        return grid_label in self.grids

    def add_selection_to_grid(self, grid_label, electrode_number, radius=4):
        cloud = PointCloud(grid_label, self.selected_points.coordinates)
        electrode = Electrode(cloud, electrode_number, radius)
        self.grids[grid_label].add_electrode(electrode, (electrode_number,))

