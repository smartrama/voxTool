import nibabel as nib
import numpy as np
import scipy.spatial.distance
from scipy.ndimage.measurements import label
from scipy.stats.mstats import mode

import util

__author__ = 'iped'


class PointCloud(object):
    TYPES = lambda: None
    TYPES.CT = 1
    TYPES.PROPOSED_ELECTRODE = 2
    TYPES.MISSING_ELECTRODE = 3
    TYPES.CONFIRMED_ELECTRODE = 4
    TYPES.SELECTED = 5

    def __init__(self, label, coordinates):
        self.coordinates, self.label = \
            coordinates, label

    def clear(self):
        self.coordinates = np.array([[], [], []]).T

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
        mask = scipy.spatial.distance.squareform((dists <= 1)).sum(0) > 1
        print 'Removing {} points'.format(np.count_nonzero(mask == 0))
        self.coordinates = self.coordinates[mask, :]

    def get_center(self):
        return np.mean(self.coordinates, 0)

    @property
    def xyz(self):
        if len(self.coordinates.shape) > 1 and self.coordinates.shape[1] > 0:
            return self.coordinates[:, 0], self.coordinates[:, 1], self.coordinates[:, 2]
        else:
            return np.array([[], [], []])

    def __getitem__(self, item):
        return self.coordinates[item]

    @property
    def bounds(self):
        return self.calculate_bounds()

    def calculate_bounds(self):
        x, y, z = self.xyz
        if len(x) == 0:
            return np.array([[0, 0, 0], [0, 0, 0]])
        return np.array([[min(x), min(y), min(z)], [max(x), max(y), max(z)]])

    def __contains__(self, coordinate):
        if ((coordinate - self.bounds[0, :]) > -1.5).all() and \
                ((coordinate - self.bounds[1, :]) < 1.5).all():
            return True
        return False


class Electrode(object):
    def __init__(self, point_cloud, electrode_label, electrode_number, grid_coordinate=(0, 0), radius=4):
        self.label = electrode_label
        self.point_cloud = point_cloud
        self.number = electrode_number
        self.radius = radius
        self.grid_coordinate = grid_coordinate
        # self.bounds = self.calculate_bounds()

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
    def __init__(self, grid_label, type='G', dimensions=(1, 4), spacing=10):
        self.label = grid_label
        self.dimensions = dimensions
        self.electrodes = {}
        self.type = type
        self.spacing = spacing

    def add_electrode(self, electrode, grid_coordinate):
        self.electrodes[grid_coordinate] = electrode

    @property
    def xyz(self):
        if self.electrodes.values():
            coordinates = np.concatenate([electrode.coordinates for electrode in self.electrodes.values()])
            return coordinates[:, 0], coordinates[:, 1], coordinates[:, 2]
        else:
            return np.array([[], [], []])

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

    # THRESHOLD = 98

    def __init__(self, img_file):
        self.img_file = img_file
        img = nib.load(self.img_file)
        img = np.fliplr(img.get_data())  # ?? CT image is flipped across sagittal for some reason
        self.data = img.squeeze()
        self.brainmask = np.zeros(nib.load(self.img_file).get_data().shape)
        mask = self.data >= max(np.percentile(self.data, self.THRESHOLD), 1)
        indices = np.array(mask.nonzero()).T
        connected_points, num_features = label(mask)
        self.connected_points = connected_points[mask]
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
        """
        Computes the nearest connected components in the point cloud that is within nearby_range of point.

        :param point:
        :param nearby_range:
        :return: None
        """
        vector_dist = self.all_points.coordinates - point
        dists = np.sqrt(np.sum(np.square(vector_dist), 1))

        conn_comp_id, _ = mode(self.connected_points[dists < nearby_range])
        while not (conn_comp_id[0]):
            nearby_range += 1
            conn_comp_id, _ = mode(self.connected_points[dists < nearby_range])
        self.select_points(self.all_points.coordinates[self.connected_points == conn_comp_id[0]])
        # self.select_points(self.all_points.get_points_in_range(point, nearby_range))

    def select_weighted_center(self, point, radius=10, iterations=1):
        self.select_points_near(point, radius)
        for _ in range(iterations):
            centered_point = self.selected_points.get_center()
            self.select_points_near(centered_point, radius)
        return self.selected_points.get_center()

    def add_grid(self, grid_label, *args):
        self.grids[grid_label] = Grid(grid_label, *args)

    def contains_grid(self, grid_label):
        return grid_label in self.grids

    def add_selection_to_grid(self, grid_label, electrode_label, grid_coordinate, radius=4):
        cloud = PointCloud(grid_label, self.selected_points.coordinates)
        electrode = Electrode(cloud, electrode_label, radius,
                              grid_coordinate=grid_coordinate, radius=radius)
        self.grids[grid_label].add_electrode(electrode, grid_coordinate)

    def create_electrode_from_selection(self, electrode_label, radius):
        cloud = PointCloud(electrode_label, self.selected_points.coordinates)
        return Electrode(cloud, electrode_label, radius)

    def add_mask(self, filename):
        mask = nib.load(filename).get_data()
        mask = np.fliplr(mask)
        self.brainmask = mask

    def interpolate_all(self):
        for grid_label in self.grids.keys():
            # Check if all points already added (manually or interpolated before)
            d = self.grids[grid_label].dimensions
            if not len(self.grids[grid_label].electrodes.keys()) == d[0] * d[1]:
                self.interpolate(grid_label)

    def interpolate(self, grid_label):
        """
        Interpolates on either grid or strip. Requires that all 4 corners in the case of a grid or 2 contacts in the
        case of strips have been labeled and submitted.

        :param grid_label: Label of grid and strip
        :return: None
        """
        d = self.grids[grid_label].dimensions

        # Check if grid
        if d[1] > 1:
            # Check to see all 4 corners present in grid
            if all(x in self.grids[grid_label].electrodes.keys() for x in [(1, 1), (1, d[1]), (d[0], 1), (d[0], d[1])]):
                # Interpolate
                start = self.grids[grid_label].electrodes[(1, 1)].label
                coor1 = self.grids[grid_label].electrodes[(1, 1)].point_cloud.get_center()
                coor2 = self.grids[grid_label].electrodes[(1, d[1])].point_cloud.get_center()
                coor3 = self.grids[grid_label].electrodes[(d[0], d[1])].point_cloud.get_center()

                points = util.interpol(coor1, coor2, coor3, d[1], d[0])

                # Select point (snap) to the closes point
                for ii, point in enumerate(points):
                    grid_coordinate = np.unravel_index([ii], d)
                    grid_coordinate = tuple(map(lambda x: int(x) + 1, grid_coordinate))
                    self.select_points_near(point, nearby_range=1)
                    self.add_selection_to_grid(grid_label, str(ii + int(start)), grid_coordinate, radius=4)
        else:
            # Check to see both ends present in strip/depth
            if all(x in self.grids[grid_label].electrodes.keys() for x in [(1, 1), (d[0], 1)]):
                # Interpolate
                start = self.grids[grid_label].electrodes[(1, 1)].label
                coor1 = self.grids[grid_label].electrodes[(1, 1)].point_cloud.get_center()
                coor2 = self.grids[grid_label].electrodes[(d[0], 1)].point_cloud.get_center()
                points = util.interpol(coor1, coor2, [], d[0], 1)

                # Select point (snap) to the closes point
                for ii, point in enumerate(points):
                    grid_coordinate = np.unravel_index([ii], d)
                    grid_coordinate = tuple(map(lambda x: int(x) + 1, grid_coordinate))
                    self.select_points_near(point, nearby_range=1)
                    self.add_selection_to_grid(grid_label, str(ii + int(start)), grid_coordinate,
                                               radius=4)
