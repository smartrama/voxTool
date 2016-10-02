#!/usr/bin/python

""" This utility module interpolates all electrode grids in a strip or grid configuration.
"""

import numpy as np

__version__ = '0.1'
__author__ = 'Xavier Islam'
__email__ = 'islamx@seas.upenn.edu'
__copyright__ = "Copyright 2016, University of Pennsylvania"
__credits__ = ["iped", "islamx", "lkini", "srdas",
               "jmstein", "kadavis"]
__status__ = "Development"


def normalize(v):
    """ returns the unit vector in the direction of v

    @param v : Vector (numpy)
    @rtype numpy array
    """

    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def totuple(a):
    """ returns numpy arrays as tuples
    """

    try:
        return tuple(totuple(i) for i in a)
    except TypeError:
        return a


def interpol(coor1, coor2, coor3, m, n):
    """ returns the coordinates of all interpolated electrodes in a
        grid/strip of configuration m x n along with numeric labels in
        order.

    It takes as inputs up to three sample voxel coordinates representing
        the oriented corners of the grid as well as the grid
        configuration (e.g. 8x8). The order of the corners determines
        the label order; i.e., coor1 corresponds to 1, coor2 corresponds
         to n*(m-1) + 1, coor3 corresponds to m*n.

    @param coor1 : Coordinate corresponding to label 1
    @param coor2 : Coordinate corresponding to the first electrode in
        the last row
    @param coor3 : Coordinate corresponding to the last electrode
        (optional in case of strip)
    @param m : Number of rows
    @param n : Number of columns
    @type coor1: list
    @type coor2: list
    @type coor3: list
    @type m: int
    @type n: int
    @rtype dict

    Example:
        >> interpol([77, 170, 81],[65, 106, 112],[104, 121, 158],8,8)
    """

    # Check if no third coordinate is present
    if not list(coor3):
        if m != 1 and n != 1:
            raise ValueError('Third corner coordinate not given despite\
                configuration being a grid.')
        return interpol_strip(coor1, coor2, m, n)
    else:
        if m == 1 or n == 1:
            raise ValueError('Third corner coordinate given despite\
                configuration being a strip.')
        return interpol_grid(coor1, coor2, coor3, m, n)



def interpol_grid(coor1, coor2, coor3, m, n):
    """ returns the coordinates of all interpolated electrodes in a
        grid of configuration m x n along with numeric labels in
        order.

    This is the worker function of interpol(...).

    @param coor1 : Coordinate corresponding to label 1
    @param coor2 : Coordinate corresponding to the first electrode in
        the last row
    @param coor3 : Coordinate corresponding to the last electrode
        (optional in case of strip)
    @param m : Number of rows
    @param n : Number of columns
    @type coor1: list
    @type coor2: list
    @type coor3: list
    @type m: int
    @type n: int
    @rtype dict

    Example:
        >> interpol([77, 170, 81],[65, 106, 112],[104, 121, 158],8,8)
    """
    # Turn input coordinates (which are presumably lists)
    # into numpy arrays.
    coor1 = np.asarray(coor1)
    coor2 = np.asarray(coor2)
    coor3 = np.asarray(coor3)

    # Figure out points A, B, and C of grid and respective vectors
    # (A2B and B2C) that define grid.
    vec1 = np.subtract(coor1, coor2)
    vec2 = np.subtract(coor2, coor3)
    vec3 = np.subtract(coor1, coor3)
    mag_1 = np.linalg.norm(vec1)
    mag_2 = np.linalg.norm(vec2)
    mag_3 = np.linalg.norm(vec3)

    # Reorient the vectors appropriately
    if (mag_1 >= mag_2) and (mag_1 >= mag_3):
        A = coor1
        B = coor3
        C = coor2
        A2B = -1 * vec3
        B2C = vec2
    if (mag_2 >= mag_1) and (mag_2 >= mag_3):
        A = coor2
        B = coor1
        C = coor3
        A2B = vec1
        B2C = -1 * vec3
    if (mag_3 >= mag_1) and (mag_3 >= mag_2):
        A = coor1
        B = coor2
        C = coor3
        A2B = -1 * vec1
        B2C = -1 * vec2

    # Compute unit vectors
    unit_A2B = normalize(A2B)
    unit_B2C = normalize(B2C)
    mag_A2B = np.linalg.norm(A2B)
    mag_B2C = np.linalg.norm(B2C)

    # Initialize outputs
    names = []
    elec_coor = []
    new_corr = A
    elec_coor.append(totuple(new_corr))
    count = 1
    names.append("GRID %d" % count)

    # Use for loops to deduce locations of electrodes.
    for j in range(n):
        for i in range(m - 1):
            new_corr = np.add(new_corr, (mag_A2B / (m - 1)) * (unit_A2B))
            elec_coor.append(totuple(new_corr))
            count += 1
            names.append("GRID %d" % count)
        new_corr = np.add(A, (mag_B2C / (n - 1)) * (j + 1) * (unit_B2C))
        elec_coor.append(totuple(new_corr))
        count += 1
        names.append("GRID %d" % count)
    names = names[0:-1]
    elec_coor = elec_coor[0:-1]
    # pairs = dict(zip(names, elec_coor))
    return elec_coor


def interpol_strip(coor1, coor2, m, n):
    """ returns the coordinates of all interpolated electrodes in a
        grid of configuration m x n along with numeric labels in
        order.

    This is the worker function of interpol(...).

    @param coor1 : Coordinate corresponding to label 1
    @param coor2 : Coordinate corresponding to the last electrode in
        strip
    @param m : Number of rows (could be 1)
    @param n : Number of columns (could be 1)
    @type coor1: list
    @type coor2: list
    @type m: int
    @type n: int
    @rtype dict

    Example:
        >> interpol([77, 170, 81],[65, 106, 112],6,1)
    """

    # Turn input coordinates (which are presumably lists) into numpy arrays.
    A = np.asarray(coor1)
    B = np.asarray(coor2)
    A2B = np.subtract(coor2, coor1)

    # Compute unit vectors
    unit_A2B = normalize(A2B)
    mag_A2B = np.linalg.norm(A2B)

    # Use for loops to deduce locations of electrodes.
    names = []
    elec_coor = []
    new_corr = A
    elec_coor.append(totuple(new_corr))
    count = 1
    names.append("GRID %d" % count)
    for i in range(m):
        new_corr = np.add(A, (mag_A2B / (m - 1)) * (i + 1) * (unit_A2B))
        elec_coor.append(totuple(new_corr))
        count += 1
        names.append("GRID %d" % count)
    names = names[0:-1]
    elec_coor = elec_coor[0:-1]
    # pairs = dict(zip(names, elec_coor))
    return elec_coor
