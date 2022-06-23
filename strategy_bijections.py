import numpy as np


def one_simplex_coord_to_point(coord, min_x, max_x):
    """Compute the map of the unit one-simplex coordinate to a point in the correct line segment.

    Args:
        coord (ndarray): A coordinate in a unit one-simplex.
        min_x (float): The minimum value in the interval.
        max_x (float): The maximum value in the interval.

    Returns:
        float: A point in the interval.
    """
    return min_x + coord[0] * (max_x - min_x)


def one_simplex_point_to_coord(point, min_x, max_x):
    """Compute the map of a point in an interval to a unit one-simplex coordinate.

    Args:
        point (float): A point in the interval.
        min_x (float): The minimum value in the interval.
        max_x (float): The maximum value in the interval.

    Returns:
        ndarray: A coordinate in a unit one-simplex.
    """
    x_coord = (point - min_x) / (max_x - min_x)
    y_coord = 1 - x_coord
    coord = np.array([x_coord, y_coord])
    return coord
