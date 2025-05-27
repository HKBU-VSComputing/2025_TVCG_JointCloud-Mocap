import numpy as np


def util_norm(point: np.array, axis=None):
    return np.linalg.norm(point, axis=axis)


def util_normalize(point: np.array, axis=None):
    return point / util_norm(point, axis)


def util_point2line_dist(point_a: np.array, point_b: np.array, line: np.array):
    return util_norm(np.cross((point_a - point_b).T, line))


def util_line2line_dist(point_a: np.array, line_a: np.array, point_b: np.array, line_b: np.array):
    if np.abs(np.dot(line_a, line_b)) < 1e-5:
        return util_point2line_dist(point_a, point_b, line_a)
    else:
        return np.abs(np.dot((point_a - point_b).T, util_normalize(np.cross(line_a, line_b))))


def util_lines2lines_dist(point_a: np.array, line_a: np.array, point_b: np.array, line_b: np.array):
    line_a_num = line_a.shape[1]
    line_b_num = line_b.shape[1]
    dist = np.zeros((line_a_num, line_b_num))
    for l_a_idx, l_a in enumerate(line_a.T):
        for l_b_idx, l_b in enumerate(line_b.T):
            dist[l_a_idx, l_b_idx] = util_line2line_dist(point_a, l_a, point_b, l_b)
    return dist


def util_hungarian_search(src_mat: np.array, maximize=False):
    from scipy.optimize import linear_sum_assignment

    if np.all(src_mat <= 0):
        return
    row_ind, col_ind = linear_sum_assignment(src_mat, maximize=maximize)
    return row_ind, col_ind
