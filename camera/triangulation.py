import copy

import numpy as np
import scipy.linalg
from scipy.spatial.transform import Rotation as RR

from camera.utils import util_norm, util_normalize


def triangular_parse_camera_with_noise(src_path, noisy_view=[0, 1, 2]):
    if 'pickle' in src_path:
        param = np.load(src_path, allow_pickle=True)
    else:
        param = np.load(src_path, allow_pickle=True).tolist()
    view_num = len(param)
    res = param[0]["res"]
    proj_mat = []
    RtKi_mat = []
    pos_mat = []
    for camIdx, cam in enumerate(param):
        proj = cam["P"]
        K = np.array(cam["K"])

        # make up
        if camIdx in noisy_view:
            R = np.array(cam["R"])
            T = np.array(cam["T"]).T.reshape(3, -1)
            # random
            T += np.random.randn(3, 1) * 0.025
            angle = np.random.uniform(0, np.pi / 72)
            axis = np.random.rand(3)
            axis /= np.linalg.norm(axis)
            rotation = RR.from_rotvec(angle * axis)
            rotation_matrix = rotation.as_matrix()
            R = R * rotation_matrix
            # recal
            RT = np.hstack((R, T))
            proj_2 = np.dot(K, RT)
            pos_2 = -np.dot(R.T, T)
            # # debug
            # proj_diff = np.sum(np.array(proj) - proj_2)
            # pos_diff = np.sum(cam["Pos"] - pos_2)
            # update
            cam["R"] = R.tolist()
            cam["T"] = T.tolist()
            proj = proj_2.tolist()
            cam["Pos"] = pos_2

        RtKi = np.dot(np.array(cam["R"]).T, np.linalg.inv(K))
        proj_mat.append(proj)
        RtKi_mat.append(RtKi)
        pos_mat.append(cam["Pos"])
    proj_mat = np.array(proj_mat)
    return view_num, res, proj_mat, RtKi_mat, pos_mat


def triangular_parse_camera(src_path):
    if 'pickle' in src_path:
        param = np.load(src_path, allow_pickle=True)
    else:
        param = np.load(src_path, allow_pickle=True).tolist()
    view_num = len(param)
    res = param[0]["res"]
    proj_mat = []
    RtKi_mat = []
    pos_mat = []
    for camIdx, cam in enumerate(param):
        proj = cam["P"]
        K = np.array(cam["K"])

        RtKi = np.dot(np.array(cam["R"]).T, np.linalg.inv(K))
        proj_mat.append(proj)
        RtKi_mat.append(RtKi)
        pos_mat.append(cam["Pos"])
    proj_mat = np.array(proj_mat)
    return view_num, res, proj_mat, RtKi_mat, pos_mat


def triangular_ray_cast(point, RtKi, res):
    # calc back to pixel coordinate
    point[0] *= (res[0] - 1)
    point[1] *= (res[1] - 1)
    point[2] = 1
    ray = np.dot(-RtKi, point)
    ray_norm = util_norm(ray)
    ray /= ray_norm
    return ray


def triangular_ray_cast_nx3(array: np.array, RtKi, res):
    array_cpy = copy.deepcopy(array)
    array_cpy[:, 0] *= (res[0] - 1)
    array_cpy[:, 1] *= (res[1] - 1)
    array_cpy[:, 2] = 1

    ray = np.dot(-RtKi, array_cpy.T)
    ray_normed = util_normalize(ray, axis=0)
    return ray_normed


def triangular_solve(max_iter_time=20, update_tolerance=0.0001, regular_term=0.0001, point=None, proj_mat=None,
                     view_num=5, filter_prob=0.2):
    point = point.T
    assert view_num == proj_mat.shape[0]
    pos = np.zeros((4, 1))
    pos[3] = 1
    convergent = False
    prob = point[2, :] > filter_prob

    loss = 100.
    for i in range(max_iter_time):
        ATA = np.identity(3) * regular_term
        ATb = np.zeros((3, 1))
        for view in range(view_num):
            # visible point
            if prob[view] > 0:
                current_proj = proj_mat[view]  # (3, 4)
                xyz = np.dot(current_proj, pos)
                ##
                jacobi = np.zeros((2, 3))
                jacobi[0, 0] = 1. / xyz[2]
                jacobi[0, 2] = -xyz[0] / pow(xyz[2], 2)
                jacobi[1, 1] = 1. / xyz[2]
                jacobi[1, 2] = -xyz[1] / pow(xyz[2], 2)
                ##
                jacobi = np.dot(jacobi, current_proj[:, :3])
                w = point[2, view]  # prob
                ATA += w * np.dot(jacobi.T, jacobi)
                out = (point[:2, view] - xyz[:2, 0] / xyz[2, 0]).reshape(-1, 1)
                ATb += w * np.dot(jacobi.T, out)
        delta = scipy.linalg.solve(ATA, ATb)
        loss = np.linalg.norm(delta)
        if loss < update_tolerance:
            convergent = True
            break
        else:
            pos[:3] += delta
    return convergent, pos[:3], loss
