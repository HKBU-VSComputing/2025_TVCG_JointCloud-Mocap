from itertools import combinations

import numpy as np

from camera.triangulation import triangular_ray_cast_nx3, triangular_solve
from camera.utils import util_lines2lines_dist, util_norm
from core.skel_utils import convert_COCO17_to_BODY25_detection

# for openpose
EXTERN_mid_hip_th_list = [0.85, 0.7, 0.7, 0.8, 0.8, 0.7, 0.7, 0.8, 0.3,  # 0~8
                          0.3, 0.5, 1., 0.3, 0.5, 1.,  # 9~14
                          0.9, 0.9, 0.85, 0.85,  # 15~18
                          1., 1., 1., 1., 1., 1.]  # 19~24


def normalize(arr, t_min=0., t_max=1.):
    norm_arr = []
    diff = t_max - t_min
    diff_arr = max(arr) - min(arr) + np.finfo(float).eps
    for i in arr:
        temp = (((i - min(arr)) * diff) / diff_arr) + t_min
        norm_arr.append(temp)
    return norm_arr


def giz_rearrange_joint_v2(opps_data, joint_size=25, joint_prob=0.,
                           max_body=4, min_joint_size=5):
    """
    :return: view, joint2d, joint candidate, [XY, prob]
    """
    frame_data = []
    for viewD in opps_data:
        view_data = [[] for _ in range(joint_size)]
        for skel_dict in viewD:
            skel_data = skel_dict['meta']
            skel_type = len(skel_data)
            if skel_type == 17:
                skel_data = convert_COCO17_to_BODY25_detection(skel_data)
            elif skel_type == 25:
                pass
            else:
                assert False, "Unknown skel type: {}".format(skel_type)
            for jIdx, joint in enumerate(skel_data):
                prob = joint[2]
                if prob > joint_prob:
                    view_data[jIdx].append(joint)  # [X, Y, prob]
        frame_data.append(view_data)
    return frame_data


def giz_filter_via_ray_cast(frame_data, camera_RtKi, camera_pos, camera_res, joint_size=25, dist_filter_th=0.03,
                            view_num=5):
    """
    :param frame_data: view, joint2d, joint candidate, [XY, prob]
    :return: joint3d, joint candidate, [[vIdx_a, [XY, prob]],
                                        [vIdx_b, [XY, prob]]]
    """
    valid_joint_pair_list = [[] for _ in range(joint_size)]
    for view_a_idx, view_b_idx in combinations([i for i in range(view_num)], 2):
        for joint_idx, (joint_a_set, joint_b_set) in enumerate(zip(frame_data[view_a_idx], frame_data[view_b_idx])):
            # check valid joint set
            if len(joint_a_set) == 0 or len(joint_b_set) == 0:
                continue
            # for each joint in two views
            joint_a_set = np.array(joint_a_set)
            joint_b_set = np.array(joint_b_set)
            joint_a_ray = triangular_ray_cast_nx3(joint_a_set, camera_RtKi[view_a_idx], camera_res)
            joint_b_ray = triangular_ray_cast_nx3(joint_b_set, camera_RtKi[view_b_idx], camera_res)
            # dist = [ray_a, ray_b]
            dist = util_lines2lines_dist(camera_pos[view_a_idx], joint_a_ray,
                                         camera_pos[view_b_idx], joint_b_ray)
            # filter the outlier
            for dist_idx_a in range(dist.shape[0]):
                for dist_idx_b in range(dist.shape[1]):
                    if dist[dist_idx_a, dist_idx_b] < dist_filter_th:
                        valid_joint_pair_list[joint_idx].append([view_a_idx, joint_a_set[dist_idx_a],
                                                                 view_b_idx, joint_b_set[dist_idx_b]])
    return valid_joint_pair_list


def giz_triangulate_for_valid_joint(valid_joint_pair_list, camera_proj, camera_res, joint_size=25, logger=None):
    """
    :return: joint3d, joint candidate, XYZ (3, 1)
    :return: joint3d, joint candidate, [[vIdx_a, [XY, prob]],
                                        [vIdx_b, [XY, prob]]]
    """
    if logger is not None:
        print = logger.info
    valid_joint_3d_list = [[] for _ in range(joint_size)]
    # to store 2d joint pair that can be triangulated to the 3d joint
    valid_joint_3d_list_aux = [[] for _ in range(25)]
    valid_point_num = 0
    for joint_idx, joint in enumerate(valid_joint_pair_list):
        for joint_pair in joint:
            view_a_idx, joint_a, view_b_idx, joint_b = joint_pair
            joint_pair_data = np.stack((joint_a, joint_b))  # X, Y, prob
            proj_pair = np.stack((camera_proj[view_a_idx], camera_proj[view_b_idx]))
            joint_pair_data[:, 0] *= (camera_res[0] - 1)
            joint_pair_data[:, 1] *= (camera_res[1] - 1)
            convergent, joint_converged, loss = triangular_solve(point=joint_pair_data, proj_mat=proj_pair,
                                                                 view_num=len(proj_pair))
            if convergent:
                valid_joint_3d_list[joint_idx].append(joint_converged)
                valid_joint_3d_list_aux[joint_idx].append(joint_pair)
                valid_point_num += 1
    print("=> Total valid 3D points: {}".format(valid_point_num))
    return valid_joint_3d_list, valid_joint_3d_list_aux


def get_camera_id_from_joint3d_list(joint_3d_list_aux):
    cameraId_list = []
    for jC in joint_3d_list_aux:
        jC_cId = []
        for joint in jC:
            jC_cId.append(np.array([joint[0], joint[2]], dtype=int))
        cameraId_list.append(np.array(jC_cId))
    return cameraId_list


def cluster_joint_cloud(joint_3d_list, joint_3d_list_aux,
                        cur_mid_hip=None, joint_size=25,
                        logger=None):
    """
    :param: cur_mid_hip, [body, (3, 1)]
    :return: body-wise joint cloud
    :return: body-wise camera ID
    """
    if logger is not None:
        print = logger.info
    # track by the given centroid and divide into each candidate body
    # NOTE: the mid-hip is assigned with ID (tracking)
    cur_cand_num = len(cur_mid_hip)
    cand_3d_list = [[[] for _ in range(joint_size)] for _ in range(cur_cand_num)]
    cand_3d_list_aux = [[[] for _ in range(joint_size)] for _ in range(cur_cand_num)]
    for joint_cls, joint_list in enumerate(joint_3d_list):
        if not len(joint_list):
            continue
        joint_list = np.array(joint_list)
        jc_num = joint_list.shape[0]
        # store current joint class's mid-hip distance
        mid_hip_filter_dist = np.zeros((jc_num, cur_cand_num))
        for mid_hip_idx, mid_hip in enumerate(cur_mid_hip):
            if len(mid_hip) == 0:
                mid_hip_filter_dist[:, mid_hip_idx] = 10.
                continue
            dist = util_norm(mid_hip - joint_list, axis=1)
            mid_hip_filter_dist[:, mid_hip_idx] = dist[:, 0]
        # assign joint candidate to each candidate body
        cand_id_list = np.where(mid_hip_filter_dist < EXTERN_mid_hip_th_list[joint_cls])
        for cand_jId, cand_bId in zip(cand_id_list[0], cand_id_list[1]):
            cand_3d_list[cand_bId][joint_cls].append(joint_3d_list[joint_cls][cand_jId])
            cand_3d_list_aux[cand_bId][joint_cls].append(joint_3d_list_aux[joint_cls][cand_jId])
    for cand_bId, cand_body in enumerate(cand_3d_list_aux):
        jc_num = sum([len(jc) for jc in cand_body])
        print("\tBody {} point num: {}".format(cand_bId, jc_num))
        cand_3d_list_aux[cand_bId] = get_camera_id_from_joint3d_list(cand_body)
    return cand_3d_list, cand_3d_list_aux
