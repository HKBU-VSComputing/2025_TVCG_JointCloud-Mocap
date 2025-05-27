import numpy as np

from camera.utils import util_norm
from core.skel_def import skel_def_dict


def convert_BODY25_to_SHELF14(src_data, face_y=0.125, face_z=0.145):
    if len(src_data.shape) == 3:
        src_data = src_data[0]
    assert src_data.shape == (25, 3)
    template = skel_def_dict['BODY25']['to_shelf14']

    dst_skel3d = np.zeros((15, 3))
    for src_jIdx, dst_jIdx in enumerate(template):
        if dst_jIdx == -1:
            continue
        dst_skel3d[dst_jIdx] = src_data[src_jIdx]

    # interpolate for HIP(14), TOP_HEAD(12), BOTTOM_HEAD(13)
    # HIP(14)
    # dst_skel3d[14] = (dst_skel3d[2] + dst_skel3d[3]) / 2.

    # get face direction
    face_dir = np.cross((dst_skel3d[12] - dst_skel3d[14]).T, (dst_skel3d[8] - dst_skel3d[9]).T)
    face_dir_normalized = face_dir / util_norm(face_dir)
    z_dir = np.zeros(3)
    z_dir[2] = 1.
    # calc TOP_HEAD(12) and BOTTOM_HEAD(13)
    src_ear_r = src_data[17]
    src_ear_l = src_data[18]
    head_center = (src_ear_r + src_ear_l) / 2.
    shoulder_center = (dst_skel3d[8] + dst_skel3d[9]) / 2.
    dst_skel3d[12] = shoulder_center + (head_center - shoulder_center) * 0.5
    dst_skel3d[13] = dst_skel3d[12] + face_dir_normalized * face_y + z_dir * face_z

    return dst_skel3d


def convert_SKEL19_to_SHELF14(src_data, face_y=0.125, face_z=0.145):
    if len(src_data.shape) == 3:
        src_data = src_data[0]
    assert src_data.shape == (19, 3)
    template = skel_def_dict['SKEL19']['to_shelf14']

    dst_skel3d = np.zeros((15, 3))
    for dst_jIdx, src_jIdx in enumerate(template):
        if src_jIdx == -1:
            continue
        dst_skel3d[dst_jIdx] = src_data[src_jIdx]

    # interpolate for HIP(14), TOP_HEAD(12), BOTTOM_HEAD(13)
    # HIP(14)
    # dst_skel3d[14] = (dst_skel3d[2] + dst_skel3d[3]) / 2.

    # get face direction
    face_dir = np.cross((dst_skel3d[12] - dst_skel3d[14]).T, (dst_skel3d[8] - dst_skel3d[9]).T)
    face_dir_normalized = face_dir / util_norm(face_dir)
    z_dir = np.zeros(3)
    z_dir[2] = 1.
    # calc TOP_HEAD(12) and BOTTOM_HEAD(13)
    src_ear_r = src_data[9]
    src_ear_l = src_data[10]
    head_center = (src_ear_r + src_ear_l) / 2.
    shoulder_center = (dst_skel3d[8] + dst_skel3d[9]) / 2.
    dst_skel3d[12] = shoulder_center + (head_center - shoulder_center) * 0.5
    dst_skel3d[13] = dst_skel3d[12] + face_dir_normalized * face_y + z_dir * face_z

    return dst_skel3d


def convert_BODY25_to_SKEL19(src_data):
    template = skel_def_dict['BODY25']['to_skel19']
    dst_data = np.zeros((max(template) + 1, 3))
    # joint mapping
    for jIdx, srcJ in enumerate(src_data):
        if template[jIdx] == -1:
            continue
        dst_data[template[jIdx]] = srcJ
    return dst_data


def convert_BODY25_to_SKEL17(src_data):
    # assert len(convert_map_BODY25_SKEL17) == 25
    # dst_data = np.zeros((max(convert_map_BODY25_SKEL17) + 1, src_data.shape[1]))
    # # joint mapping
    # for jIdx, srcJ in enumerate(src_data):
    #     if convert_map_BODY25_SKEL17[jIdx] == -1:
    #         continue
    #     dst_data[convert_map_BODY25_SKEL17[jIdx]] = srcJ
    # return dst_data
    return NotImplementedError


def convert_SKEL17_to_SHELF14(src_data, face_y=0.125, face_z=0.145):
    if len(src_data.shape) == 3:
        src_data = src_data[0]
    assert src_data.shape == (17, 3)
    template = skel_def_dict['SKEL17']['to_shelf14']

    dst_skel3d = np.zeros((15, 3))
    for dst_jIdx, src_jIdx in enumerate(template):
        if src_jIdx == -1:
            continue
        dst_skel3d[dst_jIdx] = src_data[src_jIdx]

    # interpolate for HIP(14), TOP_HEAD(12), BOTTOM_HEAD(13)
    # HIP(14)
    dst_skel3d[14] = (dst_skel3d[2] + dst_skel3d[3]) / 2.
    # mid-shoulder
    dst_skel3d[12] = (dst_skel3d[8] + dst_skel3d[9]) / 2.

    # get face direction
    face_dir = np.cross((dst_skel3d[12] - dst_skel3d[14]).T, (dst_skel3d[8] - dst_skel3d[9]).T)
    face_dir_normalized = face_dir / util_norm(face_dir)
    z_dir = np.zeros(3)
    z_dir[2] = 1.
    # calc TOP_HEAD(12) and BOTTOM_HEAD(13)
    src_ear_r = src_data[4]
    src_ear_l = src_data[3]
    head_center = (src_ear_r + src_ear_l) / 2.
    shoulder_center = (dst_skel3d[8] + dst_skel3d[9]) / 2.
    dst_skel3d[12] = shoulder_center + (head_center - shoulder_center) * 0.5
    dst_skel3d[13] = dst_skel3d[12] + face_dir_normalized * face_y + z_dir * face_z
    return dst_skel3d


def convert_SKEL17_to_SHELF14_v2(coco_pose):
    """
    Cited from mvpose\src\m_utils\transformation.py

    transform coco order(our method output) 3d pose to shelf dataset order with interpolation
    :param coco_pose: np.array with shape 3x17
    :return: 3D pose in shelf order with shape 14x3 # added mid-hip, so the shape is 15x3
    """
    coco_pose = coco_pose.T
    shelf_pose = np.zeros((15, 3))
    coco2shelf = np.array([16, 14, 12, 11, 13, 15, 10, 8, 6, 5, 7, 9])
    shelf_pose[0: 12] += coco_pose[coco2shelf]
    neck = (coco_pose[5] + coco_pose[6]) / 2  # L and R shoulder
    head_bottom = (neck + coco_pose[0]) / 2  # nose and head center
    head_center = (coco_pose[3] + coco_pose[4]) / 2  # middle of two ear
    # head_top = coco_pose[0] + (coco_pose[0] - head_bottom)
    head_top = head_bottom + (head_center - head_bottom) * 2
    # shelf_pose[12] += head_bottom
    # shelf_pose[13] += head_top
    shelf_pose[12] = (shelf_pose[8] + shelf_pose[9]) / 2  # Use middle of shoulder to init
    shelf_pose[13] = coco_pose[0]  # use nose to init
    shelf_pose[13] = shelf_pose[12] + (shelf_pose[13] - shelf_pose[12]) * np.array([0.75, 0.75, 1.5])
    shelf_pose[12] = shelf_pose[12] + (coco_pose[0] - shelf_pose[12]) * np.array([1. / 2., 1. / 2., 1. / 2.])
    # shelf_pose[13] = shelf_pose[12] + (shelf_pose[13] - shelf_pose[12]) * np.array ( [0.5, 0.5, 1.5] )
    # shelf_pose[12] = shelf_pose[12] + (shelf_pose[13] - shelf_pose[12]) * np.array ( [1.0 / 3, 1.0 / 3, 1.0 / 3] )
    # add mid-hip
    shelf_pose[-1] = (shelf_pose[2] + shelf_pose[3]) / 2.
    return shelf_pose


def convert_EMPTY(src_data):
    return src_data


def convert_COCO17_to_BODY25_detection(src_data):
    dst_data = np.zeros((25, 3))
    joint_map = [0, 16, 15, 18, 17, 5, 2, 6, 3, 7, 4, 12, 9, 13, 10, 14, 11]
    dst_data[joint_map] = src_data[:]
    dst_data[1] = (dst_data[2] + dst_data[5]) / 2.
    dst_data[8] = (dst_data[9] + dst_data[12]) / 2.
    return dst_data


class SkelConverter:
    def __init__(self, mode: str):
        mode = mode.lower()
        dType = None
        if '15' in mode or 'shelf' in mode:
            dType = 15
        elif '17' in mode:
            dType = 17
        elif '19' in mode:
            dType = 19
        elif '25' in mode:
            dType = 25
        else:
            assert False, "Mode {} is not supported".format(mode)
        self.dType = dType

    def __get_converter(self, sType, dType):
        if sType == dType:
            # no conversion
            return convert_EMPTY
        elif sType == 0 or dType == 0:
            # empty skeleton
            return convert_EMPTY
        else:
            assert False, "Unknown type {} -> {}".format(sType, dType)

    def __call__(self, gt_frame, pred_frame):
        """
        :param gt_frame: body, joint, XYZ
        :param pred_frame: body, (1), joint, XYZ
        :return: gt, pred under the pre-defined mode
        """
        # for gt
        gt_frame = np.array(gt_frame, dtype=object)
        gt_type = max([len(body) for body in gt_frame])
        gt_converter = self.__get_converter(gt_type, self.dType)
        gt_frame_ = [gt_converter(body) for body in gt_frame]
        # for pred
        if len(pred_frame) != 0:
            pred_frame = np.array(pred_frame)
            if len(pred_frame.shape) == 4 and pred_frame.shape[1] == 1:
                pred_frame = pred_frame.squeeze(1)
            pred_type = max([len(body) for body in pred_frame])
            pred_converter = self.__get_converter(pred_type, self.dType)
            pred_frame_ = [pred_converter(body) for body in pred_frame]
        else:
            pred_frame_ = []

        return gt_frame_, pred_frame_
