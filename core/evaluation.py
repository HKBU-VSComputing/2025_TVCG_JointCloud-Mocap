import argparse

import numpy as np

from camera.utils import util_norm
from core.skel_utils import skel_def_dict


def util_re_id(frame_src, frame_gt, th=0.2):
    # return which skeletons in frame_src belongs to frame_gt
    # aka, from src to gt
    frame_src_id_map = np.ones(len(frame_src), dtype=np.int64) * -1
    for skel_src_id, skel_src in enumerate(frame_src):
        if not len(skel_src):
            continue
        dist_list = []
        for skel_gt in frame_gt:
            if skel_gt is None or not len(skel_gt):
                dist_list.append(1000.)
                continue
            dist_list.append(np.average(np.sqrt(np.sum(np.power(skel_src - skel_gt, 2), axis=1))))
        min_dist = min(dist_list)
        if min_dist < th:
            min_idx = dist_list.index(min_dist)  # find the closest gt id
            frame_src_id_map[skel_src_id] = min_idx
    return frame_src_id_map


def util_calc_pcp(skel3d_src, skel3d_gt, limb_map):
    assert len(skel3d_src) == len(skel3d_gt), 'Input skeleton should have the same dim. {}-{}'.format(len(skel3d_src),
                                                                                                      len(skel3d_gt))
    pcp_mat = np.zeros((len(limb_map[0])))
    for limbIdx in range(len(limb_map[0])):
        jA_src = skel3d_src[limb_map[0][limbIdx]]
        jB_src = skel3d_src[limb_map[1][limbIdx]]
        # valid limb filter
        if np.all(jA_src == 0.) or np.all(jB_src == 0.):
            continue
        jA_gt = skel3d_gt[limb_map[0][limbIdx]]
        jB_gt = skel3d_gt[limb_map[1][limbIdx]]
        dist_a = util_norm(jA_src - jA_gt)
        dist_b = util_norm(jB_src - jB_gt)
        l = util_norm(jA_gt - jB_gt)
        if dist_a + dist_b < l:
            pcp_mat[limbIdx] = 1
    return pcp_mat


def util_calc_pck(skel3d_src, skel3d_gt, th=0.2):
    assert th <= 1., 'the threshold must be less than 1m. the unit is mm.'
    assert len(skel3d_src) == len(skel3d_gt), 'Input skeleton should have the same dim.'
    joint_num = len(skel3d_gt)
    correct_joint = np.zeros(joint_num)
    detected_joint = np.zeros(joint_num)
    for jIdx, (j_src, j_gt) in enumerate(zip(skel3d_src, skel3d_gt)):
        # valid joint filter
        if np.all(j_src == 0.):
            # not detected
            continue
        dist = util_norm(j_src - j_gt)
        if dist < th:
            correct_joint[jIdx] = 1
        if np.sum(np.abs(j_src)) > 0:
            detected_joint[jIdx] = 1
    return correct_joint, detected_joint


def util_calc_pck_v2(skel3d_src, skel3d_gt, th=0.2):
    assert th <= 1., 'the threshold must be less than 1m. the unit is mm.'
    assert len(skel3d_src) == len(skel3d_gt), 'Input skeleton should have the same dim.'
    # valid joint filter
    valid_map = ~np.all(skel3d_src == 0., axis=1)
    detected_joint = np.sum(valid_map)
    # assert detected_joint == 15
    valid_joint_dist = util_norm(skel3d_src[valid_map] - skel3d_gt[valid_map], axis=-1)
    correct_joint = np.sum(valid_joint_dist < th)
    gt_joint = len(skel3d_gt)
    return correct_joint, detected_joint, gt_joint


def util_calc_mpjpe(skel3d_src, skel3d_gt):
    assert len(skel3d_src) == len(skel3d_gt), 'Input skeleton should have the same dim.'
    # valid joint filter
    valid_map = ~np.all(skel3d_src == 0., axis=1)
    return util_norm(skel3d_src[valid_map] - skel3d_gt[valid_map], axis=-1).mean()


def util_evaluate_driver(frame_src, frame_gt, frame_src_id_map, pck_th=0.2, mode='SHELF14'):
    # assert len(frame_gt) == 4, "GT dim error! {}".format(len(frame_gt))
    pcp_score = [[], [], [], []]
    pck_score = [[], [], [], []]
    mpjpe_score = [[], [], [], []]
    joint_num = 0
    log_id = []
    # choose skeleton template
    PCP_limb_map = skel_def_dict[mode]['eval_map']
    for skel_src, skel_gt_id in zip(frame_src, frame_src_id_map):
        skel_gt_id = int(skel_gt_id)
        # gt_id = -1, means mis-match
        if skel_gt_id == -1:
            continue
        skel_gt = frame_gt[skel_gt_id]
        joint_num = len(skel_gt)
        # make sure gt is not empty
        if not len(skel_gt):
            continue
        log_id.append(skel_gt_id)
        # make sure src and gt are the same data structure
        assert skel_gt.shape == skel_src.shape
        # skel_src = skel_src[:, :, np.newaxis]
        # skel_gt = skel_gt[:, :, np.newaxis]
        # calc
        pcp_score[skel_gt_id].append(util_calc_pcp(skel_src, skel_gt, PCP_limb_map))
        pck_score[skel_gt_id].append(util_calc_pck(skel_src, skel_gt, pck_th))
        mpjpe_score[skel_gt_id].append(util_calc_mpjpe(skel_src, skel_gt))
    # check unpredictable result
    for gId, gt in enumerate(frame_gt):
        if gId not in log_id and len(gt):
            pcp_score[gId].append(np.zeros(len(PCP_limb_map[0])))
            tmp = np.zeros(joint_num)
            pck_score[gId].append((tmp, tmp))
            mpjpe_score[gId].append(0)
    return {
        "pcp": pcp_score,
        "pck": pck_score,
        "pck@th": pck_th,
        "mpjpe": mpjpe_score,
        "joint_num": joint_num
    }


def util_evaluate_summary(eval_metric, txt_dir, enable_log=True, mode='SHELF14'):
    import logging, os
    if enable_log:
        logging.basicConfig(level=logging.INFO, format='%(message)s')
        logger = logging.getLogger()
        logger.addHandler(logging.FileHandler(os.path.join(txt_dir, 'eval.log'), 'a'))
        printf = logger.info
    else:
        printf = print

    eps = np.finfo(float).eps
    eval_frame_num = len(eval_metric)
    joint_num = max([frame['joint_num'] for frame in eval_metric])
    body_num = []
    for frame in eval_metric:
        bNum = 0
        for item in frame['mpjpe']:
            if len(item):
                bNum += 1
        body_num.append(bNum)
    body_num = max(body_num)
    printf("=== Evaluation summary ===")
    printf("- Frame num: {}".format(eval_frame_num))
    printf("- Skel num: {}".format(body_num))
    printf("- Joint num: {} (include mid-hip)".format(joint_num))

    PCP_metric = np.zeros((4, len(skel_def_dict[mode]['eval_name'])))
    PCP_valid_frame = np.ones((4, 1)) * eps
    PCK_r_metric = np.ones((4, joint_num)) * eps
    PCK_p_metric = np.ones((4, joint_num)) * eps
    PCK_valid_frame = np.ones((4, 1)) * eps
    MPJPE_metric = np.zeros(4)
    MPJPE_valid_frame = np.ones((4, 1)) * eps
    for frame in eval_metric:
        # PCP
        for pIdx, pcp_list in enumerate(frame["pcp"]):
            if len(pcp_list):
                PCP_valid_frame[pIdx] += 1
                pcp_list = pcp_list[0]
                for jIdx, pcp_j in enumerate(pcp_list):
                    PCP_metric[pIdx, jIdx] += pcp_j
        # PCK
        for pIdx, pck_list in enumerate(frame["pck"]):
            if len(pck_list):
                PCK_valid_frame[pIdx] += 1
                pck_list = pck_list[0]
                pck_r_list = pck_list[0]
                pck_p_list = pck_list[1]
                for jIdx, (pck_r, pck_p) in enumerate(zip(pck_r_list, pck_p_list)):
                    PCK_r_metric[pIdx, jIdx] += pck_r
                    PCK_p_metric[pIdx, jIdx] += pck_p
        # MPJPE
        for pIdx, mpjpe_list in enumerate(frame["mpjpe"]):
            if len(mpjpe_list):
                MPJPE_metric[pIdx] += mpjpe_list[0]
                MPJPE_valid_frame[pIdx] += 1

    # average
    PCP_avg = PCP_metric / PCP_valid_frame
    PCP_acg3 = []
    precision_avg = PCK_r_metric / PCK_p_metric
    recall_avg = PCK_r_metric / PCK_valid_frame
    MPJPE_avg = MPJPE_metric / MPJPE_valid_frame.squeeze(axis=1)
    printf("=== PCP ===")
    for pIdx, person in enumerate(PCP_metric):
        printf("\tP {} ===".format(pIdx))
        for lIdx, limb_score in enumerate(person):
            limb_name = skel_def_dict[mode]['eval_name'][lIdx]
            printf("{}: {}/{}, {}".format(limb_name, int(limb_score), int(PCP_valid_frame[pIdx]), PCP_avg[pIdx, lIdx]))
        PCP_acg3.append(np.average(PCP_avg[pIdx]))
        printf("\tAverage: {}".format(np.average(PCP_avg[pIdx])))
    printf("=== PCP AVG P1-3 ===")
    printf("\tAverage: {}".format(np.average(PCP_acg3[:3])))
    printf("=== PCK@{} ===".format(eval_metric[0]["pck@th"]))
    for pIdx, (person_p, person_r) in enumerate(zip(precision_avg, recall_avg)):
        printf("\tP {} ===".format(pIdx))
        printf("\tAverage Precision: {}".format(np.average(person_p)))
        printf("\tAverage Recall: {}".format(np.average(person_r)))
    printf(MPJPE_avg)
    return np.average(PCP_acg3[:3])


def util_evaluator(src_data, gt_data, config=None, enable_log=False, txt_dir=None, txt_path=None):
    import logging, os
    from prettytable import PrettyTable
    if enable_log:
        logging.basicConfig(level=logging.INFO, format='%(message)s')
        logger = logging.getLogger()
        if txt_path is not None:
            logger.addHandler(logging.FileHandler(txt_path, 'a'))
        elif txt_dir is not None:
            logger.addHandler(logging.FileHandler(os.path.join(txt_dir, 'eval.log'), 'a'))
        else:
            assert False, "txt path or txt dir is empty"
        printf = logger.info
    else:
        printf = print
    assert len(src_data) == len(gt_data)
    # init
    if config is None:
        config = {'reid_th': 0.2,
                  'gt_mode': 'SHELF14',
                  'pck_th': 0.2}
        config = argparse.Namespace(**config)
    PCP_limb_map = skel_def_dict[config.gt_mode]['eval_map']
    #################################################################
    # record available gt
    body_valid = np.array([[len(b) for b in f] for f in gt_data]) > 0
    gt_body_num = np.max(np.where(body_valid)[1]) + 1
    assert gt_body_num == max([len(f) for f in gt_data]), "id is discontinuous in gt data"
    # gt_body_num = 4
    #################################################################
    gt_frame_num = len(gt_data)
    body_mpjpe = np.zeros(gt_body_num)
    body_limb_count = np.zeros((gt_body_num, len(PCP_limb_map[0])))
    body_pck = np.zeros((gt_body_num, 3))
    # process
    for fIdx, (f_src, f_gt) in enumerate(zip(src_data, gt_data)):
        # re-id
        reid_map = util_re_id(f_src, f_gt, config.reid_th)
        # calc for every body in gt
        for src_id, gt_id in enumerate(reid_map):
            if gt_id == -1:
                # mis-match
                continue
            body_mpjpe[gt_id] += util_calc_mpjpe(f_src[src_id], f_gt[gt_id])
            body_limb_count[gt_id] += util_calc_pcp(f_src[src_id], f_gt[gt_id], PCP_limb_map)
            pck_score = util_calc_pck_v2(f_src[src_id], f_gt[gt_id], config.pck_th)
            body_pck[gt_id, 0] += pck_score[0]
            body_pck[gt_id, 1] += pck_score[1]
            body_pck[gt_id, 2] += pck_score[2]
    #################################################################
    valid_frame_num = np.sum(body_valid, axis=0)
    body_mpjpe /= valid_frame_num
    body_pcp = body_limb_count / valid_frame_num[:, None]
    body_pcp[body_pcp > 1.] = 1.
    body_pcp_avg = np.mean(body_pcp, axis=1)
    body_precs = body_pck[:, 0] / body_pck[:, 1]
    body_recal = body_pck[:, 0] / body_pck[:, 2]
    #################################################################
    printf("=== Evaluation summary ===")
    printf("- Frame num: {}".format(gt_frame_num))
    printf("- Skel num: {}".format(gt_body_num))
    printf("=== Details ===")
    for bIdx in range(gt_body_num):
        printf("\tP {}".format(bIdx))
        for lIdx, limb_score in enumerate(body_limb_count[bIdx]):
            limb_name = skel_def_dict[config.gt_mode]['eval_name'][lIdx]
            printf("{}: {}/{}, {}".format(limb_name, int(limb_score), int(valid_frame_num[bIdx]), body_pcp[bIdx, lIdx]))
        printf("===========")
    #################################################################
    body_pcp_avg *= 100.
    body_precs *= 100.
    body_recal *= 100.
    body_mpjpe *= 1000.
    tb = PrettyTable(["Metric"] + ["P{}".format(bIdx) for bIdx in range(gt_body_num)] + ["AVG1-3"])
    tb.add_row(
        ["PCP"] + ['{:.1f}'.format(data) for data in body_pcp_avg] + ['{:.1f}'.format(np.mean(body_pcp_avg[:3]))])
    tb.add_row(["Preci"] + ['{:.1f}'.format(data) for data in body_precs] + ['{:.1f}'.format(np.mean(body_precs[:3]))])
    tb.add_row(["Recal"] + ['{:.1f}'.format(data) for data in body_recal] + ['{:.1f}'.format(np.mean(body_recal[:3]))])
    tb.add_row(["MPJPE"] + ['{:.1f}'.format(data) for data in body_mpjpe] + ['{:.1f}'.format(np.mean(body_mpjpe[:3]))])
    printf(tb)
    #################################################################
