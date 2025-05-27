import argparse
import logging
import os
import time
from datetime import datetime as timer
from pathlib import Path

import cv2
import numpy as np

import data.gizmo as gizmo
from camera.triangulation import triangular_parse_camera
from core.skel_def import skel_def_dict, skel_col_dict
from core.skel_painter import draw_skel_for_each_view, draw_points_for_each_view2

shelf_train_config = {'output_root': "./data/prepare/shelf_train",
                      'data_root': './data/prepare/shelf_data',
                      'camera_param': "./data/prepare/shelf_data/camera_params.npy",
                      'view_num': 5,
                      'file_root': "{:04d}",
                      'gt_path': "./data/prepare/shelf_data/skel_with_gt_in_skel19.npy",
                      'gt_range': [300, 601],
                      #####
                      'img_root': "./data/prepare/shelf_data/Shelf/Evaluate",
                      'view_list': ["img_0", "img_1", "img_2", "img_3", "img_4"],
                      'img_in_suffix': ".png",
                      'img_out_suffix': ".jpg",
                      #####
                      's1_joint_prob': 0.2,
                      's2_dist_filter_th': 0.1,
                      ####
                      'vis_flg': True,
                      'skel_type': "SKEL19",
                      'name': 'shelf_test',
                      'img_ratio': 0.6}

shelf_test_config = {'output_root': "./data/prepare/shelf_test",
                     'data_root': './data/prepare/shelf_data',
                     'camera_param': "./data/prepare/shelf_data/camera_params.npy",
                     'view_num': 5,
                     'file_root': "{:04d}",
                     'gt_path': "./data/prepare/shelf_data/skel_with_gt_in_skel19.npy",
                     'gt_range': [300, 601],
                     #####
                     'img_root': "./data/prepare/shelf_data/Shelf/Evaluate",
                     'view_list': ["img_0", "img_1", "img_2", "img_3", "img_4"],
                     'img_in_suffix': ".png",
                     'img_out_suffix': ".jpg",
                     #####
                     's1_joint_prob': 0.2,
                     's2_dist_filter_th': 0.1,
                     ####
                     'vis_flg': True,
                     'skel_type': "SKEL19",
                     'name': 'shelf_test',
                     'img_ratio': 0.6}


def prepare_data_path(config):
    """
    NOTE: use img to retrieval npy
    :return gt_data: [frame, body, 3, 1]
    """
    # prepare gt mid-hip
    gt_data = np.load(config.gt_path, allow_pickle=True)
    gt_len = len(gt_data)
    gt_hip = []
    hip_idx = skel_def_dict[config.skel_type]['root']
    for frame in gt_data:
        frame_hip = []
        for body in frame:
            if len(body) == 0:
                continue
            frame_hip.append(body[hip_idx][..., np.newaxis])
        gt_hip.append(frame_hip)
    # prepare path
    imgs_list = []
    data_list = []
    for view in config.view_list:
        view_imgs = []
        view_data = []
        view_path = os.path.join(config.img_root, view)
        item_count = 0
        for img_path in Path(view_path).rglob(f"*{config.img_in_suffix}"):
            npy_path = Path(config.data_root) / f'{view}/{img_path.stem}.npy'
            assert img_path.exists(), f'not found img_path: {img_path}'
            assert npy_path.exists(), f'not found npy_path: {npy_path}'
            view_imgs.append(str(img_path))
            view_data.append(str(npy_path))
            item_count += 1
        if item_count > gt_len:
            print("=> Crop for {}".format(gt_len))
            break
        imgs_list.append(view_imgs)
        data_list.append(view_data)
    # auto crop for ground truth
    if item_count < gt_len:
        crop_bgn, crop_end = config.gt_range
        gt_data = gt_data[crop_bgn:crop_end]
        gt_hip = gt_hip[crop_bgn:crop_end]
    imgs_list = np.array(imgs_list).transpose(1, 0).tolist()
    data_list = np.array(data_list).transpose(1, 0).tolist()
    assert len(gt_data) == len(imgs_list)
    assert len(gt_data) == len(data_list)
    return data_list, imgs_list, gt_data, gt_hip


def visualization(jointcloud_list, projMat, img_path, gt, mid_hip, out_path, sType, ratio=0.3, vis_gt=False):
    # joint cloud visualize
    vis_point_data = np.empty((0, 3))
    vis_point_color = np.empty((0, 3))
    for bId, body in enumerate(jointcloud_list):
        for jcID, jc in enumerate(body):
            jc_num = len(jc)
            if jc_num == 0:
                continue
            jc = np.array(jc).squeeze(-1)
            vis_point_data = np.vstack((vis_point_data, jc))
            vis_point_color = np.vstack((vis_point_color, np.tile(skel_col_dict[jcID], (jc_num, 1)) - bId * 10))
    view_set = draw_points_for_each_view2(vis_point_data, projMat, img_path,
                                          color_set=vis_point_color)
    # external gt skel visualize
    if vis_gt:
        # clean
        gt_ = []
        for gt_item in gt:
            if len(gt_item):
                gt_.append(gt_item)
        gt_ = np.array(gt_)
        draw_skel_for_each_view(gt_, projMat, img_path, type=sType, view_set=view_set)
    # external mid-hip visualize
    gt_mid_hip_tmp = []
    for gt_cand in mid_hip:
        if len(gt_cand) == 0:
            continue
        gt_mid_hip_tmp.append(gt_cand)
    if len(gt_mid_hip_tmp):
        gt_mid_hip_tmp = np.array(gt_mid_hip_tmp).squeeze(-1)
        view_set = draw_points_for_each_view2(gt_mid_hip_tmp, projMat, img_path, view_set=view_set,
                                              thickness=10)
    canvas_out = np.hstack(view_set)
    canvas_out = cv2.resize(canvas_out, None, fx=ratio, fy=ratio)
    # cv2.imshow("test", canvas_out)
    # cv2.waitKey(-1)
    cv2.imwrite(out_path, canvas_out)


def prepare_raw_data(config):
    cfg = argparse.Namespace(**config)
    #####
    # [New Feature] crawler the folder to prepare the data path
    opps_data_path_lst, opps_frame_path_lst, opps_gt_data, opps_gt_hip = prepare_data_path(cfg)

    # open logger from print
    Path(cfg.output_root).mkdir(parents=True, exist_ok=True)
    cur_date = timer.now().strftime("%y%m%d")
    logger_path = os.path.join(cfg.output_root, 'shelf_tri_data_{}.log'.format(cur_date))
    logger.addHandler(logging.FileHandler(logger_path, 'a'))
    print = logger.info
    # decide the output path
    cur_name = os.path.basename(__file__).split(".")[0]
    output_path = os.path.join(cfg.output_root, f'shelf_{cur_name}_{cur_date}')
    if cfg.vis_flg:
        Path(output_path).mkdir(parents=True, exist_ok=True)
    print("\n\n================\n=> I'm {}\n=> stored in {}*\n================\n".format(logger_path, output_path))

    # process
    _, camera_res, camera_proj, camera_RtKi, camera_pos = triangular_parse_camera(cfg.camera_param)
    print("Enter data root folder => {}...".format(cfg.data_root))
    scene_data = []
    dur = 0.
    for fIdx, (fDataPath, fImgsPath, fGtData, fHipData) in enumerate(zip(opps_data_path_lst,
                                                                         opps_frame_path_lst,
                                                                         opps_gt_data,
                                                                         opps_gt_hip)):
        fGtData = opps_gt_data[fIdx].tolist()
        fHipData = opps_gt_hip[fIdx]
        print('================\nFrame {}...'.format(fIdx))
        opps_frame_data = [np.load(item, allow_pickle=True).tolist() for item in fDataPath]
        ##########################################################################
        # 1. rearrange the joint
        giz_frame_data = gizmo.giz_rearrange_joint_v2(opps_frame_data, joint_prob=cfg.s1_joint_prob)
        ##########################################################################
        start = time.time()
        # 2. for view a and view b, calc each joint's ray cast distance
        giz_joint2d_pair_list = gizmo.giz_filter_via_ray_cast(giz_frame_data, camera_RtKi, camera_pos, camera_res,
                                                              dist_filter_th=cfg.s2_dist_filter_th,
                                                              view_num=cfg.view_num)
        ##########################################################################
        # 3. for each valid joint, triangulate
        # giz_joint3d_list: 3d candidate joint list
        # giz_joint3d_aux: to store 2d joint pair that can be triangulated to the 3d joint
        giz_joint3d_list, giz_joint3d_aux = gizmo.giz_triangulate_for_valid_joint(giz_joint2d_pair_list,
                                                                                  camera_proj, camera_res,
                                                                                  logger=logger)
        ##########################################################################
        # 4. cluster via body center and return corresponding camera id pair
        # get hit map
        giz_jointcloud_list, giz_jointcloud_camera = gizmo.cluster_joint_cloud(giz_joint3d_list, giz_joint3d_aux,
                                                                               cur_mid_hip=fHipData,
                                                                               logger=logger)
        ##########################################################################
        dur += time.time() - start
        # Store
        scene_data.append((giz_jointcloud_list, giz_jointcloud_camera, fHipData))
        ##########################################################################
        # visualize
        if cfg.vis_flg:
            snapshot_path = os.path.join(output_path, "{:04d}{}".format(fIdx, cfg.img_out_suffix))
            visualization(giz_jointcloud_list, camera_proj, fImgsPath, fGtData, fHipData, snapshot_path,
                          cfg.skel_type, cfg.img_ratio, vis_gt=True)
        ##########################################################################
        # store
        np.save(output_path + '.npy', scene_data)

    print("avg {}".format(dur / len(opps_data_path_lst)))


if __name__ == '__main__':
    # logger here
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logger = logging.getLogger()

    ##################################################################
    prepare_raw_data(shelf_train_config)
    prepare_raw_data(shelf_test_config)

    print("Done")
