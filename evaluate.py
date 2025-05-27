import os
import cv2
import argparse
import numpy as np
import warnings
from pathlib import Path

# from core.config import load_config
from core.evaluation import util_re_id, util_evaluate_summary, util_evaluate_driver
from camera.triangulation import triangular_parse_camera
from core.skel_painter import draw_skel_for_each_view
from core.skel_utils import SkelConverter
from inference import prepare_imgs


def snapshot(dst_root, src_list, skel_data, camera_proj_mat):
    dst_root = dst_root + '/img_eval'
    file_name = src_list[0].split('/')[-1][:-4] + '.jpg'
    snapshot_dst_path = os.path.join(dst_root, file_name)
    Path(dst_root).mkdir(parents=True, exist_ok=True)

    canvas_out = draw_skel_for_each_view(skel_data, camera_proj_mat, src_list, type='SHELF14')
    out = np.hstack(canvas_out)
    snapshot_out = cv2.resize(out, None, fx=0.6, fy=0.6)
    cv2.imwrite(snapshot_dst_path, snapshot_out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Simple parser for TMAE Mocap framework")
    parser.add_argument('--snapshot', action='store_true', default=False, help='Store snapshot for visualization.')
    parser.add_argument('--device', default='cuda:0', help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--prediction_path', required=True, help='Prediction path')
    parser.add_argument('--gt_mode', default='SHELF14')
    parser.add_argument('--dataset_name', default='shelf')
    parser.add_argument('--eval_pck_th', default=0.2)
    parser.add_argument('--eval_reid_th', default=0.5)

    args = parser.parse_args()
    args.snapshot_root = args.prediction_path
    args.out_path = os.path.join(args.snapshot_root, 'pred.npy')

    # TODO: move config to yaml
    # Shelf ############################################################################
    if args.dataset_name == 'shelf':
        args.img_root = r"E:\dataset\Shelf\Evaluate"
        args.view_list = ["img_0", "img_1", "img_2", "img_3", "img_4"]
        args.img_suffix = ".png"
        args.frame_start = 0
        args.frame_end = 301
        args.gt_path = "./data/prepare/shelf_data/shelf_3d_eval_4dassoc.npy"
        args.camera_path = "./data/prepare/shelf_data/camera_params.npy"
    else:
        assert False, 'Unknown dataset {}'.format(args.dataset_name)

    # load data
    _, _, camera_proj, _, _ = triangular_parse_camera(args.camera_path)
    mocap_skel_converter = SkelConverter(args.gt_mode)
    # [frame, skel, joint, ch]
    mocap_gt_data = np.load(args.gt_path, allow_pickle=True).tolist()
    mocap_pred_data = np.load(args.out_path, allow_pickle=True).tolist()
    mocap_frame_list = prepare_imgs(args)
    if len(mocap_gt_data) != len(mocap_pred_data):
        warn_msg = "Inconsistent data length: src {}, gt {}".format(len(mocap_pred_data), len(mocap_gt_data))
        warnings.warn(warn_msg)
    else:
        print("Load src data {} from {}\n\tgt data {} from {}".format(len(mocap_pred_data), args.out_path,
                                                                      len(mocap_gt_data), args.gt_path))

    # evaluate
    eval_metric = []
    for pred_frame_idx, (pred_frame, gt_frame) in enumerate(zip(mocap_pred_data, mocap_gt_data)):
        print("Evaluate at Frame {}".format(pred_frame_idx))
        # 1. Skeleton type conversion
        gt_frame, pred_frame = mocap_skel_converter(gt_frame, pred_frame)
        # 2. Re-id
        pred_frame_id_map = util_re_id(pred_frame, gt_frame, th=args.eval_reid_th)
        # 3. Evaluate
        eval_metric.append(util_evaluate_driver(pred_frame, gt_frame, pred_frame_id_map,
                                                pck_th=args.eval_pck_th, mode=args.gt_mode))
        # 4. Snapshot (optional)
        if args.snapshot:
            snapshot(args.snapshot_root, mocap_frame_list[pred_frame_idx], pred_frame, camera_proj)
    # summary
    util_evaluate_summary(eval_metric, args.snapshot_root)
