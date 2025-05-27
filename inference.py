import os
import cv2
import torch
import torch.backends.cudnn as cudnn
import argparse
import random
import numpy as np
from pathlib import Path
import inspect

# from core.config import load_config
from core.utils import get_rank as util_get_rank
from camera.triangulation import triangular_parse_camera
from core.skel_painter import draw_skel_for_each_view

# core packages
from data.dataset import TriDataset
from network.jcsat import get_jcsat


def prepare_model(model, model_path, device):
    if not os.path.exists(model_path):
        print("No file found at {}".format(model_path))
        return None

    client_states = torch.load(model_path, map_location=device, weights_only=False)
    state_dict = client_states['model']
    model.load_state_dict(state_dict, strict=True)

    model.eval()
    return model


def prepare_imgs(args):
    img_list = []
    if args.frame_start is None and args.frame_end is None:
        for view in args.view_list:
            view_list = []
            img_list_tmp = os.listdir(os.path.join(args.img_root, view))
            for frame_name in img_list_tmp:
                frame_list = os.path.join(args.img_root, "{}/{}".format(view, frame_name))
                view_list.append(frame_list)
            img_list.append(view_list)
        img_list = np.array(img_list)
        img_list = img_list.transpose(1, 0).tolist()
        return img_list

    for frameIdx in range(args.frame_start, args.frame_end):
        frameIdx += 1
        frame_list = [os.path.join(args.img_root, "{}/{:04d}{}".format(view, frameIdx, args.img_suffix)) for view in
                      args.view_list]
        img_list.append(frame_list)
    return img_list


def snapshot(dst_root, src_list, skel_data, camera_proj_mat):
    dst_root = dst_root + '/img'
    file_name = src_list[0].split('/')[-1][:-4] + '.jpg'
    snapshot_dst_path = os.path.join(dst_root, file_name)
    Path(dst_root).mkdir(parents=True, exist_ok=True)
    if len(skel_data) == 0:
        return
    _, _, joint_num, _ = skel_data.shape
    skel_type = 'BODY25' if joint_num == 25 else 'SHELF14'
    canvas_out = draw_skel_for_each_view(skel_data, camera_proj_mat, src_list, type=skel_type)
    out = np.hstack(canvas_out)
    snapshot_out = cv2.resize(out, None, fx=0.3, fy=0.3)
    cv2.imwrite(snapshot_dst_path, snapshot_out)


def transpose(src_data, trans_mode):
    if trans_mode == 'XZ':
        dst_data = np.array([src_data[::, 0], src_data[::, 2], src_data[::, 1]]).T
    elif trans_mode == 'XY':
        dst_data = np.array([src_data[::, 1], src_data[::, 0], -src_data[::, 2]]).T
    else:
        dst_data = src_data
    return dst_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Simple parser for JCSAT Mocap framework")
    parser.add_argument('--snapshot_flag', action='store_true', default=False, help='Store snapshot for visualization.')
    parser.add_argument('--device', default='cuda:0', help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--model_path', required=True, help='Checkpoint path')
    parser.add_argument('--dataset_name', default='shelf')
    args = parser.parse_args()

    # TODO: move config to yaml
    # Shelf ############################################################################
    if args.dataset_name == 'shelf':
        args.img_root = "./data/prepare/shelf_data/Shelf/Evaluate"
        args.view_list = ["img_0", "img_1", "img_2", "img_3", "img_4"]
        args.img_suffix = ".png"
        args.frame_start = 0
        args.frame_end = 301
        args.dataset_path = './data/prepare/shelf_test/shelf_preprocess_v4_0310.npy'
        args.camera_path = "./data/prepare/shelf_data/camera_params.npy"
        args.gt_path = "./data/prepare/shelf_data/skel_with_gt_in_shelf14.npy"
        args.trans_mode = None
    else:
        assert False, 'Unknown dataset {}'.format(args.dataset_name)

    # base config
    args.infer_buffer_length = 1
    args.infer_center = 0
    args.snapshot_root = str(Path(args.model_path).parent / f'{Path(args.model_path).stem}_results')
    args.snapshot = args.snapshot_flag and args.infer_buffer_length == 1
    args.out_path = os.path.join(args.snapshot_root, 'pred.npy')
    _, _, camera_proj, _, _ = triangular_parse_camera(args.camera_path)

    # cuda config
    device = torch.device(args.device)
    # fix the seed for reproducibility
    seed = args.seed + util_get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    # package info
    args.pack_network = inspect.getfile(get_jcsat)
    args.pack_dataset = inspect.getfile(TriDataset)

    # dataset
    args.dataset_camera_path = args.camera_path
    dataset = TriDataset(debug=False, train=False,
                         data_path=args.dataset_path,
                         center=args.infer_center,
                         camera_path=args.camera_path,
                         gt_path=args.gt_path,
                         dataset_name=args.dataset_name)
    data_loader = dataset.get_loader(shuffle=False)
    print('Eval dataset has {} items'.format(len(dataset)))
    frame_list = prepare_imgs(args)

    # model
    model = get_jcsat(device, args=args, mode='medium')
    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Model = %s - %s" % (str(model), str(model_without_ddp)))
    print('number of params: {} M'.format(n_parameters / 1e6))
    print("Model path: {}".format(args.model_path))
    model = prepare_model(model, args.model_path, device)
    if model is None:
        exit(-1)

    # store the config
    Path(args.snapshot_root).mkdir(parents=True, exist_ok=True)
    config_path = os.path.join(args.snapshot_root, 'config.log')
    with open(config_path, "w") as f:
        for i in vars(args):
            f.write(i + ":" + str(vars(args)[i]) + '\n')
    f.close()
    print("=> Saved in {}".format(config_path))

    # out
    pred_data = []
    if args.infer_center:
        pred_s = -args.infer_center - args.infer_buffer_length
        pred_e = -args.infer_center
    else:
        pred_s = -args.infer_buffer_length
        pred_e = None

    ###############################################
    torch.cuda.synchronize()
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    timings = np.zeros((301, 1))
    ###############################################

    with torch.no_grad():
        for dataIdx, (dataSkel, frame_path) in enumerate(zip(data_loader, frame_list)):
            print("=> With {} samples".format(len(dataSkel)))
            prediction_set = []
            # To batch
            if len(dataSkel) == 0:
                pred_data.append([])
                continue
            batch_traj = torch.cat([item[0][0] for item in dataSkel])
            batch_t_flg = torch.cat([item[0][1] for item in dataSkel])
            batch_stru = torch.cat([item[1][0] for item in dataSkel])
            batch_s_flg = torch.cat([item[1][1] for item in dataSkel])
            batch_clst_cntr = torch.cat([item[2] for item in dataSkel])

            starter.record()
            preds = model.prediction(((batch_traj, batch_t_flg), (batch_stru, batch_s_flg), batch_clst_cntr))
            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[dataIdx] = curr_time
            for pred in preds:
                pred = pred[pred_s:pred_e, :, :]
                pred = transpose(pred, args.trans_mode)
                prediction_set.append(pred)
            prediction_set = np.array(prediction_set)
            if args.snapshot:
                # reprojection
                snapshot(args.snapshot_root, frame_path, prediction_set, camera_proj)  # draw for each candidate body
            pred_data.append(prediction_set)
            print('-> {}'.format(dataIdx))

        np.save(args.out_path, np.array(pred_data, dtype=object))
        avg = timings.sum() / 301.
        print('\navg={}\n'.format(avg))
