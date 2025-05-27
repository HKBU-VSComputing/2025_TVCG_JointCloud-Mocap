import os
import time
import json
import torch
import torch.backends.cudnn as cudnn
import datetime
from datetime import datetime as timer
import argparse
import random
import numpy as np
from pathlib import Path
import inspect

from core.utils import NativeScalerWithGradNormCount as NativeScaler
from core.utils import TensorboardLogger
from core.utils import init_distributed_mode as util_init_distributed_mode
from core.utils import get_rank as util_get_rank
from core.utils import get_world_size as util_get_world_size
from core.utils import cosine_scheduler as util_cosine_scheduler
from core.utils import save_model as util_save_model
from core.utils import is_main_process as util_is_main_process
from core.utils import auto_load_model as util_auto_load_model

from core.optim_factory import create_optimizer
from core.engine import train_one_epoch

# core packages
from data.dataset import TriDataset
from network.jcsat import get_jcsat


def get_args():
    parser = argparse.ArgumentParser('Default Training Script', add_help=False)
    parser.add_argument('--batch_size', default=5, type=int)
    parser.add_argument('--epochs', default=500, type=int)
    parser.add_argument('--save_ckpt_freq', default=2, type=int)

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--weight_decay_end', type=float, default=None, help="""Final value of the
        weight decay. We use a cosine schedule for WD.
        (Set the same value with args.weight_decay to keep weight decay no change)""")

    # learning rate
    parser.add_argument('--lr', type=float, default=1.5e-4, metavar='LR',
                        help='learning rate (default: 1.5e-4)')
    parser.add_argument('--warmup_lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min_lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    # lr warm up protocol
    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')

    # Output config
    parser.add_argument('--output_dir', default='./out/shelf_opps',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./logger',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda:0',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument('--no_auto_resume', action='store_false', dest='auto_resume')
    parser.set_defaults(auto_resume=True)

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=0, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    return parser.parse_args()


def main(args):
    #######################################################################################
    config_set = {
        'shelf':
            {
                #######
                'data_path': './data/shelf_train/shelf_preprocess_v4_0312.npy',
                'remove_set': [300, 600],
                'eval_path': './data/shelf_test/shelf_preprocess_v4_0310.npy',
                #######
                'gt': ['./data/prepare/shelf_data/skel_with_gt_in_shelf14.npy'],
                'camera': "data/shelf_test/camera_params.npy",
            }
    }
    #######################################################################################
    data_name = 'shelf'
    config = config_set[data_name]
    # mode = 'small'
    mode = 'base'
    # mode = 'large'
    #######################################################################################
    dataset_data_path = config['data_path']
    dataset_eval_path = config['eval_path']
    dataset_remove_set = config['remove_set']
    dataset_gt_path = config['gt']
    dataset_camera = config['camera']

    util_init_distributed_mode(args)
    now = timer.now()
    current_time = now.strftime("_%y%m%d_%H%M%S")
    args.output_dir = args.output_dir + current_time
    print(args)

    # cude config
    device = torch.device(args.device)
    # fix the seed for reproducibility
    seed = args.seed + util_get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    # auto load
    args.log_dir = os.path.join(args.output_dir, args.log_dir)

    # dataset config
    args.dataset_name = data_name
    args.dataset_train = True
    args.dataset_rot_aug = True
    args.dataset_camera_path = dataset_camera
    args.dataset_data_path = dataset_data_path
    args.dataset_eval_path = dataset_eval_path
    args.dataset_remove_set = dataset_remove_set
    args.dataset_gt_path = dataset_gt_path
    args.mode = mode

    # package info
    args.pack_network = inspect.getfile(get_jcsat)
    args.pack_dataset = inspect.getfile(TriDataset)

    # dataset
    dataset_train = TriDataset(train=args.dataset_train, rot_aug=args.dataset_rot_aug,
                               data_path=args.dataset_data_path, gt_path=args.dataset_gt_path,
                               gt_bgn=args.dataset_remove_set[0], gt_end=args.dataset_remove_set[1],
                               camera_path=args.dataset_camera_path, dataset_name=args.dataset_name)
    dataset_eval = TriDataset(debug=False, train=True,
                              data_path=args.dataset_eval_path, gt_path=args.dataset_gt_path,
                              camera_path=args.dataset_camera_path, dataset_name=args.dataset_name)
    print('JcsatModel dataset has {} items'.format(len(dataset_train)))
    print('JcsatModel dataset has {} items'.format(len(dataset_eval)))

    # get data loader
    data_loader_train = dataset_train.get_loader(shuffle=True, batch_size=args.batch_size)
    data_loader_eval = dataset_eval.get_loader(shuffle=False)

    # get logger
    log_writer = None
    if args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = TensorboardLogger(log_dir=args.log_dir)

    # model
    model = get_jcsat(device, args=args)
    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Model = %s - %s" % (str(model), str(model_without_ddp)))
    print('number of params: {} M'.format(n_parameters / 1e6))

    total_batch_size = args.batch_size * util_get_world_size()
    num_training_steps_per_epoch = len(dataset_train) // args.batch_size
    print("LR = %.8f" % args.lr)
    print("Batch size = %d" % total_batch_size)
    print("Number of training steps = %d" % num_training_steps_per_epoch)
    print("Number of training examples per epoch = %d" % (total_batch_size * num_training_steps_per_epoch))

    optimizer = create_optimizer(args, model_without_ddp)
    loss_scaler = NativeScaler()

    print("Use step level LR & WD scheduler!")
    lr_schedule_values = util_cosine_scheduler(
        args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,
    )
    if args.weight_decay_end is None:
        args.weight_decay_end = args.weight_decay
    wd_schedule_values = util_cosine_scheduler(
        args.weight_decay, args.weight_decay_end, args.epochs, num_training_steps_per_epoch)
    print("Max WD = %.7f, Min WD = %.7f" % (max(wd_schedule_values), min(wd_schedule_values)))

    util_auto_load_model(args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                         loss_scaler=loss_scaler)

    # store config
    config_path = os.path.join(args.output_dir, 'config.log')
    with open(config_path, "w") as f:
        for i in vars(args):
            f.write(i + ":" + str(vars(args)[i]) + '\n')
    f.close()
    print("=> Saved in {}".format(config_path))

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    best_loss = 99999.
    for epoch in range(args.start_epoch, args.epochs):
        if log_writer is not None:
            log_writer.set_step(epoch * num_training_steps_per_epoch)
        train_stats = train_one_epoch(model, data_loader_train,
                                      optimizer, device, epoch, loss_scaler,
                                      args.clip_grad, log_writer=log_writer,
                                      start_steps=epoch * num_training_steps_per_epoch,
                                      lr_schedule_values=lr_schedule_values,
                                      wd_schedule_values=wd_schedule_values,
                                      data_loader_eval=data_loader_eval)
        if args.output_dir:
            if best_loss > train_stats['loss']:
                best_loss = train_stats['loss']
                # del last best
                for ckpt in os.listdir(args.output_dir):
                    if 'best' in ckpt:
                        os.remove(os.path.join(args.output_dir, ckpt))
                checkpoint_paths = os.path.join(args.output_dir, "best-%s-%s" % (str(epoch), str(best_loss)))
                torch.save(model.state_dict(), checkpoint_paths)
            if (epoch + 1) % args.save_ckpt_freq == 0 or epoch + 1 == args.epochs:
                util_save_model(args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                                loss_scaler=loss_scaler, epoch=epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch, 'n_parameters': n_parameters}

        if args.output_dir and util_is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    opts = get_args()
    main(opts)
