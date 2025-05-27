import math
import random
import warnings

import numpy as np
import torch
import torch.utils.data as data
from einops import repeat, rearrange

from camera.triangulation import triangular_parse_camera


class TriDataset(data.Dataset):
    def __init__(self, joint_size=25, joint_size_gt=15, window_size=10,
                 debug=False, train=True, rot_aug=False,
                 center=0, gt_bgn=None, gt_end=None,
                 data_path=None,
                 gt_path=None,
                 camera_path=None,
                 dataset_name=None):
        # frame, [jc_joint3d, jc_camera_idx, jc_cntr]
        # jc_joint3d: body, joint, joint candidate, [3, 1]
        # jc_camera_idx: body, joint, joint candidate, camera index pair
        raw_data_list = np.load(data_path, allow_pickle=True).tolist()
        self.raw_data_len = len(raw_data_list)
        self.joint_size = joint_size
        self.joint_size_gt = joint_size_gt
        self.window_size = window_size
        self.batch_size = joint_size * window_size
        self.train = train
        self.debug = debug
        # augmentation
        self.rot_aug = rot_aug
        self.rad = math.pi / 180.
        self.center = center
        # remove specific data
        self.gt_bgn = gt_bgn
        self.gt_end = gt_end
        # jc part
        self.anal_crop = True
        self.anal_filter = True and self.train
        self.avail_map, self.avail_dict, self.avail_bId = self.__analysis_sequence_data(raw_data_list,
                                                                                        self.anal_crop,
                                                                                        self.anal_filter,
                                                                                        remove_bgn=gt_bgn,
                                                                                        remove_end=gt_end)
        self.body_num = len(self.avail_bId)
        self.seq_data = raw_data_list
        self.seq_pack, self.seq_flag = self.__arrange_pair_view(self.seq_data)
        # camera part
        self.seq_gt_pair = None
        self.camera_num, self.camera_res, self.camera_proj, _, _ = triangular_parse_camera(camera_path)
        self.camera_proj = rearrange(torch.from_numpy(self.camera_proj), 'b n d -> b d n')
        # gt part
        assert gt_path is not None, "Please specify the ground-truth path"
        # 1. arrange the raw data
        self.gt = np.load(gt_path, allow_pickle=True).tolist()
        if len(self.avail_map) != len(self.gt):
            if 'shelf' in dataset_name:
                self.gt = self.gt[300:len(self.avail_map) + 300]
            elif 'campus' in dataset_name:
                self.gt = self.gt[350:len(self.avail_map) + 350]
            elif 'coop_6_jump3' in dataset_name:
                self.gt = self.gt[665:]
            elif 'coop_8_jump6' in dataset_name:
                self.gt = self.gt[600:]
            else:
                assert False, 'Unknown dataset: {}'.format(dataset_name)
        # 2. locate the source's corresponding gt
        self.seq_gt_pair = []
        self.seq_gt_map = np.zeros((len(self.gt), self.avail_map.shape[-1]), dtype=bool)
        for fIdx, (seq_frame, gt_frame) in enumerate(zip(self.seq_data, self.gt)):
            gt_batch, gt_center = self.__arrange_gt(gt_frame)
            if len(gt_batch) == 0:
                self.seq_gt_pair.append({})
                continue
            # project back to 2d
            proj_set = self._get_proj(gt_batch)
            # loc the gt skel via cluster center dist
            seq_gt_frame = {}
            gt_center = np.array(gt_center)
            for seq_bId, seq_cntr in enumerate(seq_frame[2]):
                if len(seq_cntr) == 0:
                    continue
                dist = np.sqrt(np.sum(np.square(seq_cntr.T - gt_center), axis=-1))
                if np.any(dist < 0.5):
                    gt_bId = np.argmin(dist)
                    gt_skel = gt_batch[gt_bId]
                    seq_gt_frame[seq_bId] = {'gt_skel': gt_skel[np.newaxis, ::],
                                             'gt_cntr': gt_center[gt_bId],
                                             'gt_proj': proj_set[:, gt_bId][np.newaxis, ::]}
                    self.seq_gt_map[fIdx, seq_bId] = True
                else:
                    seq_gt_frame[seq_bId] = {}
            self.seq_gt_pair.append(seq_gt_frame)

    def __getitem__(self, idx):
        idx = idx % self.raw_data_len

        # training branch
        aug_rot_ang = None
        if self.rot_aug:
            aug_rot_ang = random.randint(0, 360)
        if self.train:
            cand_body_idx = random.choice(self.avail_bId)
            avail_frame_idx = random.choice(self.avail_dict[cand_body_idx])
            batch_frame_idx = self.__get_frame_idx(avail_frame_idx['bgn'], avail_frame_idx['end'])
            cand_frame_idx = random.randint(0, len(batch_frame_idx) - self.window_size)
            batch_frame_idx = batch_frame_idx[cand_frame_idx:cand_frame_idx + self.window_size]
            return self.__fetch_one_cand_data(cand_body_idx, batch_frame_idx, idx, aug_rot_ang)
        # evaluation branch
        else:
            batch_data = []
            for cur_body_idx, avail_frame_idx in self.avail_dict.items():
                for clip_idx in avail_frame_idx:
                    clip_bgn = clip_idx['bgn']
                    clip_end = clip_idx['end']
                    if clip_bgn <= idx <= clip_end:
                        batch_frame_idx = self.__get_frame_idx_eval(clip_bgn, idx, clip_end)
                        batch_data.append(self.__fetch_one_cand_data(cur_body_idx, batch_frame_idx, idx))
            return batch_data

    def __len__(self):
        return self.raw_data_len

    def get_loader(self, batch_size=1, shuffle=True, num_workers=0):
        return data.DataLoader(self, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    def _rotation(self, skel, ang):
        """
        :param skel: joint candidate, XYZ
        :param ang: int
        :return: joint candidate, XYZ
        """
        if ang is None:
            return skel
        rot_mat = np.zeros((3, 3))
        rot_mat[0, 0] = math.cos(ang * self.rad)
        rot_mat[0, 1] = -math.sin(ang * self.rad)
        rot_mat[1, 0] = math.sin(ang * self.rad)
        rot_mat[1, 1] = math.cos(ang * self.rad)
        rot_mat[2, 2] = 1
        skel = np.dot(skel, rot_mat)
        return skel

    def __arrange_gt(self, data):
        # note: input gt is 15x3
        # clean empty list
        data_ = []
        center_ = []
        for bodyId, body in enumerate(data):
            if len(body) == 0:
                continue
            body = np.array(body)
            data_.append(body)
            if len(body) == 25:
                hip_idx = 8
            else:
                hip_idx = 14
            center_.append(body[hip_idx])
        return data_, center_

    def __fetch_one_cand_data(self, body_idx, batch_frame_idx, cur_idx,
                              rot_ang=None):
        """
        :return batch_traj: (view, joint), (jc0, jc1, ..., jcT-1), XYZ
        :return batch_flag_t: empty flag for trajectory batch
        :return batch_stru: (view, frame), jc, XYZ
        :return batch_flag_s: empty flag for structure batch
        :return batch_center: 10, (x, y, z)
        :return aug_info: rot_ang: int
        :return gt_batch: -1 if None; else gt3d (10, 15, (x, y, z)), gt_proj (10, 5, 15, (x, y))
        """
        if self.debug:
            print("=> Select frame index: {} with {} on Cand {} with rot {}".format(cur_idx,
                                                                                    batch_frame_idx.tolist(),
                                                                                    body_idx,
                                                                                    rot_ang))

        # frame, view, joint, jc, XYZ
        seq_pack = np.array([self.seq_pack[fIdx][body_idx] for fIdx in batch_frame_idx])
        # frame, view, joint, jc
        seq_flag = np.array([self.seq_flag[fIdx][body_idx] for fIdx in batch_frame_idx])
        # cluster center
        batch_center = np.array([self.seq_data[fIdx][2][body_idx] for fIdx in batch_frame_idx]).squeeze(-1)
        # sub center
        f, v, j, jc, c = seq_pack.shape
        seq_pack[seq_flag] -= repeat(batch_center, 'f c -> f v j jc c', v=v, j=j, jc=jc)[seq_flag]

        # data augmentation
        if rot_ang is not None:
            seq_pack = self._rotation(seq_pack, rot_ang)
        else:
            rot_ang = -1
        # 1. trajectory -> (view, joint), (jc0, jc1, ..., jcT-1), c
        batch_traj = rearrange(seq_pack.copy(), 'f v j jc c -> (v j) (f jc) c', f=f, v=v, j=j, jc=jc, c=3)
        batch_flag_t = rearrange(seq_flag.copy(), 'f v j jc -> (v j) (f jc)', f=f, v=v, j=j, jc=jc)
        # 2. structure -> view, frame, jc, c
        batch_stru = rearrange(seq_pack.copy(), 'f v j jc c -> (v f) (j jc) c', f=f, v=v, j=j, jc=jc, c=3)
        batch_flag_s = rearrange(seq_flag.copy(), 'f v j jc -> (v f) (j jc)', f=f, v=v, j=j, jc=jc)
        # gt part
        gt_batch = [np.zeros((self.window_size, self.joint_size_gt, 3)),
                    np.zeros((self.window_size, self.camera_num, self.joint_size_gt, 2)),
                    False]
        if self.seq_gt_map[batch_frame_idx[:], body_idx].all():
            gt_batch[0] = np.vstack((self.seq_gt_pair[batch_frame_idx[0]][body_idx]['gt_skel'],
                                     self.seq_gt_pair[batch_frame_idx[1]][body_idx]['gt_skel'],
                                     self.seq_gt_pair[batch_frame_idx[2]][body_idx]['gt_skel'],
                                     self.seq_gt_pair[batch_frame_idx[3]][body_idx]['gt_skel'],
                                     self.seq_gt_pair[batch_frame_idx[4]][body_idx]['gt_skel'],
                                     self.seq_gt_pair[batch_frame_idx[5]][body_idx]['gt_skel'],
                                     self.seq_gt_pair[batch_frame_idx[6]][body_idx]['gt_skel'],
                                     self.seq_gt_pair[batch_frame_idx[7]][body_idx]['gt_skel'],
                                     self.seq_gt_pair[batch_frame_idx[8]][body_idx]['gt_skel'],
                                     self.seq_gt_pair[batch_frame_idx[9]][body_idx]['gt_skel']))
            gt_batch[1] = np.vstack((self.seq_gt_pair[batch_frame_idx[0]][body_idx]['gt_proj'],
                                     self.seq_gt_pair[batch_frame_idx[1]][body_idx]['gt_proj'],
                                     self.seq_gt_pair[batch_frame_idx[2]][body_idx]['gt_proj'],
                                     self.seq_gt_pair[batch_frame_idx[3]][body_idx]['gt_proj'],
                                     self.seq_gt_pair[batch_frame_idx[4]][body_idx]['gt_proj'],
                                     self.seq_gt_pair[batch_frame_idx[5]][body_idx]['gt_proj'],
                                     self.seq_gt_pair[batch_frame_idx[6]][body_idx]['gt_proj'],
                                     self.seq_gt_pair[batch_frame_idx[7]][body_idx]['gt_proj'],
                                     self.seq_gt_pair[batch_frame_idx[8]][body_idx]['gt_proj'],
                                     self.seq_gt_pair[batch_frame_idx[9]][body_idx]['gt_proj']))
            gt_batch[2] = True

        if self.debug:
            return (batch_frame_idx, body_idx), \
                (batch_traj, batch_flag_t), (batch_stru, batch_flag_s), \
                batch_center, rot_ang, gt_batch
        return (batch_traj, batch_flag_t), (batch_stru, batch_flag_s), \
            batch_center, rot_ang, gt_batch

    def __analysis_sequence_data(self, seq_data, crop=True, filter=True, max_body_num=20,
                                 remove_bgn=None, remove_end=None):
        """
        find the available jc data due to bodyId change
        :param seq_data: frame, [jc_joint3d, jc_camera_idx]
        :param crop: flag for cropping avail_map for the max bodyId
        :param filter: flag for filtering the clip if the length is smaller than window size
        :param remove_bgn: the begin index that we want to exclude
        :param remove_end: the end index that we want to exclude
        :return: avail_map: available map for the input sequence, [frame, body]
        :return: avail_dict: timestamp map for each clip, [body, [s0, e0], [s1, e1], ..., [sn, en]]
        :return: avail_bId: available body index list
        """
        seq_length = self.raw_data_len
        seq_body_max_id = -1  # for cropping max_body_num. find the max bodyIdx for the input sequence.
        # 1. crop the available map
        seq_avail_flg = np.zeros((max_body_num, seq_length), dtype=np.bool8)
        for fIdx, frame in enumerate(seq_data):
            frame_hit_map = frame[1]
            for bIdx, body_hit_map in enumerate(frame_hit_map):
                joint_cloud_num = np.sum(np.array([len(item) for item in body_hit_map]))
                if joint_cloud_num:
                    seq_avail_flg[bIdx, fIdx] = True
                    if seq_body_max_id < bIdx:
                        seq_body_max_id = bIdx
        seq_avail_flg = seq_avail_flg.T
        if crop:
            seq_avail_flg = seq_avail_flg[:, :(seq_body_max_id + 1)]
        # 2. remove the specific data
        if remove_bgn and remove_end:
            assert remove_end > remove_bgn, "Wrong begin and end index {} - {}".format(remove_bgn, remove_end)
            seq_avail_flg[remove_bgn:remove_end, :] = False
        # 3. find the a/b timestamp of clips for every body
        seq_avail_dict = {}
        for bIdx, body_flg in enumerate(seq_avail_flg.T):
            # case1: all full
            if np.sum(body_flg) == seq_length:
                seq_avail_dict[bIdx] = [{'bgn': 0,
                                         'end': seq_length - 1}]
                continue
            # case2. some empty
            str_lst = []
            end_lst = []
            body_flg = body_flg.copy()
            body_flg = np.insert(body_flg, 0, False)
            body_flg = np.append(body_flg, [False])
            for flag_idx in range(len(body_flg) - 1):
                flg_a = body_flg[flag_idx]
                flg_b = body_flg[flag_idx + 1]
                # [False, True] -> Begin
                if ~flg_a and flg_b:
                    str_lst.append(flag_idx)
                # [True, False] -> End
                elif flg_a and ~flg_b:
                    end_lst.append(flag_idx - 1)
            # append the last frame if the last is True
            if len(str_lst) > len(end_lst):
                offset = len(str_lst) - len(end_lst)
                for _ in range(offset):
                    end_lst.append(seq_length - 1)
            elif len(str_lst) < len(end_lst):
                offset = len(end_lst) - len(str_lst)
                for _ in range(offset):
                    str_lst.append(0)
            # store
            for str_idx, end_idx in zip(str_lst, end_lst):
                if filter:
                    frame_num = end_idx - str_idx + 1
                    if frame_num < self.window_size:
                        continue
                if not seq_avail_dict.get(bIdx):
                    seq_avail_dict[bIdx] = []
                seq_avail_dict[bIdx].append({'bgn': str_idx,
                                             'end': end_idx})
        return seq_avail_flg, seq_avail_dict, list(seq_avail_dict.keys())

    def __get_frame_idx(self, bgn_idx, end_idx):
        frame_idx = [bgn_idx for _ in range(self.window_size - 1)]
        frame_idx.extend([fIdx for fIdx in range(bgn_idx, end_idx + 1)])
        frame_idx.extend([end_idx for _ in range(self.window_size - 1)])
        return np.array(frame_idx, dtype=int)

    def __get_frame_idx_eval(self, bgn_idx, cur_idx, end_idx):
        frame_idx = [fIdx for fIdx in range(cur_idx, cur_idx - self.window_size + self.center, -1)]
        frame_idx = frame_idx[::-1]
        if self.center:
            frame_idx += [fIdx for fIdx in
                          range(cur_idx + 1, cur_idx + self.window_size - self.center + 1)]
        frame_idx = np.array(frame_idx, dtype=int)
        frame_idx[frame_idx < bgn_idx] = bgn_idx
        frame_idx[frame_idx > end_idx] = end_idx
        return frame_idx

    def _get_proj(self, skel_seq):
        """
        :param skel_seq: body, joint, (x, y, z)
        :return: reprojected 2d skel seq: camera view, body, joint, (x, y)
        """
        skel_seq = np.array(skel_seq).copy()
        b_num, j_num, _ = skel_seq.shape
        skel_seq = np.concatenate((skel_seq, np.ones((b_num, j_num, 1))), axis=-1)
        skel_seq = skel_seq[np.newaxis, ::]
        skel_seq = np.repeat(skel_seq, self.camera_num, axis=0)
        skel_seq = torch.from_numpy(skel_seq)
        skel_seq = rearrange(skel_seq, 'b n j c -> b (n j) c')
        skel_2d = torch.bmm(skel_seq, self.camera_proj)
        division = torch.cat(((skel_2d[:, :, 2] * self.camera_res[0]).unsqueeze(-1),
                              (skel_2d[:, :, 2] * self.camera_res[1]).unsqueeze(-1)),
                             dim=-1)
        skel_2d = torch.div(skel_2d[:, :, :2], division)
        skel_2d = rearrange(skel_2d, 'b (n j) c -> b n j c', n=b_num, j=j_num).numpy()
        return skel_2d

    def __arrange_pair_view(self, seq_data, view_num=10, joint_num=25, jc_num=4):
        """
        Fill joint candidate to $jc_num$ with empty joint
        :param seq_data: view, [body, joint candidate, XYZ], [body, joint candidate, camera pair], cluster center]
        :return: seq_pack: [frame, body, view, joint, joint candidate, XYZ]
        :return: seq_flag: [frame, body, view, joint, joint candidate, bool]
        """
        view_dict = {(0, 1): 0, (0, 2): 1, (0, 3): 2, (0, 4): 3, (1, 2): 4,
                     (1, 3): 5, (1, 4): 6, (2, 3): 7, (2, 4): 8, (3, 4): 9}
        seq_pack = []
        seq_flag = []
        for fIdx, frame in enumerate(seq_data):
            seq_body_mat = []
            seq_body_flg = []
            camera_pair_body = frame[1]
            for bIdx, camera_pair_joint in enumerate(camera_pair_body):
                # arrange to [view, joint]
                seq_jc = [[[] for j in range(joint_num)] for i in range(view_num)]
                for jIdx, jc_set in enumerate(camera_pair_joint):
                    for jcIdx, joint in enumerate(jc_set):
                        if tuple(joint) not in view_dict:
                            continue
                        vIdx = view_dict[tuple(joint)]
                        seq_jc[vIdx][jIdx].append(seq_data[fIdx][0][bIdx][jIdx][jcIdx])
                # fill with empty
                seq_jc_mat = np.zeros((view_num, joint_num, jc_num, 3))
                for vIdx, view in enumerate(seq_jc):
                    for jIdx, jc_set in enumerate(view):
                        for jcIdx, jc in enumerate(jc_set):
                            if jcIdx >= jc_num:
                                warnings.warn("f{}-b{}-v{}-j{}-jc{}".format(fIdx, bIdx, vIdx, jIdx, jcIdx))
                            seq_jc_mat[vIdx, jIdx, jcIdx % jc_num] = jc.squeeze(-1)
                seq_jc_flg = np.sum(np.abs(seq_jc_mat), axis=-1) != 0.
                seq_body_mat.append(seq_jc_mat)
                seq_body_flg.append(seq_jc_flg)
            seq_pack.append(seq_body_mat)
            seq_flag.append(seq_body_flg)
        return seq_pack, seq_flag
