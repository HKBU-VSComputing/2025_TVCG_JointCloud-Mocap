import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from einops import repeat, rearrange

from camera.triangulation import triangular_parse_camera
from network.pos_emb import PosEmbFactory
from network.vit import Transformer
from thirdparty.OTK.otk.layers import OTKernel


class BaseEncoder(nn.Module):
    def __init__(self,
                 in_dim, emb_dim, out_dim,
                 mlp_depth, mlp_heads, mlp_dim_head, mlp_dim, mlp_dropout=0.,
                 emb_dropout=0., device='cuda:0'):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.emb_dim = emb_dim
        self.emb_dropout = emb_dropout

        self.mlp_depth = mlp_depth
        self.mlp_heads = mlp_heads
        self.mlp_dim_head = mlp_dim_head
        self.mlp_dim = mlp_dim
        self.mlp_dropout = mlp_dropout
        self.device = device

        self.enc_proj = nn.Linear(in_dim, emb_dim) if in_dim != emb_dim else nn.Identity()
        self.dec_proj = nn.Linear(emb_dim, out_dim) if emb_dim != out_dim else nn.Identity()

        self.dropout = nn.Dropout(emb_dropout)
        self.enc_transformer = Transformer(emb_dim, mlp_depth, mlp_heads, mlp_dim_head, mlp_dim, mlp_dropout)
        self.to_latent = nn.Identity()


class JcTrajEncoder(BaseEncoder):
    def __init__(self, in_dim=3, emb_dim=16, out_dim=32,
                 mlp_depth=4, mlp_heads=3, mlp_dim_head=2, mlp_dim=16,
                 jc_num=4, device='cuda:0'):
        super().__init__(in_dim, emb_dim, out_dim,
                         mlp_depth, mlp_heads, mlp_dim_head, mlp_dim, device=device)

        self.enc_fid_embed_fac = PosEmbFactory(emb_type="fourier", d_in=1, d_pos=self.emb_dim)
        self.enc_sig_embed_fac = PosEmbFactory(emb_type="fourier", d_in=self.in_dim, d_pos=self.emb_dim)

        self.window_size = 10
        self.view_num = 10
        self.joint_size = 25
        self.jc_num = jc_num
        frame_idx = torch.from_numpy(np.array([[[x] for x in range(self.window_size)]])).float()
        self.frame_embedding = self.enc_fid_embed_fac(frame_idx).permute(0, 2, 1).to(self.device)
        traj_token = nn.Parameter(torch.randn((1, self.view_num * self.joint_size, emb_dim)))
        self.traj_token = traj_token.to(self.device)
        self.op_relu = nn.ReLU()
        self.op_layer = OTKernel(in_dim=self.out_dim,
                                 out_size=10,
                                 heads=1,
                                 log_domain=True,
                                 position_encoding='hard')

    def forward(self, jc, flg):
        # f v j jc c -> (v j) (f jc) c
        b, vj, fjc, c = jc.shape
        jc = rearrange(jc, 'b vj fjc c -> (b vj) fjc c')
        flg = rearrange(flg, 'b vj fjc -> (b vj) fjc').to(self.device)
        flg_ = repeat(flg, '(b vj) fjc -> (b vj) h s fjc', b=b, vj=vj, h=self.mlp_heads, s=fjc)
        # signal embedding
        sig_emb = self.enc_sig_embed_fac(jc).permute(0, 2, 1)
        # frame embedding
        fra_emb = repeat(self.frame_embedding, 'ob f c -> (b ob vj) (f jc) c', b=b, jc=self.jc_num, vj=vj)
        x = self.enc_proj(jc) + sig_emb + fra_emb
        # process
        x = self.dropout(x)
        x = self.enc_transformer(x, flg_)
        x = self.dec_proj(x)
        # jc optimal transport
        x = self.op_relu(x)
        x = self.op_layer(x, flg)
        x = rearrange(x, '(b vj) f c -> b vj f c', b=b, vj=vj)
        x = rearrange(x, 'b (v j) f c -> b v f j c', v=self.view_num, j=self.joint_size)

        return x


class JcStrucEncoder(BaseEncoder):
    def __init__(self, in_dim=3, emb_dim=64, out_dim=32,
                 mlp_depth=4, mlp_heads=3, mlp_dim_head=2, mlp_dim=16,
                 jc_num=4, device='cuda:0'):
        super().__init__(in_dim, emb_dim, out_dim,
                         mlp_depth, mlp_heads, mlp_dim_head, mlp_dim, device=device)

        self.enc_jid_embed_fac = PosEmbFactory(emb_type="fourier", d_in=1, d_pos=self.emb_dim)
        self.enc_sig_embed_fac = PosEmbFactory(emb_type="fourier", d_in=self.in_dim, d_pos=self.emb_dim)

        self.window_size = 10
        self.view_num = 10
        self.joint_size = 25
        self.jc_num = jc_num
        joint_idx = torch.from_numpy(np.array([[[x] for x in range(self.joint_size)]])).float()
        self.joint_embedding = self.enc_jid_embed_fac(joint_idx).permute(0, 2, 1).to(self.device)
        stru_token = nn.Parameter(torch.randn((1, self.view_num * self.window_size, emb_dim)))
        self.stru_token = stru_token.to(self.device)
        self.op_relu = nn.ReLU()
        self.op_layer = OTKernel(in_dim=self.out_dim, out_size=25, heads=1, log_domain=True,
                                 position_encoding='hard')

    def forward(self, jc, flg):
        # f v j jc c -> (v f) (j jc) c
        b, vf, jjc, c = jc.shape
        jc = rearrange(jc, 'b vf jjc c -> (b vf) jjc c')
        flg = rearrange(flg, 'b vf jjc -> (b vf) jjc').to(self.device)
        flg_ = repeat(flg, '(b vf) jjc -> (b vf) h s jjc', b=b, vf=vf, h=self.mlp_heads, s=jjc)
        # signal embedding
        sig_emb = self.enc_sig_embed_fac(jc).permute(0, 2, 1)
        # joint embedding
        jnt_emb = repeat(self.joint_embedding, 'ob j c -> (b ob vf) (j jc) c', b=b, jc=self.jc_num, vf=vf)
        x = self.enc_proj(jc) + sig_emb + jnt_emb
        # process
        x = self.dropout(x)
        x = self.enc_transformer(x, flg_)
        x = self.dec_proj(x)
        # jc optimal transport
        x = self.op_relu(x)
        x = self.op_layer(x, flg)
        x = rearrange(x, '(b vf) j c -> b vf j c', b=b, vf=vf)
        x = rearrange(x, 'b (v f) j c -> b v f j c', v=self.view_num, f=self.window_size)

        return x


class JcsatModel(BaseEncoder):
    def __init__(self, enc_t, enc_s,
                 emb_dim=32, out_dim=3,
                 mlp_depth=2, mlp_heads=3, mlp_dim_head=2, mlp_dim=8,
                 debug=False, device='cuda:0', camera_path=None):
        super().__init__(enc_s.out_dim, emb_dim, out_dim,
                         mlp_depth, mlp_heads, mlp_dim_head, mlp_dim, device=device)
        # encoder
        self.enc_t = enc_t
        self.enc_s = enc_s
        self.device = enc_t.device
        self.morphing = nn.Sequential(
            nn.Linear(25, 20).double(),
            nn.Linear(20, 15).double(),
        )
        self.op_relu = nn.ReLU()
        self.op_layer = OTKernel(in_dim=self.emb_dim,
                                 out_size=1,
                                 heads=1,
                                 log_domain=True,
                                 position_encoding='hard')

        # attr
        self.window_size = 10
        self.view_num = 10
        self.joint_size = 25
        self.joint_size_gt = 15
        self.debug = debug
        self.dropout_ratio = 0.4

        # camera param
        camera_num, camera_res, camera_proj, _, _ = triangular_parse_camera(camera_path)
        self.camera_num = camera_num
        self.camera_res = camera_res
        self.camera_prj = torch.from_numpy(camera_proj).to(torch.float64).to(self.device)
        self.camera_prj = rearrange(self.camera_prj, 'b n d -> b d n')

        # other
        self.rad = torch.pi / 180.
        moti_token = nn.Parameter(torch.randn((self.window_size * self.joint_size, emb_dim)))
        self.moti_token = moti_token.to(self.device)
        self.pdist = nn.PairwiseDistance(p=2)
        self.bce_log_loss = nn.BCEWithLogitsLoss()
        self.relu = nn.ReLU()
        # body25
        self.limb_map_body25 = np.array([[17, 15], [15, 0], [0, 16], [16, 18],  # head (0-3)
                                         [1, 2], [2, 3], [3, 4],  # right upper torso (4-6)
                                         [1, 5], [5, 6], [6, 7],  # left upper torso (7-9)
                                         [8, 9], [9, 10], [10, 11],  # right lower (10-12)
                                         [11, 22], [11, 24], [22, 23],  # right foot (13-15)
                                         [8, 12], [12, 13], [13, 14],  # left lower (16-18)
                                         [14, 19], [14, 21], [19, 20]], dtype=int)  # left foot (19-21)
        self.symm_map_body25 = np.array([[0, 3], [1, 2],  # head
                                         [4, 7], [5, 8], [6, 9],  # upper
                                         [10, 16], [11, 17], [12, 18],  # lower
                                         [13, 19], [14, 20], [15, 21]], dtype=int)  # foot
        self.cent_map_body25 = np.array([[1, 2, 5], [8, 9, 12]], dtype=int)  # center, RJoint, LJoint
        # shelf15
        self.limb_map_shelf15 = np.array([[12, 8], [8, 7], [7, 6],  # right upper torso (0-2)
                                          [12, 9], [9, 10], [10, 11],  # left upper torso (3-5)
                                          [14, 2], [2, 1], [1, 0],  # right lower torso (6-8)
                                          [14, 3], [3, 4], [4, 5]], dtype=int)  # left lower torso (9-11)
        self.symm_map_shelf15 = np.array([[0, 3], [1, 4], [2, 5],  # upper
                                          [6, 9], [7, 10], [8, 11]], dtype=int)  # lower
        self.cent_map_shelf15 = np.array([[14, 2, 3]], dtype=int)  # center, RJoint, LJoint
        # skel type conversion
        self.body25_to_shelf14 = np.array([11, 10, 9, 12, 13, 14, 4, 3, 2, 5, 6, 7, 1, 0, 8], dtype=int)
        self.pcp_shelf14 = np.array([[9, 7, 10, 6, 3, 1, 4, 0, 12, 12],
                                     [10, 8, 11, 7, 4, 2, 5, 1, 13, 14]], dtype=int)

    def forward(self, skel_batch):
        (traj_data, traj_flag), (stru_data, stru_flag), clst_cntr, aug_rot, (gt, gt_proj, gt_flg) = skel_batch
        b, *_ = traj_data.shape

        # drop out
        drop_ratio = np.random.uniform(0, self.dropout_ratio)
        traj_num = torch.where(traj_flag)
        traj_drop = np.random.choice(len(traj_num[0]), int(len(traj_num[0]) * drop_ratio), replace=False)
        traj_flag[traj_num[0][traj_drop], traj_num[1][traj_drop], traj_num[2][traj_drop]] = False
        drop_ratio = np.random.uniform(0, self.dropout_ratio)
        stru_num = torch.where(stru_flag)
        stru_drop = np.random.choice(len(stru_num[0]), int(len(stru_num[0]) * drop_ratio), replace=False)
        stru_flag[stru_num[0][stru_drop], stru_num[1][stru_drop], stru_num[2][stru_drop]] = False

        # trajectory enc
        traj_tokens = self.enc_t(traj_data.to(self.device).float(), traj_flag)
        # structure enc
        stru_tokens = self.enc_s(stru_data.to(self.device).float(), stru_flag)
        # to motion
        moti_tokens = traj_tokens + stru_tokens
        moti_tokens = rearrange(moti_tokens, "b v f j c -> (b f j) v c")
        # process
        x = self.enc_proj(moti_tokens)
        x = self.enc_transformer(x)
        # view optimal transport
        x = self.op_relu(x)
        x = self.op_layer(x).squeeze(1)
        skel_pred = self.dec_proj(x)
        skel_pred = rearrange(skel_pred, '(b f j) c -> b (f j) c', b=b, f=self.window_size, j=self.joint_size)

        # repair the rotation and the cluster center
        clst_cntr = clst_cntr.to(self.device)
        skel_pred = self._unrotation_with_batch(skel_pred, aug_rot) \
                    + repeat(clst_cntr, 'b f c -> b (f j) c', j=self.joint_size)

        # morphing
        skel_pred = rearrange(skel_pred, 'b (f j) c -> b f j c', f=self.window_size, j=self.joint_size)
        skel_pred_15, morp_loss = self._morphing(skel_pred, 'soft')

        # skeletal loss
        b_loss = self._loss_bone_length_15(skel_pred_15)

        # gt loss
        gt_loss = torch.tensor(0.0, requires_grad=True).to(self.device)
        pcp_loss = torch.tensor(0.0, requires_grad=True).to(self.device)
        p_loss = torch.tensor(0.0, requires_grad=True).to(self.device)
        p_pcp_loss = torch.tensor(0.0, requires_grad=True).to(self.device)
        gt_log_flg = False
        if torch.any(gt_flg):
            gt_log_flg = True
            gt = gt.to(self.device)
            gt_proj = gt_proj.to(self.device)
            gt_loss = F.mse_loss(gt[gt_flg], skel_pred_15[gt_flg])
            pcp_loss = torch.mean(self._loss_pcp(gt[gt_flg], skel_pred_15[gt_flg]))
            # gt reprojection supervision
            p_loss, p_pcp_loss = self._loss_reprojection_pcp(skel_pred_15[gt_flg],
                                                             gt_proj[gt_flg])

        # total loss
        total_loss = gt_loss + pcp_loss + torch.sum(p_loss) + torch.sum(p_pcp_loss) + morp_loss + b_loss[0] + b_loss[1]

        #############################################
        print("Tot {:.5f} => Gt {:.5f}, PCP {:.5f}, Bone {:.5f} - {:.5f}".format(total_loss.item(),
                                                                                 gt_loss.item(),
                                                                                 pcp_loss.item(),
                                                                                 b_loss[0].item(),
                                                                                 b_loss[1].item()))
        if gt_log_flg:
            print("          => P0 {:.5f}, P1 {:.5f}, P2 {:.5f}, P3 {:.5f}, P4 {:.5f}".format(p_loss[0].item(),
                                                                                              p_loss[1].item(),
                                                                                              p_loss[2].item(),
                                                                                              p_loss[3].item(),
                                                                                              p_loss[4].item()))
            print("          => P0 {:.5f}, P1 {:.5f}, P2 {:.5f}, P3 {:.5f}, P4 {:.5f}".format(p_pcp_loss[0].item(),
                                                                                              p_pcp_loss[1].item(),
                                                                                              p_pcp_loss[2].item(),
                                                                                              p_pcp_loss[3].item(),
                                                                                              p_pcp_loss[4].item()))
        #############################################

        if self.debug:
            return gt, skel_pred_15

        return total_loss

    def prediction(self, skel_batch):
        (traj_data, traj_flag), (stru_data, stru_flag), clst_cntr, *_ = skel_batch
        b, *_ = traj_data.shape

        # trajectory enc
        traj_tokens = self.enc_t(traj_data.to(self.device).float(), traj_flag)
        # structure enc
        stru_tokens = self.enc_s(stru_data.to(self.device).float(), stru_flag)
        # to motion
        moti_tokens = traj_tokens + stru_tokens
        moti_tokens = rearrange(moti_tokens, "b v f j c -> (b f j) v c")
        # process
        x = self.enc_proj(moti_tokens)
        x = self.enc_transformer(x)
        # view optimal transport
        x = self.op_relu(x)
        x = self.op_layer(x).squeeze(1)
        skel_pred = self.dec_proj(x)
        skel_pred = rearrange(skel_pred, '(b f j) c -> b (f j) c', b=b, f=self.window_size, j=self.joint_size)
        # repair the rotation and the cluster center
        clst_cntr = clst_cntr.to(self.device)
        skel_pred = skel_pred + repeat(clst_cntr, 'b f c -> b (f j) c', j=self.joint_size)
        # morphing
        skel_pred = rearrange(skel_pred, 'b (f j) c -> b f j c', f=self.window_size, j=self.joint_size)
        skel_pred_15, _ = self._morphing(skel_pred, 'soft')
        # output
        skel_pred_15 = skel_pred_15.cpu().numpy()
        return skel_pred_15

    def evaluation(self, skel_batch):
        (traj_data, traj_flag), (stru_data, stru_flag), clst_cntr, aug_rot, (gt, _, gt_flg) = skel_batch
        b, *_ = traj_data.shape
        # trajectory enc
        traj_tokens = self.enc_t(traj_data.to(self.device).float(), traj_flag)
        # structure enc
        stru_tokens = self.enc_s(stru_data.to(self.device).float(), stru_flag)
        # to motion
        moti_tokens = traj_tokens + stru_tokens
        moti_tokens = rearrange(moti_tokens, "b v f j c -> (b f j) v c")
        # process
        x = self.enc_proj(moti_tokens)
        x = self.enc_transformer(x)
        # view optimal transport
        x = self.op_relu(x)
        x = self.op_layer(x).squeeze(1)
        skel_pred = self.dec_proj(x)
        skel_pred = rearrange(skel_pred, '(b f j) c -> b (f j) c', b=b, f=self.window_size, j=self.joint_size)
        # repair the rotation and the cluster center
        clst_cntr = clst_cntr.to(self.device)
        skel_pred = self._unrotation_with_batch(skel_pred, aug_rot) \
                    + repeat(clst_cntr, 'b f c -> b (f j) c', j=self.joint_size)
        # morphing
        skel_pred = rearrange(skel_pred, 'b (f j) c -> b f j c', f=self.window_size, j=self.joint_size)
        skel_pred_15, _ = self._morphing(skel_pred, 'soft')
        # gt loss
        pcp_loss = torch.tensor(0.0, requires_grad=True).to(self.device)
        if torch.any(gt_flg):
            gt = gt.to(self.device)
            gt_loss = F.mse_loss(gt[gt_flg], skel_pred_15[gt_flg])
            pcp_loss = torch.mean(self._loss_pcp(gt[gt_flg], skel_pred_15[gt_flg]))
        # total loss
        total_loss = gt_loss + pcp_loss
        return total_loss

    def _loss_velocity(self, skel_seq):
        # frame index, joint index
        # consistency of square of velocity
        skel_seq = rearrange(skel_seq, 'b (p n) c -> b p n c', p=self.window_size, n=self.joint_size)
        v_sqrt = torch.sqrt(torch.mean(torch.diff(skel_seq, dim=1).square(), dim=-1))
        loss = torch.mean(torch.mean(torch.abs(torch.diff(v_sqrt, dim=1)), dim=1))
        return loss

    def _loss_velocity_15(self, skel_seq):
        # shelf15 version
        # frame index, joint index
        # consistency of square of velocity
        v_sqrt = torch.sqrt(torch.mean(torch.diff(skel_seq, dim=1).square(), dim=-1))
        loss = torch.mean(torch.mean(torch.abs(torch.diff(v_sqrt, dim=1)), dim=1))
        return loss

    def _loss_reprojection(self, skel3d_seq, skel2d_seq):
        """
        :param skel2d_seq: 1, 10, 5, 15, 2; ground-truth
        :return: pair distance
        """
        prj_skel2d_seq = self._projection(skel3d_seq)
        skel2d_seq = rearrange(skel2d_seq[0], 'p v n c -> v (p n) c')
        loss = torch.mean(self.pdist(skel2d_seq, prj_skel2d_seq), dim=-1)
        return loss

    def _loss_reprojection_pcp(self, skel3d_seq, skel2d_seq):
        """
        :param skel3d_seq: batch, frame, joint, XYZ; prediction
        :param skel2d_seq: batch, 10, 5, 15, 2; ground-truth
        :return: view, pair distance
        """
        b = skel3d_seq.shape[0]
        prj_skel2d_seq = self._projection(skel3d_seq)
        skel2d_seq = rearrange(skel2d_seq, 'b p v n c -> (b v) (p n) c')
        loss_proj = torch.mean(self.pdist(skel2d_seq, prj_skel2d_seq), dim=-1)
        loss_proj = torch.mean(rearrange(loss_proj, '(b v) -> b v', b=b), dim=0)

        prj_skel2d_seq = rearrange(prj_skel2d_seq, 'bv (p n) c -> bv p n c', p=self.window_size, n=self.joint_size_gt)
        skel2d_seq = rearrange(skel2d_seq, 'bv (p n) c -> bv p n c', p=self.window_size, n=self.joint_size_gt)
        loss_pcp = self._loss_pcp(prj_skel2d_seq, skel2d_seq)
        loss_pcp = torch.mean(rearrange(loss_pcp, '(b v) -> b v', b=b), dim=0)
        return loss_proj, loss_pcp

    def _loss_bone_length(self, skel_seq):
        # frame index, joint index
        skel_seq = rearrange(skel_seq, 'b (p n) c -> b p n c', p=self.window_size, n=self.joint_size)
        joint_a_seq = skel_seq[:, :, self.limb_map_body25[:, 0], :]
        joint_b_seq = skel_seq[:, :, self.limb_map_body25[:, 1], :]
        limb_length = self.pdist(joint_a_seq, joint_b_seq)
        # 1. consistency of bone length in temporal
        dist_loss = torch.mean(torch.mean(torch.abs(torch.diff(limb_length, dim=1)), dim=1))

        # 2. consistency of symmetric bone length in kinematic
        symm_limb_a_seq = limb_length[:, :, self.symm_map_body25[:, 0]]
        symm_limb_b_seq = limb_length[:, :, self.symm_map_body25[:, 1]]
        symm_loss = torch.mean(torch.mean(torch.abs(symm_limb_a_seq - symm_limb_b_seq), dim=1))

        # 3. neck and mid_hip are the center of the torso
        center_seq = skel_seq[:, :, self.cent_map_body25[:, 0], :]
        side_seq = skel_seq[:, :, self.cent_map_body25[:, 1:], :]
        center_seq_ = torch.mean(side_seq, dim=2)
        cent_loss = F.mse_loss(center_seq, center_seq_)

        loss = dist_loss + symm_loss + cent_loss
        return loss

    def _loss_bone_length_15(self, skel_seq):
        # frame index, joint index
        joint_a_seq = skel_seq[:, :, self.limb_map_shelf15[:, 0], :]
        joint_b_seq = skel_seq[:, :, self.limb_map_shelf15[:, 1], :]
        limb_length = self.pdist(joint_a_seq, joint_b_seq)
        # 1. consistency of bone length in temporal
        dist_loss = torch.mean(torch.mean(torch.abs(torch.diff(limb_length, dim=1)), dim=1))

        # 2. consistency of symmetric bone length in kinematic
        symm_limb_a_seq = limb_length[:, :, self.symm_map_shelf15[:, 0]]
        symm_limb_b_seq = limb_length[:, :, self.symm_map_shelf15[:, 1]]
        symm_loss = torch.mean(torch.mean(torch.abs(symm_limb_a_seq - symm_limb_b_seq), dim=1))

        return dist_loss, symm_loss

    def _loss_morphing(self, src, tar):
        """
        :param src: body25, (b, f, 25, 3)
        :param tar: shelf15, (b, f, 15, 3)
        """
        src_hard = self._hard_morphing(src)
        loss = F.mse_loss(src_hard, tar)
        return loss

    def _projection(self, skel_seq):
        """
        :return: (batch, view), (frame, joint), XY
        """
        b, p, n, _ = skel_seq.shape
        skel_seq = rearrange(skel_seq, 'b p n c -> b (p n) c')
        skel_seq = torch.cat((skel_seq, torch.ones((b, p * n, 1)).to(self.device)), dim=-1)
        skel_seq = repeat(skel_seq, 'b n d -> (b v) n d', v=self.camera_num)
        camera_prj = repeat(self.camera_prj, "v n d -> (b v) n d", b=b)
        skel_2d = torch.bmm(skel_seq, camera_prj)
        division = torch.cat(((skel_2d[:, :, 2] * self.camera_res[0]).unsqueeze(-1),
                              (skel_2d[:, :, 2] * self.camera_res[1]).unsqueeze(-1)),
                             dim=-1)
        skel_2d = torch.div(skel_2d[:, :, :2], division)
        return skel_2d

    def _unrotation(self, skel, ang):
        """
        :param skel: 1, 250, 3
        :param ang: int
        :return: 1, 250, 3
        """
        if ang is None:
            return skel
        ang = -ang
        rot_mat = torch.zeros((3, 3))
        rot_mat[0, 0] = torch.cos(ang * self.rad)
        rot_mat[0, 1] = -torch.sin(ang * self.rad)
        rot_mat[1, 0] = torch.sin(ang * self.rad)
        rot_mat[1, 1] = torch.cos(ang * self.rad)
        rot_mat[2, 2] = 1
        skel = torch.bmm(skel, rot_mat.unsqueeze(0).to(skel.dtype).to(skel.device))
        return skel.squeeze(0)

    def _unrotation_with_batch(self, skel, ang):
        """
        :param skel: b, 250, 3
        :param ang: b, int
        :return: b, 250, 3
        """
        if ang is None:
            return skel
        ang = -ang
        b = ang.shape[0]
        rot_mat = torch.zeros((b, 3, 3))
        rot_mat[:, 0, 0] = torch.cos(ang * self.rad)
        rot_mat[:, 0, 1] = -torch.sin(ang * self.rad)
        rot_mat[:, 1, 0] = torch.sin(ang * self.rad)
        rot_mat[:, 1, 1] = torch.cos(ang * self.rad)
        rot_mat[:, 2, 2] = 1
        rot_mat = rot_mat.to(self.device).to(skel.dtype)
        skel = torch.bmm(skel, rot_mat)
        return skel

    def _morphing(self, src, type='soft'):
        if type is 'soft':
            src_temp = rearrange(src, 'b p n c -> (b p) c n', p=self.window_size, n=self.joint_size)
            dst = self.morphing(src_temp)
            dst = rearrange(dst, '(b p) c n -> b p n c', p=self.window_size, n=self.joint_size_gt)
        elif type is 'soft2':
            dst = src[:, :, :15, :]
        elif type is 'hard':
            dst = self._hard_morphing(src)
        # morphing loss
        morp_loss = self._loss_morphing(src, dst)
        return dst, morp_loss

    def _hard_morphing(self, src, face_y=0.125, face_z=0.145):
        b = src.shape[0]
        dst = src[:, :, self.body25_to_shelf14, :]
        # get face direction
        face_dir = torch.cross((dst[:, :, 12] - dst[:, :, 14]).T, (dst[:, :, 8] - dst[:, :, 9]).T)
        face_dir_normalized = face_dir / torch.norm(face_dir)
        face_dir_normalized = rearrange(face_dir_normalized, 'c p b -> b p c')
        z_dir = torch.tensor((0., 0., 1.), requires_grad=True).to(self.device)
        z_dir = repeat(z_dir, 'c -> b n c', b=b, n=self.window_size)

        # calc TOP_HEAD(12) and BOTTOM_HEAD(13)
        head_center = (src[:, :, 17] + src[:, :, 18]) / 2.
        shoulder_center = (dst[:, :, 8] + dst[:, :, 9]) / 2.
        dst[:, :, 12] = shoulder_center + (head_center - shoulder_center) * 0.5
        dst[:, :, 13] = dst[:, :, 12] + face_dir_normalized * face_y + z_dir * face_z
        return dst

    def _loss_pcp(self, src, gt):
        """
        :param src: batch, frame, joint, (x, y, z)
        :param gt: batch, frame, joint, (x, y, z)
        :return: batch
        """
        da = self.pdist(src[:, :, self.pcp_shelf14[0], :], gt[:, :, self.pcp_shelf14[0], :])
        db = self.pdist(src[:, :, self.pcp_shelf14[1], :], gt[:, :, self.pcp_shelf14[1], :])
        l = self.pdist(gt[:, :, self.pcp_shelf14[0], :], gt[:, :, self.pcp_shelf14[1], :])
        loss = torch.mean(torch.mean(self.relu(da + db - l), dim=-1), dim=-1)
        return loss


def get_jcsat(device="cuda:0", debug=False, mode='medium', args=None):
    jc_num = 4
    if mode is 'medium':
        enc1 = JcTrajEncoder(emb_dim=128, out_dim=256,
                             mlp_depth=16, mlp_heads=16, mlp_dim_head=16, mlp_dim=256,
                             jc_num=jc_num, device=device).to(device)
        enc2 = JcStrucEncoder(emb_dim=128, out_dim=256,
                              mlp_depth=16, mlp_heads=16, mlp_dim_head=16, mlp_dim=256,
                              jc_num=jc_num, device=device).to(device)
        tmae = JcsatModel(enc1, enc2,
                          emb_dim=enc2.out_dim,
                          mlp_depth=8, mlp_heads=8, mlp_dim_head=4, mlp_dim=32,
                          debug=debug, device=device, camera_path=args.dataset_camera_path).to(device)
    elif mode is 'large':
        enc1 = JcTrajEncoder(emb_dim=256, out_dim=512,
                             mlp_depth=16, mlp_heads=16, mlp_dim_head=16, mlp_dim=512,
                             jc_num=jc_num, device=device).to(device)
        enc2 = JcStrucEncoder(emb_dim=256, out_dim=512,
                              mlp_depth=16, mlp_heads=16, mlp_dim_head=16, mlp_dim=512,
                              jc_num=jc_num, device=device).to(device)
        tmae = JcsatModel(enc1, enc2,
                          emb_dim=enc2.out_dim,
                          mlp_depth=8, mlp_heads=8, mlp_dim_head=4, mlp_dim=128,
                          debug=debug, device=device).to(device)
    else:
        assert False, 'No recognized mode: {}'.format(mode)
    return tmae
