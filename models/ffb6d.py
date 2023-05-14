

import torch
import torch.nn as nn
from models.cnn.pspnet import psp_models
import models.pytorch_utils as pt_utils
from models.RandLA.RandLANet import Network as RandLANet


class FFB6DEmb(nn.Module):
    def __init__(
        self, cfg
    ):
        super().__init__()

        # ######################## prepare stages#########################

        cnn = psp_models['resnet18'.lower()]()
      
        rndla = RandLANet(cfg)

        self.cnn_pre_stages = nn.Sequential(
            cnn.feats.conv1,  # stride = 2, [bs, c, 240, 320]
            cnn.feats.bn1, 
            cnn.feats.relu,
            cnn.feats.maxpool  # stride = 2, [bs, 64, 120, 160]
        )
        self.rndla_pre_stages = rndla.fc0

        # ####################### downsample stages#######################
        self.cnn_ds_stages = nn.ModuleList([
            cnn.feats.layer1,    # stride = 1, [bs, 64, 120, 160]
            cnn.feats.layer2,    # stride = 2, [bs, 128, 60, 80]
            # stride = 1, [bs, 128, 60, 80]
            nn.Sequential(cnn.feats.layer3, cnn.feats.layer4),
            nn.Sequential(cnn.psp, cnn.drop_1)   # [bs, 1024, 60, 80]
        ])
        self.ds_sr = [4, 8, 8, 8]

        self.rndla_ds_stages = rndla.dilated_res_blocks

        self.ds_rgb_oc = [64, 128, 512, 1024]
        self.ds_rndla_oc = [item * 2 for item in cfg.d_out]
        self.ds_fuse_r2p_pre_layers = nn.ModuleList()
        self.ds_fuse_r2p_fuse_layers = nn.ModuleList()
        self.ds_fuse_p2r_pre_layers = nn.ModuleList()
        self.ds_fuse_p2r_fuse_layers = nn.ModuleList()
        for i in range(4):
            self.ds_fuse_r2p_pre_layers.append(
                pt_utils.Conv2d(
                    self.ds_rgb_oc[i], self.ds_rndla_oc[i], kernel_size=(1, 1),
                    bn=True
                )
            )
            self.ds_fuse_r2p_fuse_layers.append(
                pt_utils.Conv2d(
                    self.ds_rndla_oc[i]*2, self.ds_rndla_oc[i], kernel_size=(1, 1),
                    bn=True
                )
            )

            self.ds_fuse_p2r_pre_layers.append(
                pt_utils.Conv2d(
                    self.ds_rndla_oc[i], self.ds_rgb_oc[i], kernel_size=(1, 1),
                    bn=True
                )
            )
            self.ds_fuse_p2r_fuse_layers.append(
                pt_utils.Conv2d(
                    self.ds_rgb_oc[i]*2, self.ds_rgb_oc[i], kernel_size=(1, 1),
                    bn=True
                )
            )

        # ###################### upsample stages #############################
        self.cnn_up_stages = nn.ModuleList([
            nn.Sequential(cnn.up_1, cnn.drop_2),  # [bs, 256, 120, 160]
            nn.Sequential(cnn.up_2, cnn.drop_2),  # [bs, 64, 240, 320]
            nn.Sequential(cnn.final),  # [bs, 64, 240, 320]
            nn.Sequential(cnn.up_3, cnn.final)  # [bs, 64, 480, 640]
        ])
        self.up_rgb_oc = [256, 64, 64]
        self.up_rndla_oc = []
        for j in range(cfg.num_layers):
            if j < 3:
                self.up_rndla_oc.append(self.ds_rndla_oc[-j-2])
            else:
                self.up_rndla_oc.append(self.ds_rndla_oc[0])

        self.rndla_up_stages = rndla.decoder_blocks

        n_fuse_layer = 3
        self.up_fuse_r2p_pre_layers = nn.ModuleList()
        self.up_fuse_r2p_fuse_layers = nn.ModuleList()
        self.up_fuse_p2r_pre_layers = nn.ModuleList()
        self.up_fuse_p2r_fuse_layers = nn.ModuleList()
        for i in range(n_fuse_layer):
            self.up_fuse_r2p_pre_layers.append(
                pt_utils.Conv2d(
                    self.up_rgb_oc[i], self.up_rndla_oc[i], kernel_size=(1, 1),
                    bn=True
                )
            )
            self.up_fuse_r2p_fuse_layers.append(
                pt_utils.Conv2d(
                    self.up_rndla_oc[i]*2, self.up_rndla_oc[i], kernel_size=(1, 1),
                    bn=True
                )
            )

            self.up_fuse_p2r_pre_layers.append(
                pt_utils.Conv2d(
                    self.up_rndla_oc[i], self.up_rgb_oc[i], kernel_size=(1, 1),
                    bn=True
                )
            )
            self.up_fuse_p2r_fuse_layers.append(
                pt_utils.Conv2d(
                    self.up_rgb_oc[i]*2, self.up_rgb_oc[i], kernel_size=(1, 1),
                    bn=True
                )
            )

        # ####################### prediction headers #############################
        # We use 3D keypoint prediction header for pose estimation following PVN3D
        # You can use different prediction headers for different downstream tasks.

    @staticmethod
    def random_sample(feature, pool_idx):
        """
        :param feature: [B, N, d] input features matrix
        :param pool_idx: [B, N', max_num] N' < N, N' is the selected position after pooling
        :return: pool_features = [B, N', d] pooled features matrix
        """
        if len(feature.size()) > 3:
            feature = feature.squeeze(dim=3)  # batch*channel*npoints
        num_neigh = pool_idx.shape[-1]
        d = feature.shape[1]
        batch_size = pool_idx.shape[0]
        pool_idx = pool_idx.reshape(batch_size, -1)  # batch*(npoints,nsamples)
        pool_features = torch.gather(
            feature, 2, pool_idx.unsqueeze(1).repeat(1, feature.shape[1], 1)
        ).contiguous()
        pool_features = pool_features.reshape(batch_size, d, -1, num_neigh)
        pool_features = pool_features.max(dim=3, keepdim=True)[0]  # batch*channel*npoints*1
        return pool_features

    @staticmethod
    def nearest_interpolation(feature, interp_idx):
        """
        :param feature: [B, N, d] input features matrix
        :param interp_idx: [B, up_num_points, 1] nearest neighbour index
        :return: [B, up_num_points, d] interpolated features matrix
        """
        feature = feature.squeeze(dim=3)  # batch*channel*npoints
        batch_size = interp_idx.shape[0]
        up_num_points = interp_idx.shape[1]
        interp_idx = interp_idx.reshape(batch_size, up_num_points)
        interpolated_features = torch.gather(
            feature, 2, interp_idx.unsqueeze(1).repeat(1, feature.shape[1], 1)
        ).contiguous()
        interpolated_features = interpolated_features.unsqueeze(3)  # batch*channel*npoints*1
        return interpolated_features

    def _break_up_pc(self, pc):
        xyz = pc[:, :3, :].transpose(1, 2).contiguous()
        features = (
            pc[:, 3:, :].contiguous() if pc.size(1) > 3 else None
        )
        return xyz, features

    def forward(
        self, inputs, end_points=None
    ):
        """
        Params:
        inputs: dict of :
            rgb         : FloatTensor [bs, 3, h, w]
            dpt_nrm     : FloatTensor [bs, 6, h, w], 3c xyz in meter + 3c normal map
            cld_rgb_nrm : FloatTensor [bs, 9, npts]
            choose      : LongTensor [bs, 1, npts]
            xmap, ymap: [bs, h, w]
            K:          [bs, 3, 3]
        Returns:
            end_points:
        """
        # ###################### prepare stages #############################
        if not end_points:
            end_points = {}
        # ResNet pre + layer1 + layer2
        rgb_emb = self.cnn_pre_stages(inputs['rgb'])  # stride = 2, [bs, c, 240, 320]
        # rndla pre
        #_, p_emb = self._break_up_pc(inputs['cld_rgb_nrm'])
        p_emb = inputs['cld_rgb_nrm']
        p_emb = self.rndla_pre_stages(p_emb)
        p_emb = p_emb.unsqueeze(dim=3)  # Batch*channel*npoints*1

        # ###################### encoding stages #############################
        ds_emb = []
        for i_ds in range(4):
            # encode rgb downsampled feature
            rgb_emb0 = self.cnn_ds_stages[i_ds](rgb_emb)
            bs, c, hr, wr = rgb_emb0.size()

            # encode point cloud downsampled feature
            f_encoder_i = self.rndla_ds_stages[i_ds](
                p_emb, inputs['cld_xyz%d' % i_ds], inputs['cld_nei_idx%d' % i_ds]
            )
            f_sampled_i = self.random_sample(f_encoder_i, inputs['cld_sub_idx%d' % i_ds])
            p_emb0 = f_sampled_i
            if i_ds == 0:
                ds_emb.append(f_encoder_i)

            # fuse point feauture to rgb feature
            p2r_emb = self.ds_fuse_p2r_pre_layers[i_ds](p_emb0)
            p2r_emb = self.nearest_interpolation(
                p2r_emb, inputs['p2r_ds_nei_idx%d' % i_ds]
            )
            p2r_emb = p2r_emb.view(bs, -1, hr, wr)
            rgb_emb = self.ds_fuse_p2r_fuse_layers[i_ds](
                torch.cat((rgb_emb0, p2r_emb), dim=1)
            )

            # fuse rgb feature to point feature
            r2p_emb = self.random_sample(
                rgb_emb0.reshape(bs, c, hr*wr, 1), inputs['r2p_ds_nei_idx%d' % i_ds]
            ).view(bs, c, -1, 1)
            r2p_emb = self.ds_fuse_r2p_pre_layers[i_ds](r2p_emb)
            p_emb = self.ds_fuse_r2p_fuse_layers[i_ds](
                torch.cat((p_emb0, r2p_emb), dim=1)
            )
            ds_emb.append(p_emb)

        # ###################### decoding stages #############################
        n_up_layers = len(self.rndla_up_stages)
        for i_up in range(n_up_layers-1):
            # decode rgb upsampled feature
            rgb_emb0 = self.cnn_up_stages[i_up](rgb_emb)
            bs, c, hr, wr = rgb_emb0.size()

            # decode point cloud upsampled feature
            f_interp_i = self.nearest_interpolation(
                p_emb, inputs['cld_interp_idx%d' % (n_up_layers-i_up-1)]
            )
            f_decoder_i = self.rndla_up_stages[i_up](
                torch.cat([ds_emb[-i_up - 2], f_interp_i], dim=1)
            )
            p_emb0 = f_decoder_i

            # fuse point feauture to rgb feature
            p2r_emb = self.up_fuse_p2r_pre_layers[i_up](p_emb0)
            p2r_emb = self.nearest_interpolation(
                p2r_emb, inputs['p2r_up_nei_idx%d' % i_up]
            )
            p2r_emb = p2r_emb.view(bs, -1, hr, wr)
            rgb_emb = self.up_fuse_p2r_fuse_layers[i_up](
                torch.cat((rgb_emb0, p2r_emb), dim=1)
            )

            # fuse rgb feature to point feature
            r2p_emb = self.random_sample(
                rgb_emb0.reshape(bs, c, hr*wr), inputs['r2p_up_nei_idx%d' % i_up]
            ).view(bs, c, -1, 1)
            r2p_emb = self.up_fuse_r2p_pre_layers[i_up](r2p_emb)
            p_emb = self.up_fuse_r2p_fuse_layers[i_up](
                torch.cat((p_emb0, r2p_emb), dim=1)
            )

        # final upsample layers:
        rgb_emb = self.cnn_up_stages[n_up_layers-1](rgb_emb)
        f_interp_i = self.nearest_interpolation(
            p_emb, inputs['cld_interp_idx%d' % (0)]
        )
        p_emb = self.rndla_up_stages[n_up_layers-1](
            torch.cat([ds_emb[0], f_interp_i], dim=1)
        ).squeeze(-1)

        bs, di, _, _ = rgb_emb.size()
        rgb_emb_c = rgb_emb.view(bs, di, -1)
        choose_emb = inputs['choose'].repeat(1, di, 1)
        rgb_emb_c = torch.gather(rgb_emb_c, 2, choose_emb).contiguous()

        # Use simple concatenation. Good enough for fully fused RGBD feature.
        rgbd_emb = torch.cat([rgb_emb_c, p_emb], dim=1)
        return rgbd_emb


