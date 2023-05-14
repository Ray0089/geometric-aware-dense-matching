
import torch
import torch.nn as nn
import torch.nn.functional as F
import models.pytorch_utils as pt_utils
from models.loss import  FocalLoss,CircleLoss
from models.loss import AutomaticWeightedLoss
from utils.basic_utils import pdist
from models.SplineCNN import SplineCNN_Mesh as MeshEmbModel
from models.ffb6d import FFB6DEmb as PcdEmbModel


class GeoMatch(nn.Module):
    def __init__(
        self, cfg, cls_id
    ):
        super().__init__()

        self.awl = AutomaticWeightedLoss(2)
     
        # ######################## prepare stages#########################

        self.feat_dim = cfg["feat_dim"]
        self.positive_r = cfg["neighbor_dis_th"] * cfg["model_d"][cls_id] / 1000.0
        self.model_emb = MeshEmbModel(cfg, cls_id)
        self.pcd_emb = PcdEmbModel(cfg['ffb_config'])
        self.circle_loss = CircleLoss(16)
        self.ce_loss = nn.CrossEntropyLoss()
        self.seg_loss_func = FocalLoss(gamma=2)
        #self.seg_loss =torch.nn.CrossEntropyLoss()
        
        # ####################### prediction headers #############################

        self.seg_layer = (
            pt_utils.Seq(self.feat_dim)
            .conv1d(128, bn=True)
            .conv1d(128, bn=True)
            .conv1d(128, bn=True)
            .conv1d(2, activation=None)
        )
        self.feature_encoding_layer = (
            pt_utils.Seq(128)
            .conv1d(128, bn=True)
            .conv1d(128, bn=True)
            .conv1d(128, bn=True)
            .conv1d(self.feat_dim, activation=None, bias=False)
        )
        
        self.normalize_feature_layer = pt_utils.Conv1d(
                    self.feat_dim, self.feat_dim,
                    bn=True
                )


    def matching_loss(self,similarity,match_idx,mesh_xyz,vis_flag,RT):

        n_node = len(mesh_xyz)
        
        idx_in_mesh = (match_idx!=n_node)
        idx_mesh_in = torch.where(match_idx!=n_node)[0]
        idx_out_mesh = (match_idx==n_node)

        gt_pt = mesh_xyz[match_idx[idx_in_mesh]]

        valid_vis_pts = mesh_xyz[vis_flag.to(torch.bool)]
        dis_matrix = pdist(gt_pt,valid_vis_pts)
        positive_radius = self.positive_r# / 1000.0 * model_vis_pts_proj[:,2]
        #print(f"cur_r:{positive_radius}")


        pts_num,feat_dim = similarity.shape
        p_n_mask = torch.zeros((pts_num,feat_dim-1),dtype=torch.bool).cuda()
        p_n_vis_mask = torch.where(dis_matrix < positive_radius,torch.tensor(True).cuda(),torch.tensor(False).cuda())
        p_n_in_mesh = torch.index_select(p_n_mask,0,idx_mesh_in)
        p_n_in_mesh[:,vis_flag.to(torch.bool)] = p_n_vis_mask
        p_n_mask[idx_in_mesh] = p_n_in_mesh
        
        p_n_mask = torch.cat([p_n_mask,idx_out_mesh.unsqueeze(1)],dim=1)
        #print(p_n_mask.sum(dim=1).cpu().numpy())

        loss = self.circle_loss(similarity,p_n_mask, 0.2)

        return loss


    def matching_loss_sys(self,similarity,match_idx,idxs):

        sys_cor = self.model_emb.sys_idx

        pts_num,vert_num = similarity.shape
        cld_idx = torch.arange(pts_num)
        cld_idx = torch.cat((cld_idx,cld_idx),dim=0)
        selected_idx = torch.cat((match_idx[idxs],match_idx[sys_cor[idxs]]),dim=0)
        put_idx = [cld_idx,selected_idx]
        p_n_mask = torch.zeros((pts_num,vert_num),dtype=torch.bool).cuda()
        p_n_mask = p_n_mask.index_put_(put_idx,torch.tensor(True).cuda())
      
        loss = self.circle_loss(similarity,p_n_mask, 0.2)

        return loss

    def pointwise_feature_matching(self,rgbd_feature,mesh_feature,x):
        '''
        args:
        rgbd_feat:[bs,feat_dim,n_pts]
        mesh:[1,feat_dim,n_mesh_pts]
        
        '''
        #device = mesh.device
        match_loss = []
    
        batch, feat_dim, num_pt= rgbd_feature.shape
        rgbd_feature =rgbd_feature.transpose(1,2)
        #rgbd_feature = F.normalize(rgbd_feature,p=2,dim=2)
        mesh = mesh_feature[0]
        
        padding = -torch.ones((self.feat_dim,1),dtype=torch.float32).cuda()
        mesh_padded = torch.cat([mesh,padding],dim=1)
        mesh_padded = F.normalize(mesh_padded,p=2,dim=0)
      
        labels = x['labels']
        corr = x['match_idx']
        RTs = x['RT']

        for i in range(batch):
       
            idxs = torch.where(labels[i]==1)[0]         
            if len(idxs)<3:
                continue
            
            selected_cld = rgbd_feature[i].index_select(0,idxs)
            selected_corr = corr[i].index_select(0,idxs)

            selected_cld = F.normalize(selected_cld,p=2,dim=1)

            simlarity = torch.matmul(selected_cld,mesh_padded)

            if self.model_emb.sys_corr_idx is not None:
                #print('use sys match loss')
                match_loss_i = self.matching_loss_sys(simlarity,
                        corr[i].long(),idxs)
            else:
                match_loss_i = self.matching_loss(
                        simlarity,
                        selected_corr.long(),
                        self.model_emb._buffers['xyz'].contiguous(),
                        x['visible_flag'][i],
                        RTs[i],#cld_i
                        )

            match_loss.append(match_loss_i)
        if len(match_loss) == 0:
            match_loss = torch.tensor(0).cuda()
        else:   
            match_loss = torch.stack(match_loss)
            match_loss = torch.mean(match_loss)
        return match_loss

    def forward(
        self, inputs, end_points=None
    ):
        """
        Params:
        inputs: dict of :
            cld_rgb_nrm : FloatTensor [bs, 9, npts]
            labels : # [bs, npts]
            RT : [bs,3,4]
            match_idx : [bs,npts]
            visible_flag : [bs,n_mesh_pts]
        Returns:
            end_points:
        """
        # ###################### prepare stages #############################

        if not end_points:
            end_points = {}

        rgbd_emb = self.pcd_emb(inputs)
        mesh_features = self.model_emb()
        rgbd_features = self.feature_encoding_layer(rgbd_emb)
        rgbd_normalized = self.normalize_feature_layer(rgbd_features)
        rgbd_emb = rgbd_emb + rgbd_normalized
        seg_features = self.seg_layer(rgbd_emb)
        mesh_features = mesh_features.unsqueeze(0)
      
        # ###################### prediction stages #############################
        
        if self.training:
       
            match_loss = self.pointwise_feature_matching(rgbd_features,mesh_features,inputs)
            seg_loss = self.seg_loss_func(seg_features,inputs['labels'])

            end_points['loss'] = self.awl(seg_loss,match_loss)
            end_points['seg_loss'] = seg_loss
            end_points['match_loss'] = match_loss

        end_points['seg'] = seg_features
        end_points['mesh'] = mesh_features
        end_points['rgbd'] = rgbd_features
        return end_points


def main():
    from common import ConfigRandLA,ConfigRandLA3D
    rndla_cfg = ConfigRandLA
    rnd3d = ConfigRandLA3D

    n_cls = 1
    model = GeoMatch(1, rndla_cfg, rnd3d)
    print(model)

    print(
        "model parameters:", sum(param.numel() for param in model.parameters())
    )


if __name__ == "__main__":
    main()
