import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import models.RandLA.pytorch_utils as pt_utils
from helper_tool import DataProcessing as DP
from utils.ply import load_ply, normalize_color,normalize_pts
import numpy as np
from sklearn.metrics import confusion_matrix

class Network(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        

        self.fc0 = pt_utils.Conv1d(config.in_c, 8, kernel_size=1, bn=True)

        self.dilated_res_blocks = nn.ModuleList()
        d_in = 8
        for i in range(self.config.num_layers):
            d_out = self.config.d_out[i]
            self.dilated_res_blocks.append(Dilated_res_block(d_in, d_out))
            d_in = 2 * d_out

        d_out = d_in
        self.decoder_0 = pt_utils.Conv2d(d_in, d_out, kernel_size=(1,1), bn=True)

        self.decoder_blocks = nn.ModuleList()
        for j in range(self.config.num_layers):
            if j < 3:
                d_in = d_out + 2 * self.config.d_out[-j-2]
                d_out = 2 * self.config.d_out[-j-2]
            else:
                d_in = 4 * self.config.d_out[-4]
                d_out = 2 * self.config.d_out[-4]
            self.decoder_blocks.append(pt_utils.Conv2d(d_in, d_out, kernel_size=(1,1), bn=True))

        # self.fc1 = pt_utils.Conv2d(d_out, 128, kernel_size=(1,1), bn=True)
        # self.fc2 = pt_utils.Conv2d(128, 64, kernel_size=(1,1), bn=True)
        # #self.dropout = nn.Dropout(0.5)
        # self.fc3 = pt_utils.Conv2d(64, 64, kernel_size=(1,1), bn=True)

    def forward(self, end_points):

        features = end_points['features']  # Batch*channel*npoints
        features = self.fc0(features)

        features = features.unsqueeze(dim=3)  # Batch*channel*npoints*1

        # ###########################Encoder############################
        f_encoder_list = []
        for i in range(self.config.num_layers):
            f_encoder_i = self.dilated_res_blocks[i](
                features, end_points['xyz'][i], end_points['neigh_idx'][i]
            )

            f_sampled_i = self.random_sample(f_encoder_i, end_points['sub_idx'][i])
            features = f_sampled_i
            print("encoder%d:"%i, features.size())
            if i == 0:
                f_encoder_list.append(f_encoder_i)
            f_encoder_list.append(f_sampled_i)
        # ###########################Encoder############################

        features = self.decoder_0(f_encoder_list[-1])

        # ###########################Decoder############################
        f_decoder_list = []
        for j in range(self.config.num_layers):
            f_interp_i = self.nearest_interpolation(features, end_points['interp_idx'][-j - 1])
            f_decoder_i = self.decoder_blocks[j](torch.cat([f_encoder_list[-j - 2], f_interp_i], dim=1))

            features = f_decoder_i
            print("decoder%d:"%j, features.size())
            f_decoder_list.append(f_decoder_i)
        # ###########################Decoder############################

        # features = self.fc1(features)
        # features = self.fc2(features)
        # #features = self.dropout(features)
        # features = self.fc3(features)
        f_out = features.squeeze(3)

        end_points['logits'] = f_out
        return end_points

    @staticmethod
    def random_sample(feature, pool_idx):
        """
        :param feature: [B, N, d] input features matrix
        :param pool_idx: [B, N', max_num] N' < N, N' is the selected position after pooling
        :return: pool_features = [B, N', d] pooled features matrix
        """
        feature = feature.squeeze(dim=3)  # batch*channel*npoints
        num_neigh = pool_idx.shape[-1]
        d = feature.shape[1]
        batch_size = pool_idx.shape[0]
        pool_idx = pool_idx.reshape(batch_size, -1)  # batch*(npoints,nsamples)
        pool_features = torch.gather(feature, 2, pool_idx.unsqueeze(1).repeat(1, feature.shape[1], 1))
        pool_features = pool_features.reshape(batch_size, d, -1, num_neigh)
        pool_features = pool_features.max(dim=3, keepdim=True)[0]  # batch*channel*npoints*1
        return pool_features

    @staticmethod
    def nearest_interpolation(feature, interp_idx):
        """
        :param feature: [B, C, npoints] input features matrix
        :param interp_idx: [B, up_num_points, 1] nearest neighbour index
        :return: [B, c, up_num_points, 1] interpolated features matrix
        """
        feature = feature.squeeze(dim=3)  # batch*channel*npoints
        batch_size = interp_idx.shape[0]
        up_num_points = interp_idx.shape[1]
        interp_idx = interp_idx.reshape(batch_size, up_num_points)
        interpolated_features = torch.gather(feature, 2, interp_idx.unsqueeze(1).repeat(1,feature.shape[1],1))
        interpolated_features = interpolated_features.unsqueeze(3)  # batch*channel*npoints*1
        return interpolated_features

class RandLA3D(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model3d_pth = config.model_path
        self.n_pt = config.num_points
        self.meshes = self.load_meshes()
        self.model_to_tensor()

        self.fc0 = pt_utils.Conv1d(config.in_c, 8, kernel_size=1, bn=True)

        self.dilated_res_blocks = nn.ModuleList()
        d_in = 8
        for i in range(self.config.num_layers):
            d_out = self.config.d_out[i]
            self.dilated_res_blocks.append(Dilated_res_block(d_in, d_out))
            d_in = 2 * d_out

        d_out = d_in
        self.decoder_0 = pt_utils.Conv2d(d_in, d_out, kernel_size=(1,1), bn=True)

        self.decoder_blocks = nn.ModuleList()
        for j in range(self.config.num_layers):
            if j < 3:
                d_in = d_out + 2 * self.config.d_out[-j-2]
                d_out = 2 * self.config.d_out[-j-2]
            else:
                d_in = 4 * self.config.d_out[-4]
                d_out = 2 * self.config.d_out[-4]
            self.decoder_blocks.append(pt_utils.Conv2d(d_in, d_out, kernel_size=(1,1), bn=True))

        self.transformation_list = nn.ModuleList()
        for i in range(21):
            self.transformation_list.append(nn.Sequential(
            pt_utils.Conv2d(d_out, 64, kernel_size=(1,1), bn=True),
            pt_utils.Conv2d(64, 64, kernel_size=(1,1), bn=True),
            pt_utils.Conv2d(64, self.config.feature_dim, kernel_size=(1,1), activation=None,bias=False)
        ))

        #self.fc1 = pt_utils.Conv2d(d_out, 64, kernel_size=(1,1), bn=True)
        #self.dropout = nn.Dropout(0.5)
        #self.fc2 = pt_utils.Conv2d(64, 64, kernel_size=(1,1), bn=True)
        
        #self.fc3 = pt_utils.Conv2d(64, self.config.feature_dim, kernel_size=(1,1), activation=None)


    def forward(self,valid_idx):

        if self.training:

            ids = self.find_valid_ids(valid_idx)
        else:
            ids = [i for i in range(21)]

        features = self.mesh_features[ids].transpose(2,1)  # Batch*channel*npoints
        features = self.fc0(features)

        features = features.unsqueeze(dim=3)  # Batch*channel*npoints*1

        # ###########################Encoder############################
        f_encoder_list = []
        names = self.__dict__['_buffers']
        for i in range(self.config.num_layers):
            f_encoder_i = self.dilated_res_blocks[i](
                features, names[f'xyz_{i}'][ids], names[f'neigh_idx_{i}'][ids]
            )

            f_sampled_i = self.random_sample(f_encoder_i, names[f'sub_idx_{i}'][ids])
            features = f_sampled_i
            #print("encoder%d:"%i, features.size())
            if i == 0:
                f_encoder_list.append(f_encoder_i)
            f_encoder_list.append(f_sampled_i)
        # ###########################Encoder############################

        features = self.decoder_0(f_encoder_list[-1])

        # ###########################Decoder############################
        #f_decoder_list = []
        for j in range(self.config.num_layers):
            f_interp_i = self.nearest_interpolation(features, names[f'interp_idx_{4-j-1}'][ids])
            f_decoder_i = self.decoder_blocks[j](torch.cat([f_encoder_list[-j - 2], f_interp_i], dim=1))

            features = f_decoder_i
            #print("decoder%d:"%j, features.size())
            #f_decoder_list.append(f_decoder_i)
        # ###########################Decoder############################
        feature_list = []
        for i,idx in enumerate(ids):
            feature_list.append(self.transformation_list[idx](features[i].unsqueeze(0)))

        f_out = torch.stack(feature_list).squeeze(1).squeeze(3)


        #features = self.fc1(features)
        
        #features = self.dropout(features)
        #features = self.fc2(features)
        #features = self.fc3(features)
        #f_out = f_out.squeeze(3)

        
        return f_out

    def find_valid_ids(self,valid_ids):
        b_size, _ = valid_ids.shape
        ids = []
        for i in range(b_size):
            j = 0
            idx = valid_ids[i,j].item()
            while idx>=0:

                if idx not in ids:
                    ids.append(idx)
                j+=1
                idx = valid_ids[i,j].item()
        ids.sort()
        ids = [i - 1 for i in ids]
        return ids



    def load_meshes(self):

        pth = self.model3d_pth
        model_pth = os.path.join(pth,'models')
        file_list = os.listdir(model_pth)
        file_list.sort()
        
        meshes = {}
        mesh_xyz_list = []
        mesh_feature_list = []
        size_list = []
        resized_xyz_list = []
        n = self.n_pt

        for cls_name in file_list:
            

            data = np.load(os.path.join(model_pth,cls_name,'fp.npy'))
            pts = data[:n,:3].astype(np.float32)
            rgb = data[:n,3:6].astype(np.float32)
            nrm = data[:n,6:].astype(np.float32)
            rgb = normalize_color(rgb)
            normalized_pts,size = normalize_pts(pts)

            pts_rgb_nrm = np.concatenate((normalized_pts,rgb,nrm),axis=1)

            mesh_xyz_list.append(pts.copy())
            mesh_feature_list.append(pts_rgb_nrm.copy())
     
        mesh_xyz = np.stack(mesh_xyz_list)
        mesh_feature = np.stack(mesh_feature_list)
        
        meshes['neigh'] = DP.knn_search(mesh_xyz,mesh_xyz, 16)
        meshes['xyz'] = mesh_xyz
        #mesh_feature = self.position_encoding(mesh_feature)
        meshes['features'] = mesh_feature
        

   
        return meshes
    
    def position_encoding(self,mesh_feature):
        cls_num, npoints,dim = mesh_feature.shape
        pos_encoding = np.zeros((cls_num,npoints,30))
        color_encoding = np.zeros((cls_num,npoints,18))
        normals = mesh_feature[:,:,6:]
        for i in range(5):
            pos_encoding[:,:,i*2] = np.sin(2**i*np.pi * mesh_feature[:,:,0])
            pos_encoding[:,:,i*2+1] = np.cos(2**i*np.pi * mesh_feature[:,:,0])

            pos_encoding[:,:,10+i*2] = np.sin(2**i*np.pi * mesh_feature[:,:,1])
            pos_encoding[:,:,10+i*2+1] = np.cos(2**i*np.pi * mesh_feature[:,:,1])

            pos_encoding[:,:,20+i*2] = np.sin(2**i*np.pi * mesh_feature[:,:,2])
            pos_encoding[:,:,20+i*2+1] = np.cos(2**i*np.pi * mesh_feature[:,:,2])

        for i in range(3):
            color_encoding[:,:,i*2] = np.sin(2**i*np.pi * mesh_feature[:,:,3])
            color_encoding[:,:,i*2+1] = np.cos(2**i*np.pi * mesh_feature[:,:,3])

            color_encoding[:,:,6+i*2] = np.sin(2**i*np.pi * mesh_feature[:,:,4])
            color_encoding[:,:,6+i*2+1] = np.cos(2**i*np.pi * mesh_feature[:,:,4])

            color_encoding[:,:,12+i*2] = np.sin(2**i*np.pi * mesh_feature[:,:,5])
            color_encoding[:,:,12+i*2+1] = np.cos(2**i*np.pi * mesh_feature[:,:,5])
        mesh_feature = np.concatenate((pos_encoding,color_encoding,normals),axis=2)
        return mesh_feature

    def model_to_tensor(self):

        batch_features = self.meshes['features']
        # idx = [i for i in range(batch_features.shape[1])]
        # random.shuffle(idx)
        batch_pc = self.meshes['xyz']
        input_points = []
        input_neighbors = []
        input_pools = []
        input_up_samples = []


        neighbour_idx = self.meshes['neigh']

        for i in range(4):
            # if i == 0:
            #     idx = idx[:batch_pc.shape[1] // ycb_m_cfg.sub_sampling_ratio[i]]
            #     sub_points = batch_pc[:, idx, :]
            #     pool_i = neighbour_idx[:, idx, :]
            # else:
            sub_points = batch_pc[:, :batch_pc.shape[1] // self.config.sub_sampling_ratio[i], :]
            pool_i = neighbour_idx[:, :batch_pc.shape[1] // self.config.sub_sampling_ratio[i], :]

            
            # unfitted_pts = project(sub_points[1],Ks[0],unify_RT)
            # show_pts = np.zeros((480,640),dtype=np.uint8)

            # for j in range(unfitted_pts.shape[0]):
            #     show_pts[int(unfitted_pts[j,1]),int(unfitted_pts[j,0])] = 255
            # cv2.imwrite(f'sampled_cld/downsample_mesh_{i}.jpg',show_pts)

            up_i = DP.knn_search(sub_points, batch_pc, 1)
            input_points.append(batch_pc)
            input_neighbors.append(neighbour_idx)
            input_pools.append(pool_i)
            input_up_samples.append(up_i)
            batch_pc = sub_points
            neighbour_idx = DP.knn_search(batch_pc,batch_pc,self.config.k_n)


        
        self.meshes['xyz'] = input_points
        self.meshes['neighbor'] = input_neighbors
        self.meshes['pools'] = input_pools
        self.meshes['upsample'] = input_up_samples


        #batch_meshes = {}
        #batch_meshes['xyz'] = []
        for i,tmp in enumerate(self.meshes['xyz']):
            self.register_buffer(f'xyz_{i}',torch.from_numpy(tmp).float())
            #batch_meshes['xyz'].append(torch.from_numpy(tmp).float())
        #batch_meshes['neigh_idx'] = []
        for i,tmp in enumerate(self.meshes['neighbor']):
            self.register_buffer(f'neigh_idx_{i}',torch.from_numpy(tmp).long())
            #batch_meshes['neigh_idx'].append(torch.from_numpy(tmp).long())
        #batch_meshes['sub_idx'] = []
        for i,tmp in enumerate(self.meshes['pools']):
            self.register_buffer(f'sub_idx_{i}',torch.from_numpy(tmp).long())
            #batch_meshes['sub_idx'].append(torch.from_numpy(tmp).long())
        #batch_meshes['interp_idx'] = []
        for i,tmp in enumerate(self.meshes['upsample']):
            self.register_buffer(f'interp_idx_{i}',torch.from_numpy(tmp).long())
            #batch_meshes['interp_idx']interp_idx.append(torch.from_numpy(tmp).long())

        self.register_buffer('norm_xyz',torch.from_numpy(self.meshes['xyz'][0]).float())

        self.register_buffer('mesh_features',torch.from_numpy(self.meshes['features']).float())
        
        #batch_meshes['features'] = torch.from_numpy(mesh_features).float()

        

        
    @staticmethod
    def random_sample(feature, pool_idx):
        """
        :param feature: [B, N, d] input features matrix
        :param pool_idx: [B, N', max_num] N' < N, N' is the selected position after pooling
        :return: pool_features = [B, N', d] pooled features matrix
        """
        feature = feature.squeeze(dim=3)  # batch*channel*npoints
        num_neigh = pool_idx.shape[-1]
        d = feature.shape[1]
        batch_size = pool_idx.shape[0]
        pool_idx = pool_idx.reshape(batch_size, -1)  # batch*(npoints,nsamples)
        pool_features = torch.gather(feature, 2, pool_idx.unsqueeze(1).repeat(1, feature.shape[1], 1))
        pool_features = pool_features.reshape(batch_size, d, -1, num_neigh)
        pool_features = pool_features.max(dim=3, keepdim=True)[0]  # batch*channel*npoints*1
        return pool_features

    @staticmethod
    def nearest_interpolation(feature, interp_idx):
        """
        :param feature: [B, C, npoints] input features matrix
        :param interp_idx: [B, up_num_points, 1] nearest neighbour index
        :return: [B, c, up_num_points, 1] interpolated features matrix
        """
        feature = feature.squeeze(dim=3)  # batch*channel*npoints
        batch_size = interp_idx.shape[0]
        up_num_points = interp_idx.shape[1]
        interp_idx = interp_idx.reshape(batch_size, up_num_points)
        interpolated_features = torch.gather(feature, 2, interp_idx.unsqueeze(1).repeat(1,feature.shape[1],1))
        interpolated_features = interpolated_features.unsqueeze(3)  # batch*channel*npoints*1
        return interpolated_features
        

class RandLA3DSingle(nn.Module):
    def __init__(self, config,idx):
        super().__init__()
        self.config = config
        self.model3d_pth = config.model_path
        self.meshes = self.load_meshes(idx)
        self.model_to_tensor()
        

        self.fc0 = pt_utils.Conv1d(config.in_c, 8, kernel_size=1, bn=True)

        self.dilated_res_blocks = nn.ModuleList()
        d_in = 8
        for i in range(self.config.num_layers):
            d_out = self.config.d_out[i]
            self.dilated_res_blocks.append(Dilated_res_block(d_in, d_out))
            d_in = 2 * d_out

        d_out = d_in
        self.decoder_0 = pt_utils.Conv2d(d_in, d_out, kernel_size=(1,1), bn=True)

        self.decoder_blocks = nn.ModuleList()
        for j in range(self.config.num_layers):
            if j < 3:
                d_in = d_out + 2 * self.config.d_out[-j-2]
                d_out = 2 * self.config.d_out[-j-2]
            else:
                d_in = 4 * self.config.d_out[-4]
                d_out = 2 * self.config.d_out[-4]
            self.decoder_blocks.append(pt_utils.Conv2d(d_in, d_out, kernel_size=(1,1), bn=True))

        self.fc1 = pt_utils.Conv2d(d_out, 128, kernel_size=(1,1), bn=True)
        self.fc2 = pt_utils.Conv2d(128, 64, kernel_size=(1,1), bn=True)
        self.dropout = nn.Dropout(0.5)
        self.fc3 = pt_utils.Conv2d(64, self.config.feature_dim, kernel_size=(1,1), activation=None)

    def load_meshes(self,idx):

        pth = self.model3d_pth
        file_list = os.listdir(self.model3d_pth)
        file_list.sort()
        file_name = file_list[idx]
        mesh_xyz_list = []
        mesh_feature_list = []
        meshes = {}
        size_list = []
        resized_xyz_list = []
        #face_list = []

        mesh = load_ply(os.path.join(pth,file_name,'poisson.ply'))
        n,_ = mesh['pts'].shape
        pts = np.zeros((8192,3),np.float32)
        normals = np.zeros((8192,3),np.float32)
        rgb = np.zeros((8192,3),np.float32)
        normalized_pts = np.zeros((8192,3),np.float32)
        
        if n >= 8192:
            pts = mesh['pts'][:8192]
            normals = mesh['normals'][:8192]
            rgb = mesh['colors'][:8192]
        else:
            pts[:n,:] = mesh['pts']
            normals[:n,:] = mesh['normals']
            rgb[:n,:] = mesh['colors']
        normalized_pts,size = normalize_pts(pts)
        resized_xyz_list.append(normalized_pts)
        size_list.append(size.copy())
        rgb = normalize_color(rgb)
        pts_rgb_nrm = np.concatenate((pts,rgb,normals),axis=1)
        mesh_xyz_list.append(pts)
        mesh_feature_list.append(pts_rgb_nrm)
     
        mesh_xyz = np.stack(mesh_xyz_list)
        mesh_feature = np.stack(mesh_feature_list)
        resized_xyz = np.stack(resized_xyz_list) / max(size_list) 
        mesh_feature[:,:,:3] = resized_xyz
        meshes['neigh'] = DP.knn_search(mesh_xyz,mesh_xyz, 16)
        meshes['xyz'] = mesh_xyz
        meshes['features'] = mesh_feature
        return meshes

    def model_to_tensor(self):

        input_points = []
        input_neighbors = []
        input_pools = []
        input_up_samples = []
        batch_pc = self.meshes['xyz']
        neighbour_idx = self.meshes['neigh']

        for i in range(4):
            sub_points = batch_pc[:, :batch_pc.shape[1] // self.config.sub_sampling_ratio[i], :]
            pool_i = neighbour_idx[:, :batch_pc.shape[1] // self.config.sub_sampling_ratio[i], :]

            up_i = DP.knn_search(sub_points, batch_pc, 1)
            input_points.append(batch_pc)
            input_neighbors.append(neighbour_idx)
            input_pools.append(pool_i)
            input_up_samples.append(up_i)
            batch_pc = sub_points
            neighbour_idx = DP.knn_search(batch_pc,batch_pc,self.config.k_n)

        self.meshes['xyz'] = input_points
        self.meshes['neighbor'] = input_neighbors
        self.meshes['pools'] = input_pools
        self.meshes['upsample'] = input_up_samples

        for i,tmp in enumerate(self.meshes['xyz']):
            self.register_buffer(f'xyz_{i}',torch.from_numpy(tmp).float())

        for i,tmp in enumerate(self.meshes['neighbor']):
            self.register_buffer(f'neigh_idx_{i}',torch.from_numpy(tmp).long())

        for i,tmp in enumerate(self.meshes['pools']):
            self.register_buffer(f'sub_idx_{i}',torch.from_numpy(tmp).long())

        for i,tmp in enumerate(self.meshes['upsample']):
            self.register_buffer(f'interp_idx_{i}',torch.from_numpy(tmp).long())

        self.register_buffer('mesh_features',torch.from_numpy(self.meshes['features']).float())
              
    def forward(self):

        features = self.mesh_features.transpose(2,1)  # Batch*channel*npoints
        features = self.fc0(features)

        features = features.unsqueeze(dim=3)  # Batch*channel*npoints*1

        # ###########################Encoder############################
        f_encoder_list = []
        names = self.__dict__['_buffers']
        for i in range(self.config.num_layers):
            f_encoder_i = self.dilated_res_blocks[i](
                features, names[f'xyz_{i}'], names[f'neigh_idx_{i}']
            )

            f_sampled_i = self.random_sample(f_encoder_i, names[f'sub_idx_{i}'])
            features = f_sampled_i
            #print("encoder%d:"%i, features.size())
            if i == 0:
                f_encoder_list.append(f_encoder_i)
            f_encoder_list.append(f_sampled_i)
        # ###########################Encoder############################

        features = self.decoder_0(f_encoder_list[-1])

        # ###########################Decoder############################
        f_decoder_list = []
        for j in range(self.config.num_layers):
            f_interp_i = self.nearest_interpolation(features, names[f'interp_idx_{4-j-1}'])
            f_decoder_i = self.decoder_blocks[j](torch.cat([f_encoder_list[-j - 2], f_interp_i], dim=1))

            features = f_decoder_i
            #print("decoder%d:"%j, features.size())
            f_decoder_list.append(f_decoder_i)
        # ###########################Decoder############################

        features = self.fc1(features)
        
        
        features = self.fc2(features)
        features = self.dropout(features)
        features = self.fc3(features)
        f_out = features.squeeze(3)

        
        return f_out



   
    @staticmethod
    def random_sample(feature, pool_idx):
        """
        :param feature: [B, N, d] input features matrix
        :param pool_idx: [B, N', max_num] N' < N, N' is the selected position after pooling
        :return: pool_features = [B, N', d] pooled features matrix
        """
        feature = feature.squeeze(dim=3)  # batch*channel*npoints
        num_neigh = pool_idx.shape[-1]
        d = feature.shape[1]
        batch_size = pool_idx.shape[0]
        pool_idx = pool_idx.reshape(batch_size, -1)  # batch*(npoints,nsamples)
        pool_features = torch.gather(feature, 2, pool_idx.unsqueeze(1).repeat(1, feature.shape[1], 1))
        pool_features = pool_features.reshape(batch_size, d, -1, num_neigh)
        pool_features = pool_features.max(dim=3, keepdim=True)[0]  # batch*channel*npoints*1
        return pool_features

    @staticmethod
    def nearest_interpolation(feature, interp_idx):
        """
        :param feature: [B, C, npoints] input features matrix
        :param interp_idx: [B, up_num_points, 1] nearest neighbour index
        :return: [B, c, up_num_points, 1] interpolated features matrix
        """
        feature = feature.squeeze(dim=3)  # batch*channel*npoints
        batch_size = interp_idx.shape[0]
        up_num_points = interp_idx.shape[1]
        interp_idx = interp_idx.reshape(batch_size, up_num_points)
        interpolated_features = torch.gather(feature, 2, interp_idx.unsqueeze(1).repeat(1,feature.shape[1],1))
        interpolated_features = interpolated_features.unsqueeze(3)  # batch*channel*npoints*1
        return interpolated_features
        



def compute_acc(end_points):

    logits = end_points['valid_logits']
    labels = end_points['valid_labels']
    logits = logits.max(dim=1)[1]
    acc = (logits == labels).sum().float() / float(labels.shape[0])
    end_points['acc'] = acc
    return acc, end_points


class IoUCalculator:
    def __init__(self, cfg):
        self.gt_classes = [0 for _ in range(cfg.num_classes)]
        self.positive_classes = [0 for _ in range(cfg.num_classes)]
        self.true_positive_classes = [0 for _ in range(cfg.num_classes)]
        self.cfg = cfg

    def add_data(self, end_points):
        logits = end_points['valid_logits']
        labels = end_points['valid_labels']
        pred = logits.max(dim=1)[1]
        pred_valid = pred.detach().cpu().numpy()
        labels_valid = labels.detach().cpu().numpy()

        val_total_correct = 0
        val_total_seen = 0

        correct = np.sum(pred_valid == labels_valid)
        val_total_correct += correct
        val_total_seen += len(labels_valid)

        conf_matrix = confusion_matrix(labels_valid, pred_valid, np.arange(0, self.cfg.num_classes, 1))
        self.gt_classes += np.sum(conf_matrix, axis=1)
        self.positive_classes += np.sum(conf_matrix, axis=0)
        self.true_positive_classes += np.diagonal(conf_matrix)

    def compute_iou(self):
        iou_list = []
        for n in range(0, self.cfg.num_classes, 1):
            if float(self.gt_classes[n] + self.positive_classes[n] - self.true_positive_classes[n]) != 0:
                iou = self.true_positive_classes[n] / float(self.gt_classes[n] + self.positive_classes[n] - self.true_positive_classes[n])
                iou_list.append(iou)
            else:
                iou_list.append(0.0)
        mean_iou = sum(iou_list) / float(self.cfg.num_classes)
        return mean_iou, iou_list



class Dilated_res_block(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()

        self.mlp1 = pt_utils.Conv2d(d_in, d_out//2, kernel_size=(1,1), bn=True)
        self.lfa = Building_block(d_out)
        self.mlp2 = pt_utils.Conv2d(d_out, d_out*2, kernel_size=(1, 1), bn=True, activation=None)
        self.shortcut = pt_utils.Conv2d(d_in, d_out*2, kernel_size=(1,1), bn=True, activation=None)

    def forward(self, feature, xyz, neigh_idx):
        f_pc = self.mlp1(feature)  # Batch*channel*npoints*1
        f_pc = self.lfa(xyz, f_pc, neigh_idx)  # Batch*d_out*npoints*1
        f_pc = self.mlp2(f_pc)
        shortcut = self.shortcut(feature)
        return F.leaky_relu(f_pc+shortcut, negative_slope=0.2)


class Building_block(nn.Module):
    def __init__(self, d_out):  #  d_in = d_out//2
        super().__init__()
        self.mlp1 = pt_utils.Conv2d(10, d_out//2, kernel_size=(1,1), bn=True)
        self.att_pooling_1 = Att_pooling(d_out, d_out//2)

        self.mlp2 = pt_utils.Conv2d(d_out//2, d_out//2, kernel_size=(1, 1), bn=True)
        self.att_pooling_2 = Att_pooling(d_out, d_out)

    def forward(self, xyz, feature, neigh_idx):  # feature: Batch*channel*npoints*1
        f_xyz = self.relative_pos_encoding(xyz, neigh_idx)  # batch*npoint*nsamples*10
        f_xyz = f_xyz.permute((0, 3, 1, 2)).contiguous()  # batch*10*npoint*nsamples
        f_xyz = self.mlp1(f_xyz)
        f_neighbours = self.gather_neighbour(
            feature.squeeze(-1).permute((0, 2, 1)).contiguous(),neigh_idx
        )  # batch*npoint*nsamples*channel
        f_neighbours = f_neighbours.permute((0, 3, 1, 2)).contiguous()  # batch*channel*npoint*nsamples
        f_concat = torch.cat([f_neighbours, f_xyz], dim=1)
        f_pc_agg = self.att_pooling_1(f_concat)  # Batch*channel*npoints*1

        f_xyz = self.mlp2(f_xyz)
        f_neighbours = self.gather_neighbour(
            f_pc_agg.squeeze(-1).permute((0, 2, 1)).contiguous(), neigh_idx
        ).contiguous()  # batch*npoint*nsamples*channel
        f_neighbours = f_neighbours.permute((0, 3, 1, 2)).contiguous()  # batch*channel*npoint*nsamples
        f_concat = torch.cat([f_neighbours, f_xyz], dim=1)
        f_pc_agg = self.att_pooling_2(f_concat)
        return f_pc_agg

    def relative_pos_encoding(self, xyz, neigh_idx):
        neighbor_xyz = self.gather_neighbour(xyz, neigh_idx)  # batch*npoint*nsamples*3

        xyz_tile = xyz.unsqueeze(2).repeat(1, 1, neigh_idx.shape[-1], 1)  # batch*npoint*nsamples*3
        relative_xyz = xyz_tile - neighbor_xyz  # batch*npoint*nsamples*3
        relative_dis = torch.sqrt(torch.sum(torch.pow(relative_xyz, 2), dim=-1, keepdim=True))  # batch*npoint*nsamples*1
        relative_feature = torch.cat([relative_dis, relative_xyz, xyz_tile, neighbor_xyz], dim=-1)  # batch*npoint*nsamples*10
        return relative_feature

    @staticmethod
    def gather_neighbour(pc, neighbor_idx):  # pc: batch*npoint*channel
        # gather the coordinates or features of neighboring points
        batch_size = pc.shape[0]
        num_points = pc.shape[1]
        d = pc.shape[2]
        index_input = neighbor_idx.reshape(batch_size, -1)
        features = torch.gather(pc, 1, index_input.unsqueeze(-1).repeat(1, 1, pc.shape[2])).contiguous()
        features = features.reshape(batch_size, num_points, neighbor_idx.shape[-1], d)  # batch*npoint*nsamples*channel
        return features


class Att_pooling(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.fc = nn.Conv2d(d_in, d_in, (1, 1), bias=False)
        self.mlp = pt_utils.Conv2d(d_in, d_out, kernel_size=(1,1), bn=True)

    def forward(self, feature_set):

        att_activation = self.fc(feature_set)
        att_scores = F.softmax(att_activation, dim=3)
        f_agg = feature_set * att_scores
        f_agg = torch.sum(f_agg, dim=3, keepdim=True)
        f_agg = self.mlp(f_agg)
        return f_agg


def compute_loss(end_points, cfg):

    logits = end_points['logits']
    labels = end_points['labels']

    logits = logits.transpose(1, 2).reshape(-1, cfg.num_classes)
    labels = labels.reshape(-1)

    # Boolean mask of points that should be ignored
    ignored_bool = labels == 0
    for ign_label in cfg.ignored_label_inds:
        ignored_bool = ignored_bool | (labels == ign_label)

    # Collect logits and labels that are not ignored
    valid_idx = ignored_bool == 0
    valid_logits = logits[valid_idx, :]
    valid_labels_init = labels[valid_idx]

    # Reduce label values in the range of logit shape
    reducing_list = torch.range(0, cfg.num_classes).long().cuda()
    inserted_value = torch.zeros((1,)).long().cuda()
    for ign_label in cfg.ignored_label_inds:
        reducing_list = torch.cat([reducing_list[:ign_label], inserted_value, reducing_list[ign_label:]], 0)
    valid_labels = torch.gather(reducing_list, 0, valid_labels_init)
    loss = get_loss(valid_logits, valid_labels, cfg.class_weights)
    end_points['valid_logits'], end_points['valid_labels'] = valid_logits, valid_labels
    end_points['loss'] = loss
    return loss, end_points


def get_loss(logits, labels, pre_cal_weights):
    # calculate the weighted cross entropy according to the inverse frequency
    class_weights = torch.from_numpy(pre_cal_weights).float().cuda()
    # one_hot_labels = F.one_hot(labels, self.config.num_classes)

    criterion = nn.CrossEntropyLoss(weight=class_weights, reduction='none')
    output_loss = criterion(logits, labels)
    output_loss = output_loss.mean()
    return output_loss
