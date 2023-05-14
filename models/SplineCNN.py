
import os
import torch
import torch.nn.functional as F
from torch.nn import Linear as Lin
from torch_geometric.nn import SplineConv
import torch_geometric.transforms as T
from utils.ply import read_ply_to_data,load_ply
from models.RandLA.helper_tool import DataProcessing as DP
import numpy as np
import ref
from utils.icp import nearest_neighbor
#from lib.pysixd import  misc

class SplineCNN_Mesh_backup(torch.nn.Module):

    def __init__(self, cfg, idx, mesh_in_channels = 9,
                mesh_coord_dim = 3, num_mesh_layers = 6,
                cat = True, lin = True, dropout = 0.1):

        super(SplineCNN_Mesh, self).__init__()

        self.model3d_pth = cfg["model_pth"]
        self.selected_mesh_num = cfg["n_mesh_node"]
        self.mesh = self.load_mesh(self.model3d_pth,idx)
        data = read_ply_to_data(self.mesh)
        x,edge_idx,edge_attr = self.set_mesh_graph(data)
        self.register_buffer('xyz',torch.from_numpy(self.mesh['pts']).float())
        self.register_buffer('mesh_graph_x',x)
        self.register_buffer('mesh_graph_edge_index',edge_idx)
        self.register_buffer('mesh_graph_edge_attr',edge_attr)
        self.register_buffer('const_one',torch.tensor(1))
        out_channels = cfg["feat_dim"]
        self.mesh_in_channels = mesh_in_channels
        self.out_channels = out_channels
        self.num_mesh_layers = num_mesh_layers
        self.mesh_coord_dim = mesh_coord_dim

        self.lin = Lin
        self.cat = cat
        self.dropout = dropout
        self.mesh_convs = torch.nn.ModuleList()

        for _ in range(self.num_mesh_layers):
            mesh_conv = SplineConv(self.mesh_in_channels, out_channels,
             self.mesh_coord_dim, kernel_size = 5)
            self.mesh_convs.append(mesh_conv)
            self.mesh_in_channels = out_channels

        if self.cat:
            mesh_in_channel = mesh_in_channels + num_mesh_layers * out_channels
        else:
            mesh_in_channel = out_channels

        if self.lin:
            self.out_channels = out_channels
            self.mesh_final = Lin(mesh_in_channel,out_channels)
        else:
            self.out_channels = mesh_in_channels

    def set_mesh_graph(self,mesh):
    
        mesh_transform = T.Compose([
                T.KNNGraph(k=4),
                T.Cartesian()
            ])
        mesh_graph = mesh_transform(mesh)
        return mesh_graph.x, mesh_graph.edge_index,mesh_graph.edge_attr

    def load_mesh(self,pth,cls_id):

        model_pth = os.path.join(pth,f"obj_{cls_id:06d}_fps.npy")
        data = np.load(model_pth)
        mesh = {}
        mesh['pts'] = data[:self.selected_mesh_num,:3].astype(np.float32) / 1000.0
        mesh['colors'] = data[:self.selected_mesh_num,3:6].astype(np.uint8)
        mesh['normals'] = data[:self.selected_mesh_num,6:9].astype(np.float32)
        #pts_rgb_nrm = np.concatenate((pts,rgb,nrm),axis=1)
        return mesh

    def forward(self):
       
        meshes = [self.mesh_graph_x]
        for conv in self.mesh_convs:
            meshes += [F.relu(conv(
                meshes[-1], self.mesh_graph_edge_index, self.mesh_graph_edge_attr),inplace=True)]

        # ## TODO:add bn
        if self.cat:
            meshes = torch.cat(meshes, dim = -1)
        else:
            meshes = meshes[-1]

        meshes = F.dropout(meshes, p=self.dropout, training=self.training)

        if self.lin:
            meshes = self.mesh_final(meshes)
        return meshes.transpose(0,1)


class SplineCNN_Mesh(torch.nn.Module):

    def __init__(self, cfg, idx, mesh_in_channels = 9, out_channels = 128,
                mesh_coord_dim = 3, num_mesh_layers = 3,
                cat = True, lin = True, dropout = 0.1):

        super(SplineCNN_Mesh, self).__init__()
        self.model3d_pth = cfg["model_pth"]
        self.selected_mesh_num = cfg["n_mesh_node"]
        self.name = cfg["model_name"]

        #self.model3d_pth = 'datasets/linemod/lm/kps/'
        #self.n_points = 4096
        #self.mesh = self.load_meshes(idx)
        print(f'spline model id:{idx}')
        self.mesh = self.load_mesh_np(idx)
        
        data = read_ply_to_data(self.mesh)
        x,edge_idx,edge_attr = self.set_mesh_graph(data)
        self.register_buffer('xyz',torch.from_numpy(self.mesh['pts']).float())
        self.register_buffer('mesh_graph_x',x)
        self.register_buffer('mesh_graph_edge_index',edge_idx)
        self.register_buffer('mesh_graph_edge_attr',edge_attr)
        self.register_buffer('const_one',torch.tensor(1))

        self.mesh_in_channels = mesh_in_channels
        self.out_channels = out_channels
        self.num_mesh_layers = num_mesh_layers
        self.mesh_coord_dim = mesh_coord_dim

        self.lin = Lin
        self.cat = cat
        self.dropout = dropout
        self.mesh_convs = torch.nn.ModuleList()

        for _ in range(self.num_mesh_layers):
            mesh_conv = SplineConv(self.mesh_in_channels, out_channels,
             self.mesh_coord_dim, kernel_size = 5)
            self.mesh_convs.append(mesh_conv)
            self.mesh_in_channels = out_channels

        if self.cat:
            mesh_in_channel = mesh_in_channels + num_mesh_layers * out_channels
        else:
            mesh_in_channel = out_channels

        if self.lin:
            self.out_channels = out_channels
            self.mesh_final = Lin(mesh_in_channel,out_channels)
        else:
            self.out_channels = mesh_in_channels
        
        self.sys_corr_idx = None

        loaded_models_info = ref.__dict__[self.name].get_models_info()
        model_info = loaded_models_info[str(idx)]
        if "symmetries_discrete" in model_info or "symmetries_continuous" in model_info:
            self.sym_transforms = misc.get_symmetry_transformations(model_info, max_sym_disc_step=0.01)
    
            self.sys_corr_idx = self.cal_sys_idx()
            self.register_buffer('sys_idx',torch.from_numpy(self.sys_corr_idx).long())
        
    def cal_sys_idx(self):
        model_pts = self.mesh['pts']
        sys_r = self.sym_transforms[1]['R']
        sys_t = self.sym_transforms[1]['t'] / 1000.0
        sys_model = np.dot(model_pts,sys_r.T) + sys_t.T
        dist,sys_idx = nearest_neighbor(model_pts,sys_model,1)
        return sys_idx

    def set_mesh_graph(self,mesh):
    
        mesh_transform = T.Compose([
                T.KNNGraph(k=4),
                T.Cartesian()
            ])
        mesh_graph = mesh_transform(mesh)
        return mesh_graph.x, mesh_graph.edge_index,mesh_graph.edge_attr

    def load_mesh_np(self,idx):
        pth = self.model3d_pth
        model_pth = os.path.join(pth,f"obj_{idx:06d}_fps.npy")
        data = np.load(model_pth)
        mesh = {}
        #pts = data[:self.n_points,:3].astype(np.float32) / 1000.0
        mesh['pts'] = data[:self.selected_mesh_num,:3].astype(np.float32) / 1000.0
        rgb = data[:self.selected_mesh_num,3:6].astype(np.uint8)
        mesh['colors'] = rgb
        nrm = data[:self.selected_mesh_num,6:9].astype(np.float32)
        mesh['normals'] = nrm
        mesh['neigh'] = DP.knn_search(mesh['pts'][np.newaxis,:,:],mesh['pts'][np.newaxis,:,:], 16)
        #pts_rgb_nrm = np.concatenate((pts,rgb,nrm),axis=1)
        return mesh


    def load_meshes(self,idx):

        pth = self.model3d_pth


        mesh = load_ply(os.path.join(pth,'obj_'+'%02d'%idx+'.ply'))
        # n,_ = mesh['pts'].shape
        # pts = np.zeros((8192,3),np.float32)
        # normals = np.zeros((8192,3),np.float32)
        # rgb = np.zeros((8192,3),np.float32)
        # normalized_pts = np.zeros((8192,3),np.float32)
        
        # if n >= 8192:
        #     pts = mesh['pts'][:8192]
        #     normals = mesh['normals'][:8192]
        #     rgb = mesh['colors'][:8192]
        # else:
        #     pts[:n,:] = mesh['pts']
        #     normals[:n,:] = mesh['normals']
        #     rgb[:n,:] = mesh['colors']
        # normalized_pts,size = normalize_pts(pts)
        # resized_xyz_list.append(normalized_pts)
        # size_list.append(size.copy())
        # rgb = normalize_color(rgb)
        # pts_rgb_nrm = np.concatenate((pts,rgb,normals),axis=1)
        # mesh_xyz_list.append(pts)
        # mesh_feature_list.append(pts_rgb_nrm)
     
        # mesh_xyz = np.stack(mesh_xyz_list)
        # mesh_feature = np.stack(mesh_feature_list)
        # resized_xyz = np.stack(resized_xyz_list) / max(size_list) 
        # mesh_feature[:,:,:3] = resized_xyz
        mesh['neigh'] = DP.knn_search(mesh['pts'][np.newaxis,:,:],mesh['pts'][np.newaxis,:,:], 16)
        # meshes['pts'] = mesh_xyz
        # meshes['features'] = mesh_feature
       
        return mesh

    def forward(self):
       
        meshes = [self.mesh_graph_x]
        for conv in self.mesh_convs:
            meshes += [F.relu(conv(
                meshes[-1], self.mesh_graph_edge_index, self.mesh_graph_edge_attr),inplace=True)]

        # ## TODO:add bn
        if self.cat:
            meshes = torch.cat(meshes, dim = -1)
        else:
            meshes = meshes[-1]

        meshes = F.dropout(meshes, p=self.dropout, training=self.training)

        if self.lin:
            meshes = self.mesh_final(meshes)
        return meshes.transpose(0,1)
