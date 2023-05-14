#!/usr/bin/env python3
import os
import time
from numpy.lib.function_base import append
import torch
import numpy as np
import pickle as pkl
import concurrent.futures
from common import Config
from utils.basic_utils import Basic_Utils, check_match_distance
from utils.meanshift_pytorch import MeanShiftTorch
from cv2 import imshow, waitKey
from sklearn.neighbors import NearestNeighbors
import torch.nn.functional as F

# config = Config(ds_name='ycb')
# bs_utils = Basic_Utils(config)
# cls_lst = config.ycb_cls_lst
# try:
#     config_lm = Config(ds_name="linemod")
#     bs_utils_lm = Basic_Utils(config_lm)
# except Exception as ex:
#     print(ex)

def nearest_neighbor(src, dst):
    '''
    Find the nearest (Euclidean) neighbor in dst for each point in src
    Input:
        src: Nxm array of points
        dst: Nxm array of points
    Output:
        distances: Euclidean distances of the nearest neighbor
        indices: dst indices of the nearest neighbor
    '''

    assert src.shape == dst.shape

    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(dst)
    distances, indices = neigh.kneighbors(src, return_distance=True)
    return distances.ravel(), indices.ravel()

def best_fit_transform(A, B):
    '''
    Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
    Input:
        A: Nxm numpy array of corresponding points, usually points on mdl
        B: Nxm numpy array of corresponding points, usually points on camera axis
    Returns:
    T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
    R: mxm rotation matrix
    t: mx1 translation vector
    '''

    assert A.shape == B.shape
    # get number of dimensions
    m = A.shape[1]
    # translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B
    # rotation matirx
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)
    # special reflection case
    if np.linalg.det(R) < 0:
        Vt[m-1, :] *= -1
        R = np.dot(Vt.T, U.T)
    # translation
    t = centroid_B.T - np.dot(R, centroid_A.T)
    T = np.zeros((3, 4))
    T[:, :3] = R
    T[:, 3] = t
    return T


def best_fit_transform_with_RANSAC(A, B, max_iter = 20, match_err = 0.015, fix_percent = 0.7):

    best_RT = np.zeros((3,4),dtype=np.float32)
    ptsnum, m = A.shape
    if ptsnum < 4:
        return best_RT

    iter = 0
    best_inlier_nums = 0
    
    #extend_A = np.ones((ptsnum,4),dtype=np.float32)
    #extend_A[:,:3] = A
    #extend_B = np.ones((ptsnum,4),dtype=np.float32)
    #extend_B[:,:3] = B
    #extend_RT = np.eye(4,dtype=np.float)
    curr_RT = best_fit_transform(A,B)

    while iter < max_iter:
    # get num of points, get number of dimensions
        curr_R = curr_RT[:,:3]
        curr_T = curr_RT[:,3:4].T

        tran_A = np.dot(A,curr_R.T) + curr_T
        err_dis = np.linalg.norm(tran_A - B,axis=1)
        match_idx = (err_dis <= match_err)
        inliers_num = match_idx.sum()
        if inliers_num > best_inlier_nums:
            best_inlier_nums = inliers_num
            best_RT = curr_RT
        
        if best_inlier_nums > fix_percent * ptsnum:
            best_RT = best_fit_transform(A[match_idx],B[match_idx])
            return best_RT
        np.random.seed()
        selected_idx = np.random.randint(0,ptsnum,4)
        selected_A = A[selected_idx]
        selected_B = B[selected_idx]
        curr_RT = best_fit_transform(selected_A,selected_B)
        #extend_RT[:3,:] = curr_RT
        #trans_A = np.dot(extend_RT,extend_A.T)
        #err_dis = np.linalg.norm(trans_A - extend_B.T,axis=0)
        

        iter+=1
   
    return best_RT

def best_fit_transform_icp(A, B):
    '''
    Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
    Input:
        A: Nxm numpy array of corresponding points, usually points on mdl
        B: Nxm numpy array of corresponding points, usually points on camera axis
    Returns:
    T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
    R: mxm rotation matrix
    t: mx1 translation vector
    '''

    assert A.shape == B.shape
    # get number of dimensions
    m = A.shape[1]
    # translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B
    # rotation matirx
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)
    # special reflection case
    if np.linalg.det(R) < 0:
        Vt[m-1, :] *= -1
        R = np.dot(Vt.T, U.T)
    # translation
    t = centroid_B.T - np.dot(R, centroid_A.T)
    T = np.identity(m+1)
    T[:m, :m] = R
    T[:m, m] = t
    return T

def icp(A, B, init_pose=None, max_iterations=20, tolerance=0.001):
    '''
    The Iterative Closest Point method: finds best-fit transform that maps points A on to points B
    Input:
        A: Nxm numpy array of source mD points
        B: Nxm numpy array of destination mD point
        init_pose: (m+1)x(m+1) homogeneous transformation
        max_iterations: exit algorithm after max_iterations
        tolerance: convergence criteria
    Output:
        T: final homogeneous transformation that maps A on to B
        distances: Euclidean distances (errors) of the nearest neighbor
        i: number of iterations to converge
    '''

    assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    # make points homogeneous, copy them to maintain the originals
    src = np.ones((m+1,A.shape[0]))
    dst = np.ones((m+1,B.shape[0]))
    src[:m,:] = np.copy(A.T)
    dst[:m,:] = np.copy(B.T)

    # apply the initial pose estimation
    if init_pose is not None:
        src = np.dot(init_pose, src)

    prev_error = 0

    for i in range(max_iterations):
        # find the nearest neighbors between the current source and destination points
        distances, indices = nearest_neighbor(src[:m,:].T, dst[:m,:].T)

        # compute the transformation between the current source and nearest destination points
        T = best_fit_transform_icp(src[:m,:].T, dst[:m,indices].T)

        # update the current source
        src = np.dot(T, src)

        # check error
        mean_error = np.mean(distances)
        if np.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

    # calculate final transformation
    T = best_fit_transform_icp(A, src[:m,:].T)

    return T[:m,:]



# ###############################YCB Evaluation###############################
def cal_frame_poses(
     cld, labels, seg_features, mesh_features, rgbd_features,match_idx
):
    """
    Calculates pose parameters by dense keypoints matching 
    then use least-squares fitting to get the pose parameters.
    """
    padding = - torch.ones((128,1),dtype=torch.float32).cuda()
    padding = F.normalize(padding,p=2,dim=0)
    seg_res = torch.argmax(seg_features,dim=0)
    rgbd_features = rgbd_features.transpose(0,1)
    #mesh_features = mesh_features.transpose(1,2)
    # Use center clustering filter to improve the predicted mask.
    
    #pred_cls_ids = np.unique(seg_res[seg_res > 0].contiguous().cpu().numpy()).astype(np.int)

    pred_cls_ids = np.unique(seg_res[seg_res > 0].contiguous().cpu().numpy()).astype(np.int)
    #gt_cls_ids = np.unique(labels[labels > 0].contiguous().cpu().numpy()).astype(np.int)
    # for icls,cls_id in enumerate(gt_cls_ids):
    #     if cls_id == 0:
    #          break
    #     cls_msk = (labels == cls_id)
    #     print(f'id:{cls_id}, mask_num:{cls_msk.sum()}\n')
    # 3D keypoints voting and least squares fitting for pose parameters estimation.
    pred_pose_lst = []
    ap_lst = []
    for icls, cls_id in enumerate(pred_cls_ids):
        if cls_id == 0:
            break
        cls_msk = (seg_res == cls_id)
        if cls_msk.sum() <= 1:
            #continue
            pred_pose_lst.append(np.identity(4)[:3, :])
            ap_lst.append(0.0)
            continue

                #ap
        gt_idx = torch.where(labels == cls_id)[0].contiguous().cpu().numpy()
        
        

        #pose
        mesh_features_i = mesh_features[cls_id-1]
        selected_idx = torch.where(seg_res == cls_id)[0]
        #gt_idx = match_idx[selected_idx].cpu().numpy()

        if len(gt_idx)<1:
            ap_lst.append(0.0)
            pred_pose_lst.append(np.identity(4)[:3, :])
            continue
        pred_idx = selected_idx.contiguous().cpu().numpy()
        IOU = len(np.intersect1d(gt_idx,pred_idx)) / len(np.union1d(gt_idx,pred_idx))
        ap_lst.append(IOU)
        
        selected_cld = cld[selected_idx].contiguous().cpu().numpy()
        selected_rgbd_feature = rgbd_features[selected_idx]
        selected_rgbd_feature = F.normalize(selected_rgbd_feature,p=2,dim=1)
        mesh_features_i = F.normalize(mesh_features_i,p=2,dim=0)
        mesh_i_padded = torch.cat([mesh_features_i, padding],dim=1)
        #sim = torch.sum((selected_rgbd_feature.unsqueeze(1) - mesh_features_i.unsqueeze(0)).pow(2), 2)
        obj_pts_sim = torch.matmul(selected_rgbd_feature,mesh_i_padded)
        #general way to eval rt
        #print(obj_pts_sim[0].cpu().numpy())
        #np.savetxt('sim.txt',obj_pts_sim[0].cpu().numpy(),fmt='%.2f')
        #np.savetxt('top1000.txt',torch.topk(obj_pts_sim[0],1000)[0].cpu().numpy(),fmt='%.2f')
        obj_pts_idx = torch.argmax(obj_pts_sim,dim=1)
        valid_obj_pts_idx = torch.where(obj_pts_idx!=8192)[0]
        if len(valid_obj_pts_idx) < 5:
            pred_pose_lst.append(np.identity(4)[:3, :])
            continue
            
        selected_cld = selected_cld[valid_obj_pts_idx.cpu().numpy()]
        #max_prob = torch.max(obj_pts_sim,dim=1)[0]
        #s_idx = (max_prob > 0.6)
        #obj_pts_idx = obj_pts_idx[s_idx]
        #selected_cld = selected_cld[s_idx.cpu().numpy()]
        # if len(selected_cld) < 5:
        #     pred_pose_lst.append(np.identity(4)[:3, :])
        #     ap_lst.append(0.0)
        #     continue


     
        # min_prob = torch.min(max_prob[0])
        
        # dis = check_match_distance(gt_idx,obj_pts_idx,bs_utils.get_mesh_xyz_cuda(cls_id))
        # if len(dis) != 0:
        #     all = len(dis)
        #     ok = len(torch.where(dis<0.02)[0])
        #     mean_dis  = dis.mean()
        #     max_dis = torch.max(dis)
        #     print(f'all:{all} ok:{ok} mean:{mean_dis} max:{max_dis}\n')
        selected_mesh_pts = bs_utils.get_mesh_xyz_cuda(cls_id)[obj_pts_idx[valid_obj_pts_idx]].contiguous().cpu().numpy()
        #test gt to eval rt
        
        # selected_mesh_nodes = match_idx[selected_idx]
        # selected_mesh_idx = torch.where(selected_mesh_nodes!= -1)[0]
        # selected_mesh_nodes = selected_mesh_nodes[selected_mesh_idx]
        # selected_cld = selected_cld[selected_mesh_idx.contiguous().cpu().numpy()]
        #selected_mesh_pts = bs_utils.get_mesh_xyz_cuda(cls_id)[selected_mesh_nodes].contiguous().cpu().numpy()

        
        #pred_RT = best_fit_transform_with_RANSAC(selected_mesh_pts, selected_cld)
        pred_RT = best_fit_transform(selected_mesh_pts, selected_cld)
        pred_pose_lst.append(pred_RT)

    return (pred_cls_ids, pred_pose_lst,ap_lst)


def find_valid_ids(valid_ids):

    j = 0
    ids = []
    idx = valid_ids[j].item()
    while idx>=0:
        if idx not in ids:
            ids.append(idx)
        j+=1
        idx = valid_ids[j].item()
    ids.sort()
    return ids
def eval_metric(
      cls_ids, pred_pose_lst, pred_cls_ids, RTs,ap_lst
):
    n_cls = config.n_classes
    cls_add_dis = [list() for i in range(n_cls)]
    cls_adds_dis = [list() for i in range(n_cls)]
    cls_ap = [list() for i in range(n_cls)]
    cls_ids = find_valid_ids(cls_ids)
    for icls, cls_id in enumerate(cls_ids):
        if cls_id == 0:
            break

        cls_idx = np.where(pred_cls_ids == cls_id)[0]
        if len(cls_idx) == 0:
            pred_RT = torch.zeros(3, 4).cuda()
            cls_ap[cls_id].append(0.0)
            cls_ap[0].append(0.0)
        else:
            pred_RT = pred_pose_lst[cls_idx[0]]
            pred_RT = torch.from_numpy(pred_RT.astype(np.float32)).cuda()
            cls_ap[cls_id].append(ap_lst[cls_idx[0]])
            cls_ap[0].append(ap_lst[cls_idx[0]])
        gt_RT = RTs[icls]
        mesh_pts = bs_utils.get_pointxyz_cuda(cls_id-1).clone()
        add = bs_utils.cal_add_cuda(pred_RT, gt_RT, mesh_pts)
        adds = bs_utils.cal_adds_cuda(pred_RT, gt_RT, mesh_pts)
        cls_add_dis[cls_id].append(add.item())
        cls_adds_dis[cls_id].append(adds.item())
        
        cls_add_dis[0].append(add.item())
        cls_adds_dis[0].append(adds.item())
        

    return (cls_add_dis, cls_adds_dis,cls_ap)


def eval_one_frame_pose_ycb(item):

    cld, labels, seg_features, mesh_features, rgbd_features,RTs,cls_ids,match_idx = item
    pred_cls_ids, pred_pose_lst,ap_lst = cal_frame_poses(
     cld, labels, seg_features, mesh_features, rgbd_features,match_idx
)

    cls_add_dis, cls_adds_dis,cls_ap = eval_metric(
        cls_ids, pred_pose_lst, pred_cls_ids, RTs,ap_lst
    )
    return (cls_add_dis, cls_adds_dis,cls_ap)

def eval_one_frame_pose(
    item
):
    pcld, mask, ctr_of, pred_kp_of, RTs, cls_ids, use_ctr, n_cls, \
        min_cnt, use_ctr_clus_flter, label, epoch, ibs, gt_kps, gt_ctrs, kp_type = item

    pred_cls_ids, pred_pose_lst, pred_kpc_lst = cal_frame_poses(
        pcld, mask, ctr_of, pred_kp_of, use_ctr, n_cls, use_ctr_clus_flter,
        gt_kps, gt_ctrs, kp_type=kp_type
    )

    cls_add_dis, cls_adds_dis, cls_kp_err = eval_metric(
        cls_ids, pred_pose_lst, pred_cls_ids, RTs, mask, label, gt_kps, gt_ctrs,
        pred_kpc_lst
    )
    return (cls_add_dis, cls_adds_dis, pred_cls_ids, pred_pose_lst, cls_kp_err)

# ###############################End YCB Evaluation###############################


# ###############################LineMOD Evaluation###############################

def cal_frame_poses_lm(
   cld, labels, seg_features, mesh_features, rgbd_features,RTs,match_idx,cls_id,use_ctr_filter=False
):
    """
    Calculates pose parameters by 3D keypoints & center points voting to build
    the 3D-3D corresponding then use least-squares fitting to get the pose parameters.
    """
    seg_res = torch.argmax(seg_features,dim=0)
    rgbd_features = rgbd_features.transpose(0,1)
    padding = - torch.ones((128,1),dtype=torch.float32).cuda()

    pred_pose_lst = []
    ap_lst = []

    cls_msk = (seg_res == 1)
    if cls_msk.sum() <= 1:
        #continue
        pred_pose_lst.append(np.identity(4)[:3, :])
        ap_lst.append(0.0)
        return pred_pose_lst,ap_lst
        
    #pose

    selected_idx = torch.where(seg_res == 1)[0]
    mesh_features = mesh_features[0]
    mesh_node_num = mesh_features.shape[1]
    
    selected_cld = cld[selected_idx].contiguous().cpu().numpy()
    selected_rgbd_feature = rgbd_features[selected_idx]
    mesh_i_padded = torch.cat([mesh_features, padding],dim=1)
    selected_rgbd_feature = F.normalize(selected_rgbd_feature,p=2,dim=1)
    mesh_i_padded = F.normalize(mesh_i_padded,p=2,dim=0)
    obj_pts_sim = torch.matmul(selected_rgbd_feature,mesh_i_padded)
    obj_pts_idx = torch.argmax(obj_pts_sim,dim=1)


    valid_obj_pts_idx = torch.where(obj_pts_idx!=mesh_node_num)[0]
    if len(valid_obj_pts_idx) < 3:
        pred_pose_lst.append(np.identity(4)[:3, :])
        ap_lst.append(0.0)
        return pred_pose_lst,ap_lst
        #continue

    gt_obj_idx = match_idx[selected_idx].long()
    gt_valid_obj_idx = torch.where(gt_obj_idx!=mesh_node_num)[0]

    #selected_cld = selected_cld[gt_valid_obj_idx.cpu().numpy()]
            
    selected_cld = selected_cld[valid_obj_pts_idx.cpu().numpy()]

    #selected_mesh_pts = bs_utils.get_mesh_xyz_cuda_lm(cls_id)[gt_obj_idx[gt_valid_obj_idx]].contiguous().cpu().numpy()
    selected_mesh_pts = bs_utils.get_mesh_xyz_cuda_lm(cls_id)[obj_pts_idx[valid_obj_pts_idx]].contiguous().cpu().numpy()
    #selected_mesh_pts = bs_utils.get_mesh_xyz_cuda(cls_id)[obj_pts_idx].contiguous().cpu().numpy()
    

    #use center voting to filter out the outliers
    if use_ctr_filter:

        ms = MeanShiftTorch(bandwidth=0.03)
        ctr_pts = selected_cld - selected_mesh_pts
        ctr, ctr_labels = ms.fit(ctr_pts)

        final_cld = selected_cld[ctr_labels]
        final_mesh_pts = selected_cld[ctr_labels]
        pred_RT = best_fit_transform(final_cld, final_mesh_pts)
    else:
        pred_RT = best_fit_transform(selected_mesh_pts, selected_cld)

        #pred_RT = best_fit_transform_with_RANSAC(selected_mesh_pts, selected_cld)
    pred_pose_lst.append(pred_RT)
    #ap
    gt_idx = torch.where(labels == 1)[0].contiguous().cpu().numpy()
    if gt_idx.sum()<1:
        ap_lst.append(0.0)
        
    pred_idx = selected_idx.contiguous().cpu().numpy()
    IOU = len(np.intersect1d(gt_idx,pred_idx)) / len(np.union1d(gt_idx,pred_idx))
    ap_lst.append(IOU)
    return pred_pose_lst,ap_lst


def eval_metric_lm(cls_id, pred_pose_lst, RTs, ap_lst,labels):
    n_cls = config.n_classes
    cls_add_dis = [list() for i in range(n_cls)]
    cls_adds_dis = [list() for i in range(n_cls)]
    cls_ap_dis = [list() for i in range(n_cls)]

    pred_RT = pred_pose_lst[0]
    ap = ap_lst[0]
    pred_RT = torch.from_numpy(pred_RT.astype(np.float32)).cuda()
    gt_RT = RTs
    mesh_pts = bs_utils_lm.get_mesh_xyz_cuda_lm(cls_id).clone()
    add = bs_utils_lm.cal_add_cuda(pred_RT, gt_RT, mesh_pts,cls_id)
    adds = bs_utils_lm.cal_adds_cuda(pred_RT, gt_RT, mesh_pts)
    # print("obj_id:", obj_id, add, adds)
    cls_add_dis[cls_id].append(add.item())
    cls_adds_dis[cls_id].append(adds.item())
    cls_ap_dis[cls_id].append(ap)
    cls_add_dis[0].append(add.item())
    cls_adds_dis[0].append(adds.item())
    cls_ap_dis[0].append(ap)

    return (cls_add_dis, cls_adds_dis,cls_ap_dis)


def eval_one_frame_pose_lm(
    item
):
    cld, labels, seg_features, mesh_features, rgbd_features,RTs,match_idx,cls_id = item
    pred_pose_lst, ap_lst = cal_frame_poses_lm(
        cld, labels, seg_features, mesh_features, rgbd_features,RTs,match_idx,cls_id
    )

    cls_add_dis, cls_adds_dis,cls_ap_dis = eval_metric_lm(
        cls_id, pred_pose_lst, RTs, ap_lst,labels
    )
    return (cls_add_dis, cls_adds_dis,cls_ap_dis)

# ###############################End LineMOD Evaluation###############################


# ###############################Shared Evaluation Entry###############################
class TorchEval():

    def __init__(self):
        n_cls = 22
        self.n_cls = 22
        self.cls_add_dis = [list() for i in range(n_cls)]
        self.cls_adds_dis = [list() for i in range(n_cls)]
        self.cls_add_s_dis = [list() for i in range(n_cls)]
        self.cls_ap = [list() for i in range(n_cls)]
        #self.pred_kp_errs = [list() for i in range(n_cls)]
        #self.pred_id2pose_lst = []
        self.sym_cls_ids = []

    def cal_auc(self):
        add_auc_lst = []
        adds_auc_lst = []
        add_s_auc_lst = []
        ap_lst = []
        for cls_id in range(1, self.n_cls):
            if (cls_id) in config.ycb_sym_cls_ids:
                self.cls_add_s_dis[cls_id] = self.cls_adds_dis[cls_id]
            else:
                self.cls_add_s_dis[cls_id] = self.cls_add_dis[cls_id]
            self.cls_add_s_dis[0] += self.cls_add_s_dis[cls_id]
        for i in range(self.n_cls):
            add_auc = bs_utils.cal_auc(self.cls_add_dis[i])
            adds_auc = bs_utils.cal_auc(self.cls_adds_dis[i])
            add_s_auc = bs_utils.cal_auc(self.cls_add_s_dis[i])
            per_cls_ap = np.mean(self.cls_ap[i])
            add_auc_lst.append(add_auc)
            adds_auc_lst.append(adds_auc)
            add_s_auc_lst.append(add_s_auc)
            ap_lst.append(per_cls_ap)
            if i == 0:
                continue
            print(cls_lst[i-1])
            print("***************add:\t", add_auc)
            print("***************adds:\t", adds_auc)
            print("***************add(-s):\t", add_s_auc)
            print("***************mAP:\t", per_cls_ap)
        # kp errs:
        # n_objs = sum([len(l) for l in self.pred_kp_errs])
        # all_errs = 0.0
        # for cls_id in range(1, self.n_cls):
        #     all_errs += sum(self.pred_kp_errs[cls_id])
        # print("mean kps errs:", all_errs / n_objs)

        print("Average of all object:")
        print("***************add:\t", np.mean(add_auc_lst[1:]))
        print("***************adds:\t", np.mean(adds_auc_lst[1:]))
        print("***************add(-s):\t", np.mean(add_s_auc_lst[1:]))
        print("***************mAP:\t", np.mean(ap_lst[1:]))

        print("All object (following PoseCNN):")
        print("***************add:\t", add_auc_lst[0])
        print("***************adds:\t", adds_auc_lst[0])
        print("***************add(-s):\t", add_s_auc_lst[0])

        sv_info = dict(
            add_dis_lst=self.cls_add_dis,
            adds_dis_lst=self.cls_adds_dis,
            add_auc_lst=add_auc_lst,
            adds_auc_lst=adds_auc_lst,
            add_s_auc_lst=add_s_auc_lst,
        )
        sv_pth = os.path.join(
            config.log_eval_dir,
            'pvn3d_eval_cuda_{}_{}_{}.pkl'.format(
                adds_auc_lst[0], add_auc_lst[0], add_s_auc_lst[0]
            )
        )
        pkl.dump(sv_info, open(sv_pth, 'wb'))
        sv_pth = os.path.join(
            config.log_eval_dir,
            'pvn3d_eval_cuda_{}_{}_{}_id2pose.pkl'.format(
                adds_auc_lst[0], add_auc_lst[0], add_s_auc_lst[0]
            )
        )
        #pkl.dump(self.pred_id2pose_lst, open(sv_pth, 'wb'))

    def cal_lm_add(self, obj_id, test_occ=False):
        add_auc_lst = []
        adds_auc_lst = []
        add_s_auc_lst = []
        cls_id = obj_id
        if (obj_id) in config_lm.lm_sym_cls_ids:
            self.cls_add_s_dis[cls_id] = self.cls_adds_dis[cls_id]
        else:
            self.cls_add_s_dis[cls_id] = self.cls_add_dis[cls_id]
        self.cls_add_s_dis[0] += self.cls_add_s_dis[cls_id]
        add_auc = bs_utils_lm.cal_auc(self.cls_add_dis[cls_id])
        adds_auc = bs_utils_lm.cal_auc(self.cls_adds_dis[cls_id])
        add_s_auc = bs_utils_lm.cal_auc(self.cls_add_s_dis[cls_id])
        add_auc_lst.append(add_auc)
        adds_auc_lst.append(adds_auc)
        add_s_auc_lst.append(add_s_auc)
        d = config_lm.lm_r_lst[obj_id]['diameter'] / 1000.0 * 0.1
        print("obj_id: ", obj_id, "0.1 diameter: ", d)
        add = np.mean(np.array(self.cls_add_dis[cls_id]) < d) * 100
        adds = np.mean(np.array(self.cls_adds_dis[cls_id]) < d) * 100

        cls_type = config_lm.lm_id2obj_dict[obj_id]
        print(obj_id, cls_type)
        print("***************add auc:\t", add_auc)
        print("***************adds auc:\t", adds_auc)
        print("***************add(-s) auc:\t", add_s_auc)
        print("***************add < 0.1 diameter:\t", add)
        print("***************adds < 0.1 diameter:\t", adds)

        sv_info = dict(
            add_dis_lst=self.cls_add_dis,
            adds_dis_lst=self.cls_adds_dis,
            add_auc_lst=add_auc_lst,
            adds_auc_lst=adds_auc_lst,
            add_s_auc_lst=add_s_auc_lst,
            add=add,
            adds=adds,
        )
        occ = "occlusion" if test_occ else ""
        sv_pth = os.path.join(
            config_lm.log_eval_dir,
            'pvn3d_eval_cuda_{}_{}_{}_{}.pkl'.format(
                cls_type, occ, add, adds
            )
        )
        pkl.dump(sv_info, open(sv_pth, 'wb'))

    def eval_pose_parallel(
        self, pclds, rgbs, masks, pred_ctr_ofs, gt_ctr_ofs, labels, cnt,
        cls_ids, RTs, pred_kp_ofs, gt_kps, gt_ctrs, min_cnt=20, merge_clus=False,
        use_ctr_clus_flter=True, use_ctr=True, obj_id=0, kp_type='farthest',
        ds='ycb'
    ):
        bs, n_kps, n_pts, c = pred_kp_ofs.size()
        masks = masks.long()
        cls_ids = cls_ids.long()
        use_ctr_lst = [use_ctr for i in range(bs)]
        n_cls_lst = [self.n_cls for i in range(bs)]
        min_cnt_lst = [min_cnt for i in range(bs)]
        epoch_lst = [cnt*bs for i in range(bs)]
        bs_lst = [i for i in range(bs)]
        use_ctr_clus_flter_lst = [use_ctr_clus_flter for i in range(bs)]
        obj_id_lst = [obj_id for i in range(bs)]
        kp_type = [kp_type for i in range(bs)]
        if ds == "ycb":
            data_gen = zip(
                pclds, masks, pred_ctr_ofs, pred_kp_ofs, RTs,
                cls_ids, use_ctr_lst, n_cls_lst, min_cnt_lst, use_ctr_clus_flter_lst,
                labels, epoch_lst, bs_lst, gt_kps, gt_ctrs, kp_type
            )
        else:
            data_gen = zip(
                pclds, masks, pred_ctr_ofs, pred_kp_ofs, RTs,
                cls_ids, use_ctr_lst, n_cls_lst, min_cnt_lst, use_ctr_clus_flter_lst,
                labels, epoch_lst, bs_lst, obj_id_lst
            )
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=bs
        ) as executor:
            if ds == "ycb":
                eval_func = eval_one_frame_pose
            else:
                eval_func = eval_one_frame_pose_lm
            for res in executor.map(eval_func, data_gen):
                if ds == 'ycb':
                    cls_add_dis_lst, cls_adds_dis_lst, pred_cls_ids, pred_poses, pred_kp_errs = res
                    self.pred_id2pose_lst.append(
                        {cid: pose for cid, pose in zip(pred_cls_ids, pred_poses)}
                    )
                    self.pred_kp_errs = self.merge_lst(
                        self.pred_kp_errs, pred_kp_errs
                    )
                else:
                    cls_add_dis_lst, cls_adds_dis_lst = res
                self.cls_add_dis = self.merge_lst(
                    self.cls_add_dis, cls_add_dis_lst
                )
                self.cls_adds_dis = self.merge_lst(
                    self.cls_adds_dis, cls_adds_dis_lst
                )

    def eval_pose_parallel_linemod(self,
        cld, labels, seg_features, mesh_features, rgbd_features ,RTs,match_idx,cls_id
    ):
        
        bs,_,_ = cld.shape
        data_gen = zip(
                cld, labels, seg_features, mesh_features, rgbd_features,RTs,match_idx,cls_id)
        
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=bs
        ) as executor:
            eval_func = eval_one_frame_pose_lm
           
            for res in executor.map(eval_func, data_gen):
            
                cls_add_dis_lst, cls_adds_dis_lst,cls_ap = res
                self.cls_add_dis = self.merge_lst(
                    self.cls_add_dis, cls_add_dis_lst
                )
                self.cls_adds_dis = self.merge_lst(
                    self.cls_adds_dis, cls_adds_dis_lst
                )
                self.cls_ap = self.merge_lst(
                    self.cls_ap,cls_ap
                )





    def eval_pose_parallel_ycb(self,
        cld, labels, seg_features, mesh_features, rgbd_features ,RTs,cls_ids,match_idx
    ):
        
        bs,_,_ = cld.shape
        data_gen = zip(
                cld, labels, seg_features, mesh_features, rgbd_features,RTs,cls_ids,match_idx)
        
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=bs
        ) as executor:
            eval_func = eval_one_frame_pose_ycb
           
            for res in executor.map(eval_func, data_gen):
            
                cls_add_dis_lst, cls_adds_dis_lst,cls_ap = res
                self.cls_add_dis = self.merge_lst(
                    self.cls_add_dis, cls_add_dis_lst
                )
                self.cls_adds_dis = self.merge_lst(
                    self.cls_adds_dis, cls_adds_dis_lst
                )
                self.cls_ap = self.merge_lst(
                    self.cls_ap,cls_ap
                )
               
              

    def merge_lst(self, targ, src):
        for i in range(len(targ)):
            targ[i] += src[i]
        return targ

# vim: ts=4 sw=4 sts=4 expandtab
