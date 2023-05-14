#!/usr/bin/env python3

import os.path as osp
import os
import torch
from utils.ip_basic.ip_basic import depth_map_utils_ycb as depth_map_utils
import cv2
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from utils.ply import normalize_color
from utils.icp import nearest_neighbor
from utils.compute_visibility import VisiblePoints
import normalSpeed
import json
import ref.lmo as lm_ref
from models.RandLA.helper_tool import DataProcessing as DP
from utils.dataset_utils import crop_resize_by_warp_affine

def ensure_fd(fd):
    if not os.path.exists(fd):
        os.system('mkdir -p {}'.format(fd))

class LMDataset(torch.utils.data.Dataset):
    def __init__(self, data_cfg, dataset_name,demo_id=None):

        self.data_cfg = data_cfg
        if demo_id is None:
            self.obj_ids = np.array(data_cfg["OBJ_IDS"])
        else:
            self.obj_ids = np.array([demo_id])

        self.selected_id = data_cfg["SELECTED_OBJ_ID"]
        self.pbr_annos = []
        self.n_points = data_cfg["MODEL_PT_NUM"]
        self.dataset_name = dataset_name
        self.n_sample_pts = data_cfg["NUM_SAMPLE_POINTS"]
        self.nn_dist_th = data_cfg["NN_DIST_TH"] * data_cfg["MODEL_D"][self.selected_id] / 1000.0
        
        self.im_h,self.im_w = data_cfg["IMG_SIZE"]
        self.dzi_scale_ratio = data_cfg["DZI_SCALE_RATIO"]
        self.dzi_shift_ratio = data_cfg["DZI_SHIFT_RATIO"]
        self.dzi_pad_ratio = data_cfg["DZI_PAD_RATIO"]
        self.in_size = data_cfg["INPUT_SIZE"]

        if dataset_name == 'train':
            self.subsets = data_cfg["TRAIN"]
            self.add_noise = True
        else:
            self.subsets = data_cfg["TEST"]
            self.add_noise = False
        
        for subset in self.subsets:
            subset_pth = osp.join(data_cfg["DATA_ROOT_DIR"],subset)
            idx_file = osp.join(data_cfg["DATA_ROOT_DIR"],subset,'train.txt')
            self.pbr_annos += self.load_subset_dicts(idx_file,subset_pth)
        if dataset_name == "test":
            self.test_annos = self.pbr_annos
   
        self.length = len(self.pbr_annos)
        print(f'Totally load {self.length} instences.')
       
        self.trancolor = transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)
        self.obj_mesh = self.load_mesh(osp.join(data_cfg["DATA_ROOT_DIR"],'kps'),self.selected_id)        
        self.rng = np.random

    def fill_missing(
             self,dpt, cam_scale, scale_2_80m, fill_type='multiscale',
            extrapolate=False, show_process=False, blur_type='bilateral'
    ):
        dpt = dpt / cam_scale * scale_2_80m
        projected_depth = dpt.copy()
        if fill_type == 'fast':
            final_dpt = depth_map_utils.fill_in_fast(
                projected_depth, extrapolate=extrapolate, blur_type=blur_type,
            )
        elif fill_type == 'multiscale':
            final_dpt, process_dict = depth_map_utils.fill_in_multiscale(
                projected_depth, extrapolate=extrapolate, blur_type=blur_type,
                show_process=show_process,
                max_depth=3.0
            )
        else:
            raise ValueError('Invalid fill_type {}'.format(fill_type))
        dpt = final_dpt / scale_2_80m * cam_scale
        return dpt
        

    def load_mesh(self,pth,cls_id):

        model_pth = osp.join(pth,f"obj_{cls_id:06d}_fps.npy")
        data = np.load(model_pth)
        pts = data[:self.n_points,:3].astype(np.float32) / 1000.0
        rgb = data[:self.n_points,3:6].astype(np.uint8)
        nrm = data[:self.n_points,6:9].astype(np.float32)
        pts_rgb_nrm = np.concatenate((pts,rgb,nrm),axis=1)
        return pts_rgb_nrm

    def aug_bbox_DZI(self, bbox_xyxy):
        """Used for DZI, the augmented box is a square (maybe enlarged)
        Args:
            bbox_xyxy (np.ndarray):
        Returns:
             aug bbox_xyxy
        """
        x1, y1, x2, y2 = bbox_xyxy.copy()
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        bh = y2 - y1
        bw = x2 - x1
        scale_ratio = 1 + self.dzi_scale_ratio * (2 * np.random.random_sample() - 1)  # [1-0.25, 1+0.25]
        shift_ratio = self.dzi_shift_ratio * (2 * np.random.random_sample(2) - 1)  # [-0.25, 0.25]
        if self.dataset_name == "test":
            scale_ratio = 1
            shift_ratio = np.array([0,0])

        bbox_center = np.array([cx + bw * shift_ratio[0], cy + bh * shift_ratio[1]])  # (h/2, w/2)
        scale = max(y2 - y1, x2 - x1) * scale_ratio * self.dzi_pad_ratio
        scale = min(scale, max(self.im_h, self.im_w))
        return bbox_center, scale

    
    def load_subset_dicts(self,idx_file,img_root):

        scene_gt_dicts = {}
        scene_gt_info_dicts = {}
        scene_cam_dicts = {}
        scene_im_ids = []  # store tuples of (scene_id, im_id)
        if self.dataset_name == 'test':
            est_bbox_file = osp.join(img_root, "real_det.json")
            assert osp.exists(est_bbox_file), est_bbox_file
            with open(est_bbox_file,'r') as f:
                est_dict = json.load(f)

        with open(idx_file, "r") as f:
            for line in f:
                line_split = line.strip("\r\n").split("/")
                scene_id = int(line_split[0])
                im_id = int(line_split[1])
                scene_im_ids.append((scene_id, im_id))
                if scene_id not in scene_gt_dicts:
                    scene_gt_file = osp.join(img_root, f"{scene_id:06d}/scene_gt.json")
                    assert osp.exists(scene_gt_file), scene_gt_file
                    with open(scene_gt_file,'r') as f:
                        scene_gt_dicts[scene_id] = json.load(f)

                if scene_id not in scene_gt_info_dicts:
                    scene_gt_info_file = osp.join(img_root, f"{scene_id:06d}/scene_gt_info.json")
                    assert osp.exists(scene_gt_info_file), scene_gt_info_file
                    with open(scene_gt_info_file,'r') as f:
                        scene_gt_info_dicts[scene_id] = json.load(f)

                if scene_id not in scene_cam_dicts:
                    scene_cam_file = osp.join(img_root, f"{scene_id:06d}/scene_camera.json")
                    assert osp.exists(scene_cam_file), scene_cam_file
                    with open(scene_cam_file,'r') as f:
                        scene_cam_dicts[scene_id] = json.load(f)
        ######################################################
        scene_im_ids = sorted(scene_im_ids)  # sort to make it reproducible
        dataset_dicts = []
        num_instances_without_valid_box = 0
        fail_to_det = np.zeros((len(self.obj_ids)))
        found = np.zeros((len(self.obj_ids)))
        if 'pbr' in img_root:
            img_dtype = 'jpg'
        else:
            img_dtype = 'png'

        for scene_id, im_id in scene_im_ids:
            rgb_path = osp.join(img_root, f"{scene_id:06d}/rgb/{im_id:06d}.{img_dtype}")
            assert osp.exists(rgb_path), rgb_path
            str_im_id = str(im_id)

            # for ycbv/tless, load cam K from image infos
            cam_anno = np.array(scene_cam_dicts[scene_id][str_im_id]["cam_K"], dtype=np.float32).reshape(3, 3)
            depth_factor = 1000.0 / scene_cam_dicts[scene_id][str_im_id]["depth_scale"]
            depth_file = osp.join(img_root, f"{scene_id:06d}/depth/{im_id:06d}.png")
            assert osp.exists(depth_file), depth_file
        
            if "pbr" in rgb_path:
                img_type = 'pbr'
            else:
                img_type = "test"

            anno_dict_list = scene_gt_dicts[scene_id][str(im_id)]
            info_dict_list = scene_gt_info_dicts[scene_id][str(im_id)]
            for anno_i, anno in enumerate(anno_dict_list):
                info = info_dict_list[anno_i]
                obj_id = anno["obj_id"]
                valid_px = info["px_count_visib"]

                if self.dataset_name == 'train' and self.selected_id != obj_id:
                    continue

                if obj_id not in self.obj_ids or valid_px < 30:
                    continue
                ################ pose ###########################
                R = np.array(anno["cam_R_m2c"], dtype="float32").reshape(3, 3)
                trans = np.array(anno["cam_t_m2c"], dtype="float32") / 1000.0  # mm->m
                pose = np.hstack([R, trans.reshape(3, 1)])

                ############# bbox ############################
                if self.dataset_name == "test":
                    bbox_est = None
                    file_name = est_dict.get(f"{scene_id}/{im_id}",None)
                    if file_name == None:
                        print(f'picture lost.id:{scene_id}/{im_id}')
                    if file_name != None:
                        if str(obj_id) in est_dict[f"{scene_id}/{im_id}"]:
                            found[self.obj_ids==obj_id] += 1
                            max_score = 0
                            for item in est_dict[f"{scene_id}/{im_id}"][str(obj_id)]:
                              
                                if item['score']>max_score:
                                    max_score = item['score']
                                    x1, y1, x2, y2 = item['bbox']
                                    x1 = int(x1)
                                    x2 = int(x2)
                                    y1 = int(y1)
                                    y2 = int(y2)
                                    bbox_est = [int(x1), int(y1), int(x2), int(y2)]

                bbox = info["bbox_obj"]
                x1, y1, w, h = bbox
                x2 = x1 + w
                y2 = y1 + h
                x1 = max(min(x1, self.im_w), 0)
                y1 = max(min(y1, self.im_h), 0)
                x2 = max(min(x2, self.im_w), 0)
                y2 = max(min(y2, self.im_h), 0)
                bbox = [x1, y1, x2, y2]
                
                bw = bbox[2] - bbox[0]
                bh = bbox[3] - bbox[1]
                if bh <= 1 or bw <= 1:
                    num_instances_without_valid_box += 1
                    continue

                ############## mask ###########################
                mask_full_file = osp.join(img_root, f"{scene_id:06d}/mask_visib/{im_id:06d}_{anno_i:06d}.png")
                assert osp.exists(mask_full_file), mask_full_file
                ################ depth ########################

                record = {
                "rgb_file": rgb_path,
                "cam": cam_anno,  # self.cam,
                "depth_factor": depth_factor,
                "depth_file": depth_file,
                "img_type": img_type,
                "bbox": bbox,  # TODO: load both bbox_obj and bbox_visib
                "pose": pose,
                "mask_file": mask_full_file,  # TODO: load as mask_full, rle
                "obj_id": obj_id,
                }
                if self.dataset_name == 'test':
                    record["file_name"] = f"{scene_id:06d}/{im_id:06d}"
                    if bbox_est is None:
                        fail_to_det[self.obj_ids==obj_id] += 1
                        bbox_est = [0,0,0,0]
                    record["bbox_est"] = bbox_est

                dataset_dicts.append(record)
        print(
                "Filtered out {} instances without valid box. {} objects are not detected. {} found"
                "There might be issues in your dataset generation process.".format(num_instances_without_valid_box,fail_to_det,found)
            )
        return dataset_dicts

    def rand_range(self, rng, lo, hi):
        return rng.rand()*(hi-lo)+lo

    def gaussian_noise(self, rng, img, sigma):
        """add gaussian noise of given sigma to image"""
        img = img + rng.randn(*img.shape) * sigma
        img = np.clip(img, 0, 255).astype('uint8')
        return img

    def linear_motion_blur(self, img, angle, length):
        """:param angle: in degree"""
        rad = np.deg2rad(angle)
        dx = np.cos(rad)
        dy = np.sin(rad)
        a = int(max(list(map(abs, (dx, dy)))) * length * 2)
        if a <= 0:
            return img
        kern = np.zeros((a, a))
        cx, cy = a // 2, a // 2
        dx, dy = list(map(int, (dx * length + cx, dy * length + cy)))
        cv2.line(kern, (cx, cy), (dx, dy), 1.0)
        s = kern.sum()
        if s == 0:
            kern[cx, cy] = 1.0
        else:
            kern /= s
        return cv2.filter2D(img, -1, kern)

    def rgb_add_noise(self, img):
        rng = self.rng
        # apply HSV augmentor
        if rng.rand() > 0:
            hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.uint16)
            hsv_img[:, :, 1] = hsv_img[:, :, 1] * self.rand_range(rng, 1.25, 1.45)
            hsv_img[:, :, 2] = hsv_img[:, :, 2] * self.rand_range(rng, 1.15, 1.35)
            hsv_img[:, :, 1] = np.clip(hsv_img[:, :, 1], 0, 255)
            hsv_img[:, :, 2] = np.clip(hsv_img[:, :, 2], 0, 255)
            img = cv2.cvtColor(hsv_img.astype(np.uint8), cv2.COLOR_HSV2BGR)

        if rng.rand() > .8:  # sharpen
            kernel = -np.ones((3, 3))
            kernel[1, 1] = rng.rand() * 3 + 9
            kernel /= kernel.sum()
            img = cv2.filter2D(img, -1, kernel)

        if rng.rand() > 0.8:  # motion blur
            r_angle = int(rng.rand() * 360)
            r_len = int(rng.rand() * 15) + 1
            img = self.linear_motion_blur(img, r_angle, r_len)

        if rng.rand() > 0.8:
            if rng.rand() > 0.2:
                img = cv2.GaussianBlur(img, (3, 3), rng.rand())
            else:
                img = cv2.GaussianBlur(img, (5, 5), rng.rand())

        if rng.rand() > 0.2:
            img = self.gaussian_noise(rng, img, rng.randint(15))
        else:
            img = self.gaussian_noise(rng, img, rng.randint(25))

        if rng.rand() > 0.8:
            img = img + np.random.normal(loc=0.0, scale=7.0, size=img.shape)

        return np.clip(img, 0, 255).astype(np.uint8)

    def add_real_back(self, rgb, labels, dpt, dpt_msk):
        rnd_h = self.rng.randint(0, self.im_h - self.in_size - 1)
        rnd_w = self.rng.randint(0,self.im_w - self.in_size - 1)

    
        real_item_idx = self.rng.randint(0, len(self.real_annos))
        depth_factor = self.real_annos[real_item_idx]['depth_factor']
        with Image.open(self.real_annos[real_item_idx]["depth_file"]) as di:
            real_dpt = np.array(di) / 1000
            dpt_clip = real_dpt[rnd_h:(rnd_h+self.in_size),rnd_w:(rnd_w+self.in_size)]
        with Image.open(self.real_annos[real_item_idx]["mask_file"]) as li:
            bk_label = np.array(li)
            bk_clip = bk_label[rnd_h:(rnd_h+self.in_size),rnd_w:(rnd_w+self.in_size)]

        bk_clip = (bk_clip < 255).astype(rgb.dtype)
        #bk_label = (bk_label < 255).astype(rgb.dtype)
        if len(bk_label.shape) > 2:
            bk_clip = bk_clip[:, :, 0]
        with Image.open(self.real_annos[real_item_idx]["rgb_file"]) as ri:
            bk_rgb = np.array(ri)[:, :, :3]
            rgb_clip = bk_rgb[rnd_h:(rnd_h+self.in_size),rnd_w:(rnd_w+self.in_size)]
            back = rgb_clip * bk_clip[:, :, None]
        dpt_back = dpt_clip.astype(np.float32) * bk_clip.astype(np.float32)

        #if self.rng.rand() < 0.6:
        msk_back = (labels <= 0).astype(rgb.dtype)
        msk_back = msk_back[:, :, None]
        rgb = rgb * (msk_back == 0).astype(rgb.dtype) + back * msk_back

        dpt = dpt * (dpt_msk > 0).astype(dpt.dtype) + \
            dpt_back * (dpt_msk <= 0).astype(dpt.dtype)
        return rgb, dpt

    def dpt_2_pcld2(self, dpt, bbox, K):


        if len(dpt.shape) > 2:
            dpt = dpt[:, :, 0]
        x1,y1,x2,y2 = bbox
        xmap,ymap = np.mgrid[y1:y2,x1:x2]
      
        dpt = dpt.astype(np.float32) 
        msk = (dpt > 1e-8).astype(np.float32)
        row = (ymap - K[0][2]) * dpt / K[0][0]
        col = (xmap - K[1][2]) * dpt / K[1][1]
        dpt_3d = np.concatenate(
            (row[..., None], col[..., None], dpt[..., None]), axis=2
        )
        dpt_3d = dpt_3d * msk[:, :, None]
        return dpt_3d

    def transform_cld(self, cld, pose):

        t = pose[:3,3:4].T
        rand_t = (np.random.random_sample(3) - 0.5) * 0.6
        rad = (np.random.uniform(1) - 0.5) * 2
        rotation_axis = t / np.linalg.norm(t)
        R_rand = cv2.Rodrigues(rotation_axis * rad * np.pi)
        rotated_cld = np.dot(cld,R_rand[0])
        transformed_cld = rotated_cld + rand_t
        return cld
        

    def dpt_2_pcld(self, dpt, cam_scale, K):
        if len(dpt.shape) > 2:
            dpt = dpt[:, :, 0]
        xmap = np.array([[j for i in range(640)] for j in range(480)])
        ymap = np.array([[i for i in range(640)] for j in range(480)])
        dpt = dpt.astype(np.float32) / cam_scale
        msk = (dpt > 1e-8).astype(np.float32)
        row = (ymap - K[0][2]) * dpt / K[0][0]
        col = (xmap - K[1][2]) * dpt / K[1][1]
        dpt_3d = np.concatenate(
            (row[..., None], col[..., None], dpt[..., None]), axis=2
        )
        dpt_3d = dpt_3d * msk[:, :, None]
        return dpt_3d

    def get_item(self, item_name):

        anno = self.pbr_annos[item_name]

        ##data type
        img_typ = anno["img_type"]
        depth_factor = anno['depth_factor']

        #rgb
        with Image.open(anno["rgb_file"]) as ri:
            rgb = np.array(ri)[:, :, :3]
            origin_rgb = rgb.copy()
        ##depth
        with Image.open(anno["depth_file"]) as di:  
            dpt = np.array(di).astype(np.float32)
            if img_typ == "pbr" or img_typ == "test":
                dpt_m = dpt / depth_factor
            else:
                dpt_m= dpt / 1000.0
        msk_dp = (dpt_m > 1e-6).astype(np.uint8)
        ##mask
        with Image.open(anno["mask_file"]) as li: 
            labels = np.array(li)

        ###Intrinsic matrix
        K = anno['cam']
        ##data type
        success_det = 1

        if self.dataset_name == "train":
            bbox = anno["bbox"]
        else:
            if anno["bbox_est"][2] != 0:
                bbox = anno["bbox_est"]
            else:
                success_det = 0
                bbox = anno["bbox"]

    
        bbox_center,scale = self.aug_bbox_DZI(bbox)
        dpt_xyz = self.dpt_2_pcld(dpt_m, 1,K)
        ## calculate normals#
        
        def norm2bgr(norm):
            norm = ((norm + 1.0) * 127).astype("uint8")
            return norm
        ###maybe exsist bug here 
        dpt_mm = (dpt_m * 1000).astype(np.uint16)
        nrm_map_whole = normalSpeed.depth_normal(
            dpt_mm, K[0][0], K[1][1], 5, 2000, 20, False
        )
        # nrm_map_whole = norm2bgr(nrm_map_whole)
       
        dpt_xyz[np.isnan(dpt_xyz)] = 0.0
        dpt_xyz[np.isinf(dpt_xyz)] = 0.0
        rgb_clip = crop_resize_by_warp_affine(rgb, bbox_center, scale,(self.in_size),interpolation = cv2.INTER_LINEAR)
        nrm_map_clip = crop_resize_by_warp_affine(nrm_map_whole, bbox_center, scale,(self.in_size),interpolation = cv2.INTER_LINEAR)
        mask_clip = crop_resize_by_warp_affine(labels, bbox_center, scale,(self.in_size),interpolation = cv2.INTER_NEAREST)
        dpt_m_clip = crop_resize_by_warp_affine(dpt_m, bbox_center, scale,(self.in_size),interpolation = cv2.INTER_NEAREST)
        msk_dp_clip = crop_resize_by_warp_affine(msk_dp, bbox_center, scale,(self.in_size),interpolation = cv2.INTER_NEAREST)
        dpt_xyz_clip = crop_resize_by_warp_affine(dpt_xyz, bbox_center, scale,(self.in_size),interpolation = cv2.INTER_NEAREST)

        rgb_clip = normalize_color(rgb_clip)
        msk_dp = dpt_m_clip > 1e-6
        choose = msk_dp.flatten().nonzero()[0].astype(np.uint32)

        if len(choose) < 200 and self.dataset_name == 'train':
            return None
        elif self.dataset_name == "test" and len(choose) == 0:

            choose = [0]

        choose_2 = np.array([i for i in range(len(choose))])  
        if len(choose_2) > self.n_sample_pts:
            c_mask = np.zeros(len(choose_2), dtype=int)
            c_mask[:self.n_sample_pts] = 1
            np.random.shuffle(c_mask)
            choose_2 = choose_2[c_mask.nonzero()]
        else:
            choose_2 = np.pad(choose_2, (0, self.n_sample_pts-len(choose_2)), 'wrap')
        choose = np.array(choose)[choose_2]
        sf_idx = np.arange(choose.shape[0])
        np.random.shuffle(sf_idx)
        choose = choose[sf_idx]
        
        cld = dpt_xyz_clip.reshape(-1, 3)[choose, :]
        rgb_pt = rgb_clip.reshape(-1, 3)[choose, :].astype(np.float32)
        nrm_pt = nrm_map_clip[:, :, :3].reshape(-1, 3)[choose, :]
        labels_pt = mask_clip.flatten()[choose]
        labels_pt[labels_pt==255] = 1
        choose = np.array([choose])
        rgb_clip = np.transpose(rgb_clip, (2, 0, 1))
        
        labels_refined, mesh_match_idx, visible_flag,valid_data = self.get_pose_gt_info(
                cld, labels_pt,anno
            )
        if self.dataset_name == "train" and not valid_data:
            return None

        cld_rgb_nrm = np.concatenate((cld, rgb_pt, nrm_pt), axis=1).transpose(1, 0)


        xyz_lst = [dpt_xyz_clip.transpose(2, 0, 1)]  # c, h, w
        msk_lst = [dpt_xyz_clip[2, :, :] > 1e-8]

        for i in range(3):
            scale = pow(2, i+1)
            nh, nw = self.in_size // pow(2, i+1), self.in_size // pow(2, i+1)
            ys, xs = np.mgrid[:nh, :nw]
            xyz_lst.append(xyz_lst[0][:, ys*scale, xs*scale])
            msk_lst.append(xyz_lst[-1][2, :, :] > 1e-8)
        sr2dptxyz = {
            pow(2, ii): item.reshape(3, -1).transpose(1, 0)
            for ii, item in enumerate(xyz_lst)
        }

        rgb_ds_sr = [4, 8, 8, 8]
        n_ds_layers = 4
        pcld_sub_s_r = [4, 4, 4, 4]
        inputs = {}
        # DownSample stage
        for i in range(n_ds_layers):
            nei_idx = DP.knn_search(
                cld[None, ...], cld[None, ...], 16
            ).astype(np.int32).squeeze(0)
            sub_pts = cld[:cld.shape[0] // pcld_sub_s_r[i], :]
            pool_i = nei_idx[:cld.shape[0] // pcld_sub_s_r[i], :]
            up_i = DP.knn_search(
                sub_pts[None, ...], cld[None, ...], 1
            ).astype(np.int32).squeeze(0)
            inputs['cld_xyz%d' % i] = cld.astype(np.float32).copy()
            inputs['cld_nei_idx%d' % i] = nei_idx.astype(np.int32).copy()
            inputs['cld_sub_idx%d' % i] = pool_i.astype(np.int32).copy()
            inputs['cld_interp_idx%d' % i] = up_i.astype(np.int32).copy()
            nei_r2p = DP.knn_search(
                sr2dptxyz[rgb_ds_sr[i]][None, ...], sub_pts[None, ...], 16
            ).astype(np.int32).squeeze(0)
            inputs['r2p_ds_nei_idx%d' % i] = nei_r2p.copy()
            nei_p2r = DP.knn_search(
                sub_pts[None, ...], sr2dptxyz[rgb_ds_sr[i]][None, ...], 1
            ).astype(np.int32).squeeze(0)
            inputs['p2r_ds_nei_idx%d' % i] = nei_p2r.copy()
            cld = sub_pts

        n_up_layers = 3
        rgb_up_sr = [4, 2, 2]
        for i in range(n_up_layers):
            r2p_nei = DP.knn_search(
                sr2dptxyz[rgb_up_sr[i]][None, ...],
                inputs['cld_xyz%d'%(n_ds_layers-i-1)][None, ...], 16
            ).astype(np.int32).squeeze(0)
            inputs['r2p_up_nei_idx%d' % i] = r2p_nei.copy()
            p2r_nei = DP.knn_search(
                inputs['cld_xyz%d'%(n_ds_layers-i-1)][None, ...],
                sr2dptxyz[rgb_up_sr[i]][None, ...], 1
            ).astype(np.int32).squeeze(0)
            inputs['p2r_up_nei_idx%d' % i] = p2r_nei.copy()

        #print(item_name)
        item_dict = dict(
            rgb = rgb_clip.astype(np.float32),
            cld_rgb_nrm=cld_rgb_nrm.astype(np.float32),  # [9, npts]
            choose=choose.astype(np.int32),  # [1, npts]
            origin_labels = labels_pt.astype(np.int32),
            labels= labels_refined.astype(np.int32),  # [npts]
            RT=anno["pose"].astype(np.float32),
            match_idx = mesh_match_idx.astype(np.int32),
            visible_flag = visible_flag.astype(np.uint8),
            K = K.astype(np.float32),
            bbox = np.array(anno["bbox"],dtype=np.int32),
            rgb_origin = origin_rgb.astype(np.uint8),
        )
        item_dict.update(inputs)
        # for key in item_dict:
        #     print(item_dict[key].shape)

        if self.dataset_name == 'test':
           
            inputs = dict(
                #rgb = origin_rgb.astype(np.uint8),
                cls_id = np.array(anno["obj_id"],dtype=np.int32),
                bbox_est = np.array(anno["bbox_est"]).astype(np.int32),
                file_name = anno['file_name'],
                det = success_det
            )
            item_dict.update(inputs)  
        return item_dict


    def get_pose_gt_info(self, cld, pt_labels,anno):

        ## for each point in 3D, find cooresponding point in 3d model model, 
        # save the corresponding 3D model mesh idx to "mesh_match_idx"
        ##
        
        valid_data = True
        mesh_match_idx = np.ones((len(cld)),dtype=np.int32) * self.n_points
   
        visible_flag = np.zeros((self.n_points),dtype=np.uint8)
        # valid_data = True
        # unfit_nums = 0
        filtered_pt_labels = pt_labels.copy()
        mask_idx = np.where(pt_labels > 0)[0]
       
        r = anno["pose"][:, 0:3]
        t = anno["pose"][:, 3:4]
        T = np.eye(4,dtype=np.float32)
        T[:3,:3] = r
        T[:3,3:4] = t
        T = np.linalg.inv(T)
        inv_t = T[:3,3:4]
        
        RT = np.concatenate((r, t), axis=1)
        obj_cld = cld[pt_labels>0]
      
        if len(obj_cld[:,0]) == 0:
            valid_data = False
            return filtered_pt_labels, mesh_match_idx, visible_flag, valid_data

        model_pts = self.obj_mesh[:,:3]
        visible_idx = VisiblePoints(model_pts,inv_t.T)
        model_vis_pts = model_pts[visible_idx]
        model_vis_pts_proj = np.dot(model_vis_pts,RT[:, :3].T) + RT[:, 3:].T
        visible_flag[visible_idx] = 1

        dist,vis_match_idx = nearest_neighbor(obj_cld,model_vis_pts_proj,1)
        matching_idx = visible_idx[vis_match_idx]
        
        unfitted_cld_idx = np.where(dist > 0.01)[0]
       

        if len(unfitted_cld_idx) == len(obj_cld):
            valid_data = False
            return filtered_pt_labels, mesh_match_idx, visible_flag, valid_data
            
        if unfitted_cld_idx.size!=0:

            matching_idx[unfitted_cld_idx] = self.n_points
            filtered_pt_labels[mask_idx[unfitted_cld_idx]]=0

        mesh_match_idx[pt_labels>0] = np.array(matching_idx.copy(),dtype=np.int32)

        return filtered_pt_labels, mesh_match_idx, visible_flag,valid_data

    def __len__(self):
        return self.length
    def gen_idx(self):
        idx = self.rng.randint(0, self.length)
        return idx
    def __getitem__(self, idx):
        if self.dataset_name == 'train':
            data = self.get_item(idx)
            while data is None:
                id = self.gen_idx()
                data = self.get_item(id)
            return data
        else:
            return self.get_item(idx)
