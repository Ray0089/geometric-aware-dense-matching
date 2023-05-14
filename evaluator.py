# -*- coding: utf-8 -*-
"""inference on dataset; save results; evaluate with custom evaluation
funcs."""

import logging
import os.path as osp
import time
from collections import OrderedDict
import concurrent.futures
import mmcv
import numpy as np
import torch
from detectron2.evaluation import DatasetEvaluator
from tabulate import tabulate
from tqdm import tqdm
import torch.nn.functional as F
cur_dir = osp.dirname(osp.abspath(__file__))
import ref
from utils.my_comm import all_gather, is_main_process, synchronize
from utils.pose_utils import get_closest_rot
from utils.pvn3d_eval_utils_kpls import best_fit_transform, best_fit_transform_with_RANSAC
from lib.pysixd import inout, misc
from lib.pysixd.pose_error import add, adi, arp_2d, re, te
from config import ycbv_cfg
PROJ_ROOT = osp.normpath(osp.join(cur_dir, "../.."))
from utils.icp import nearest_neighbor

class ModelContainer():
    def __init__(self,cfg) -> None:
        self.data_ref = ref.__dict__[cfg.dataset_name]
        self.obj_names = self.data_ref.objects
        self.obj_ids = [self.data_ref.obj2id[obj_name] for obj_name in self.obj_names]
        self.cfg = cfg
        self.model_paths = [
            osp.join(cfg.MESH["MESH_DIR"], "obj_{:06d}_fps.npy".format(obj_id)) for obj_id in self.obj_ids
        ]
        self.models_3d = {}
        self.sys_corr_idx = {}
        for idx,model_path in enumerate(self.model_paths):
            self.models_3d[self.obj_ids[idx]] = np.load(model_path)[:cfg.DATASETS["MODEL_PT_NUM"],:3]/1000.0
        
        loaded_models_info = self.data_ref.get_models_info()
        for id in self.obj_ids:
            model_info = loaded_models_info[str(id)]
            if "symmetries_discrete" in model_info or "symmetries_continuous" in model_info:
                self.sym_transforms = misc.get_symmetry_transformations(model_info, max_sym_disc_step=0.01)
            
                self.sys_corr_idx[str(id)] = self.cal_sys_idx(id)
    def cal_sys_idx(self,model_id):
        model_pts = self.models_3d[model_id][:,:3]
        sys_r = self.sym_transforms[1]['R']
        sys_t = self.sym_transforms[1]['t'] / 1000.0
        sys_model = np.dot(model_pts,sys_r.T) + sys_t.T
        dist,sys_idx = nearest_neighbor(model_pts,sys_model,1)
        return sys_idx


model3ds = ModelContainer(ycbv_cfg)

def cal_frame_poses(item):

    """
    Calculates pose parameters by dense keypoints matching 
    then use least-squares fitting to get the pose parameters.
    """
    cld, seg_features, mesh_features, rgbd_features, cls_id,det = item
    if str(cls_id.item()) in model3ds.sys_corr_idx:
        rot_idx = model3ds.sys_corr_idx[str(cls_id.item())]
        rot_idx = torch.tensor(rot_idx).long().cuda()
    pred_RT = np.eye(4,dtype=np.float32)
    pred_RT[2,3] = -1000
    if not det:
        return pred_RT[:3,:]

    feat_dim = model3ds.cfg.MODEL['feat_dim']
    num_model_pt = model3ds.cfg.DATASETS['MODEL_PT_NUM']
    padding = -torch.ones((feat_dim,1),dtype=torch.float32).cuda()
    seg_res = torch.argmax(seg_features,dim=0)
    rgbd_features = rgbd_features.transpose(0,1)

    
    cls_msk = (seg_res == 1)
    if cls_msk.sum() <= 1:
        return pred_RT[:3,:]
    cld = cld.transpose(0,1)[:,:3]
    
    selected_cld = cld[cls_msk].contiguous().cpu().numpy()
    selected_rgbd_feature = rgbd_features[cls_msk]
    selected_rgbd_feature = F.normalize(selected_rgbd_feature,p=2,dim=1)
    mesh_features = F.normalize(mesh_features,p=2,dim=0)
    obj_pts_sim = torch.matmul(selected_rgbd_feature, mesh_features)

    max_th, obj_pts_idx = torch.max(obj_pts_sim,dim=1)


    if len(obj_pts_idx) < 5:
        return pred_RT[:3,:]

    selected_mesh_pts = model3ds.models_3d[cls_id.item()][obj_pts_idx.cpu().numpy()]
    pred_RT = best_fit_transform(selected_mesh_pts, selected_cld)

    return pred_RT

def cal_pose_no_seg(item):

    """
    Calculates pose parameters by dense keypoints matching 
    then use least-squares fitting to get the pose parameters.
    """
    cld, mesh_features, rgbd_features, cls_id = item
  
    feat_dim = model3ds.cfg.MODEL['feat_dim']
    num_model_pt = model3ds.cfg.DATASETS['MODEL_PT_NUM']
    padding = - torch.ones((feat_dim,1),dtype=torch.float32).cuda()

    rgbd_features = rgbd_features.transpose(0,1)
    pred_RT = np.eye(4,dtype=np.float32)
    pred_RT[2,3] = -1000
  
    cld = cld.transpose(0,1)[:,:3]
    
    cld = cld.contiguous().cpu().numpy()

    rgbd_features = F.normalize(rgbd_features,p=2,dim=1)
    mesh_padded = torch.cat([mesh_features, padding],dim=1)
    mesh_features = F.normalize(mesh_features,p=2,dim=0)
    obj_pts_sim = torch.matmul(rgbd_features, mesh_padded)

    obj_pts_idx = torch.argmax(obj_pts_sim,dim=1)
    valid_obj_pts_idx = torch.where(obj_pts_idx!=num_model_pt)[0]

    if len(valid_obj_pts_idx) < 5:
        return pred_RT[:3,:]
    cld = cld[valid_obj_pts_idx.cpu().numpy()]
    final_obj_idx = obj_pts_idx[valid_obj_pts_idx].cpu().numpy()
    selected_mesh_pts = model3ds.models_3d[cls_id-1][final_obj_idx]
    pred_RT = best_fit_transform(selected_mesh_pts, cld)
    return pred_RT

class Evaluator(DatasetEvaluator):
    """custom evaluation of 6d pose."""

    def __init__(self, cfg, dataset_name, distributed, output_dir, dataset, train_objs=None):

        self.cfg = cfg
        self._distributed = distributed
        self._output_dir = output_dir
        self.dataset = dataset
        mmcv.mkdir_or_exist(output_dir)
        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger(__name__)

        # if test objs are just a subset of train objs
        self.train_objs = train_objs
        self._predictions = {}
        self.dataset_name = dataset_name
        cur_sym_infos = {}  # label based key
        self.data_ref = ref.__dict__[dataset_name]
        self.obj_names = self.data_ref.objects
        #print(self.obj_names)
        self.obj_ids = [self.data_ref.obj2id[obj_name] for obj_name in self.obj_names]
        loaded_models_info = self.data_ref.get_models_info()
        for i, obj_name in enumerate(self.obj_names):
            obj_id = self.data_ref.obj2id[obj_name]
            model_info = loaded_models_info[str(obj_id)]
            if "symmetries_discrete" in model_info or "symmetries_continuous" in model_info:
                sym_transforms = misc.get_symmetry_transformations(model_info, max_sym_disc_step=0.01)
                sym_info = np.array([sym["R"] for sym in sym_transforms], dtype=np.float32)
            else:
                sym_info = None
            cur_sym_infos[i] = sym_info
        self.sym_infos = cur_sym_infos

        self.model_paths = [
            osp.join(self.data_ref.model_eval_dir, "obj_{:06d}.ply".format(obj_id)) for obj_id in self.obj_ids
        ]
        #print(self.model_paths)
        self.diameters = [self.data_ref.diameters[self.data_ref.objects.index(obj_name)] for obj_name in self.obj_names]
        self.models_3d = [
            inout.load_ply(model_path, vertex_scale=self.data_ref.vertex_scale) for model_path in self.model_paths
        ]
  
        self.eval_precision = cfg.VAL.get("EVAL_PRECISION", False)
        self._logger.info(f"eval precision: {self.eval_precision}")
        self.use_cache = False

    def reset(self):
        self._predictions = OrderedDict()

    def _maybe_adapt_label_cls_name(self, label):
        if self.train_objs is not None:
            cls_name = self.obj_names[label]
            if cls_name not in self.train_objs:
                return None, None  # this class was not trained
            label = self.train_objs.index(cls_name)
        else:
            cls_name = self.obj_names[label]
        return label, cls_name

    def process(self, inputs, feat, out_dict):
        """
        Args:
            inputs: the inputs to a model.
                It is a list of dict. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name", "image_id", "scene_id".
            outputs:
        """
        cfg = self.cfg

        results = self.eval_pose_parallel(inputs,feat)

        # out_rots = out_dict["rot"].detach().to(self._cpu_device).numpy()
        # out_transes = out_dict["trans"].detach().to(self._cpu_device).numpy()
        inputs  = zip(
                inputs["file_name"], 
                inputs['cls_id'])

        out_i = -1
        for i, (_input, output) in enumerate(zip(inputs, out_dict)):
            start_process_time = time.perf_counter()
            
            out_i += 1
            file_name = _input[0]

            cls_name = self.data_ref.id2obj[_input[1].item()]

            # get pose
            pose_est = results['pose_est'][i]

            output["time"] += time.perf_counter() - start_process_time

            if cls_name not in self._predictions:
                self._predictions[cls_name] = OrderedDict()

            result = { "R": pose_est[:,:3], "t": pose_est[:,3:4], "time": output["time"],"det":results['det'][i]}
            self._predictions[cls_name][file_name] = result

    def evaluate(self):
        # bop toolkit eval in subprocess, no return value
        if self._distributed:
            synchronize()
            _predictions = all_gather(self._predictions)
            # NOTE: gather list of OrderedDict
            self._predictions = OrderedDict()
            for preds in _predictions:
                for _k, _v in preds.items():
                    self._predictions[_k] = _v
            # self._predictions = list(itertools.chain(*_predictions))
            if not is_main_process():
                return
        if self.eval_precision:
            return self._eval_predictions_precision()
        return self._eval_predictions()
        # return copy.deepcopy(self._eval_predictions())

    def get_gts(self):
        # NOTE: it is cached by dataset dicts loader
        self.gts = OrderedDict()

        instances = self.dataset.test_annos
        self._logger.info("load gts of {}".format(self.dataset_name))
        for inst in tqdm(instances):
            file_name = inst["file_name"]
        
            K = inst["cam"]
            
            obj_name = self.data_ref.id2obj[inst["obj_id"]]
            if obj_name not in self.gts:
                self.gts[obj_name] = OrderedDict()
            self.gts[obj_name][file_name] = {"R": inst['pose'][:3,:3], "t": inst['pose'][:3,3:4], "K": K}

    
    
    def eval_pose_parallel(self,
        annos,features
    ):
        result = OrderedDict()
        
        bs,_,_ = annos["cld_rgb_nrm"].shape
        #mesh = features['mesh'].unsqueeze(0).repeat(bs,1,1).transpose(1,2)

        data_gen = zip(
                annos["cld_rgb_nrm"],
                features['seg'],
                features['mesh'], 
                features['rgbd'],
                annos['cls_id'],
                annos['det']
                )
        
        result['pose_est'] = []
        result['det'] = []
        
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=bs
        ) as executor:
            eval_func = cal_frame_poses
           
            for i,res in enumerate(executor.map(eval_func, data_gen)):
            
               pose_est = res
               result['pose_est'].append(pose_est)
               result['det'].append(annos['det'][i])

        return result
                

    def _eval_predictions(self):
        """Evaluate self._predictions on 6d pose.

        Return results with the metrics of the tasks.
        """
        self._logger.info("Eval results ...")
        cfg = self.cfg

        recalls = OrderedDict()
        errors = OrderedDict()
        self.get_gts()

        error_names = ["ad", "re", "te", "proj"]
        metric_names = [
            "ad_2",
            "ad_5",
            "ad_10",
            "ad_0.1",
            "rete_2",
            "rete_5",
            "rete_10",
            "re_2",
            "re_5",
            "re_10",
            "te_2",
            "te_5",
            "te_10",
            "proj_2",
            "proj_5",
            "proj_10",
        ]
        lines = ['scene_id,im_id,obj_id,score,R,t,time']
        for obj_name in self.gts:
            if obj_name not in self._predictions:
                continue
            cur_label = self.obj_names.index(obj_name)
            if obj_name not in recalls:
                recalls[obj_name] = OrderedDict()
                for metric_name in metric_names:
                    recalls[obj_name][metric_name] = []

            if obj_name not in errors:
                errors[obj_name] = OrderedDict()
                for err_name in error_names:
                    errors[obj_name][err_name] = []

            #################
            obj_gts = self.gts[obj_name]
            obj_preds = self._predictions[obj_name]
            for file_name, gt_anno in obj_gts.items():
                if file_name not in obj_preds:  # no pred found
                    for metric_name in metric_names:
                        recalls[obj_name][metric_name].append(0.0)
                    continue
                # compute each metric
                R_pred = obj_preds[file_name]["R"]
                t_pred = obj_preds[file_name]["t"]
                cost_time = obj_preds[file_name]["time"]
                t_mm_pre=t_pred*1000
                R_gt = gt_anno["R"]
                t_gt = gt_anno["t"]
                lines.append('{scene_id},{im_id},{obj_id},{score},{R},{t},{time}'.format(
                    scene_id=int(file_name.split("/")[0]),
                    im_id=file_name.split("/")[-1],
                    obj_id=self.data_ref.obj2id[obj_name],
                    score=-1,
                    R=' '.join(map(str, R_pred.flatten().tolist())),
                    t=' '.join(map(str, t_mm_pre.flatten().tolist())),
                    time=-1))

                t_error = te(t_pred, t_gt)

                if obj_name in cfg.DATASETS["SYM_OBJS"]:
                    R_gt_sym = get_closest_rot(R_pred, R_gt, self.sym_infos[cur_label])
                    r_error = re(R_pred, R_gt_sym)

                    proj_2d_error = arp_2d(
                        R_pred, t_pred, R_gt_sym, t_gt, pts=self.models_3d[cur_label]["pts"], K=gt_anno["K"]
                    )

                    ad_error = adi(
                        R_pred, t_pred, R_gt, t_gt, pts=self.models_3d[self.obj_names.index(obj_name)]["pts"]
                    )
                else:
                    r_error = re(R_pred, R_gt)

                    proj_2d_error = arp_2d(
                        R_pred, t_pred, R_gt, t_gt, pts=self.models_3d[cur_label]["pts"], K=gt_anno["K"]
                    )

                    ad_error = add(
                        R_pred, t_pred, R_gt, t_gt, pts=self.models_3d[self.obj_names.index(obj_name)]["pts"]
                    )

                #########
                errors[obj_name]["ad"].append(ad_error)
                errors[obj_name]["re"].append(r_error)
                errors[obj_name]["te"].append(t_error)
                errors[obj_name]["proj"].append(proj_2d_error)
                ############
                recalls[obj_name]["ad_2"].append(float(ad_error < 0.02 * self.diameters[cur_label]))
                recalls[obj_name]["ad_5"].append(float(ad_error < 0.05 * self.diameters[cur_label]))
                recalls[obj_name]["ad_10"].append(float(ad_error < 0.1 * self.diameters[cur_label]))
                recalls[obj_name]["ad_0.1"].append(float(ad_error < 0.1)) #* self.diameters[cur_label]
                # deg, cm
                recalls[obj_name]["rete_2"].append(float(r_error < 2 and t_error < 0.02))
                recalls[obj_name]["rete_5"].append(float(r_error < 5 and t_error < 0.05))
                recalls[obj_name]["rete_10"].append(float(r_error < 10 and t_error < 0.1))

                recalls[obj_name]["re_2"].append(float(r_error < 2))
                recalls[obj_name]["re_5"].append(float(r_error < 5))
                recalls[obj_name]["re_10"].append(float(r_error < 10))

                recalls[obj_name]["te_2"].append(float(t_error < 0.02))
                recalls[obj_name]["te_5"].append(float(t_error < 0.05))
                recalls[obj_name]["te_10"].append(float(t_error < 0.1))
                # px
                recalls[obj_name]["proj_2"].append(float(proj_2d_error < 2))
                recalls[obj_name]["proj_5"].append(float(proj_2d_error < 5))
                recalls[obj_name]["proj_10"].append(float(proj_2d_error < 10))
            
        path="lib/bop_toolkit-master/gt_ycbv-test.csv"
        with open(path, 'w') as f:
                   f.write('\n'.join(lines))
        obj_names = sorted(list(recalls.keys()))
        header = ["objects"] + obj_names + [f"Avg({len(obj_names)})"]
        big_tab = [header]
        for metric_name in metric_names:
            line = [metric_name]
            this_line_res = []
            for obj_name in obj_names:
                res = recalls[obj_name][metric_name]
                if len(res) > 0:
                    line.append(f"{100 * np.mean(res):.2f}")
                    this_line_res.append(np.mean(res))
                else:
                    line.append(0.0)
                    this_line_res.append(0.0)
            # average
            if len(obj_names) > 0:
                line.append(f"{100 * np.mean(this_line_res):.2f}")
            big_tab.append(line)

        for error_name in ["re", "te"]:
            line = [error_name]
            this_line_res = []
            for obj_name in obj_names:
                res = errors[obj_name][error_name]
                if len(res) > 0:
                    line.append(f"{np.mean(res):.2f}")
                    this_line_res.append(np.mean(res))
                else:
                    line.append(float("nan"))
                    this_line_res.append(float("nan"))
            # mean
            if len(obj_names) > 0:
                line.append(f"{np.mean(this_line_res):.2f}")
            big_tab.append(line)
        ### log big tag
        self._logger.info("recalls")
        res_log_tab_str = tabulate(
            big_tab,
            tablefmt="plain",
            # floatfmt=floatfmt
        )
        self._logger.info("\n{}".format(res_log_tab_str))
        errors_cache_path = osp.join(self._output_dir, f"_{self.dataset_name}_errors.pkl")
        recalls_cache_path = osp.join(self._output_dir, f"_{self.dataset_name}_recalls.pkl")
        mmcv.dump(errors, errors_cache_path)
        mmcv.dump(recalls, recalls_cache_path)

        dump_tab_name = osp.join(self._output_dir, f"_{self.dataset_name}_tab.txt")
        with open(dump_tab_name, "w") as f:
            f.write("{}\n".format(res_log_tab_str))

        if self._distributed:
            self._logger.warning("\n The current evaluation on multi-gpu is not correct, run with single-gpu instead.")

        return {}

    def _eval_predictions_precision(self):
        """NOTE: eval precision instead of recall
        Evaluate self._predictions on 6d pose.
        Return results with the metrics of the tasks.
        """
        self._logger.info("Eval results ...")
        cfg = self.cfg
        method_name = f"{cfg.EXP_ID.replace('_', '-')}"
        cache_path = osp.join(self._output_dir, f"{method_name}_{self.dataset_name}_preds.pkl")
        if osp.exists(cache_path) and self.use_cache:
            self._logger.info("load cached predictions")
            self._predictions = mmcv.load(cache_path)
        else:
            if hasattr(self, "_predictions"):
                mmcv.dump(self._predictions, cache_path)
            else:
                raise RuntimeError("Please run inference first")

        precisions = OrderedDict()
        errors = OrderedDict()
        self.get_gts()

        error_names = ["ad", "re", "te", "proj"]
        metric_names = [
            "ad_2",
            "ad_5",
            "ad_10",
            "rete_2",
            "rete_5",
            "rete_10",
            "re_2",
            "re_5",
            "re_10",
            "te_2",
            "te_5",
            "te_10",
            "proj_2",
            "proj_5",
            "proj_10",
        ]

        for obj_name in self.gts:
            if obj_name not in self._predictions:
                continue
            cur_label = self.obj_names.index(obj_name)
            if obj_name not in precisions:
                precisions[obj_name] = OrderedDict()
                for metric_name in metric_names:
                    precisions[obj_name][metric_name] = []

            if obj_name not in errors:
                errors[obj_name] = OrderedDict()
                for err_name in error_names:
                    errors[obj_name][err_name] = []

            #################
            obj_gts = self.gts[obj_name]
            obj_preds = self._predictions[obj_name]
            for file_name, gt_anno in obj_gts.items():
                # compute precision as in DPOD paper
                if file_name not in obj_preds:  # no pred found
                    # NOTE: just ignore undetected
                    continue
                # compute each metric
                R_pred = obj_preds[file_name]["R"]
                t_pred = obj_preds[file_name]["t"]

                R_gt = gt_anno["R"]
                t_gt = gt_anno["t"]

                t_error = te(t_pred, t_gt)

                if obj_name in cfg.DATASETS.SYM_OBJS:
                    R_gt_sym = get_closest_rot(R_pred, R_gt, self._metadata.sym_infos[cur_label])
                    r_error = re(R_pred, R_gt_sym)

                    proj_2d_error = arp_2d(
                        R_pred, t_pred, R_gt_sym, t_gt, pts=self.models_3d[cur_label]["pts"], K=gt_anno["K"]
                    )

                    ad_error = adi(
                        R_pred, t_pred, R_gt, t_gt, pts=self.models_3d[self.obj_names.index(obj_name)]["pts"]
                    )
                else:
                    r_error = re(R_pred, R_gt)

                    proj_2d_error = arp_2d(
                        R_pred, t_pred, R_gt, t_gt, pts=self.models_3d[cur_label]["pts"], K=gt_anno["K"]
                    )

                    ad_error = add(
                        R_pred, t_pred, R_gt, t_gt, pts=self.models_3d[self.obj_names.index(obj_name)]["pts"]
                    )

                #########
                errors[obj_name]["ad"].append(ad_error)
                errors[obj_name]["re"].append(r_error)
                errors[obj_name]["te"].append(t_error)
                errors[obj_name]["proj"].append(proj_2d_error)
                ############
                precisions[obj_name]["ad_2"].append(float(ad_error < 0.02 * self.diameters[cur_label]))
                precisions[obj_name]["ad_5"].append(float(ad_error < 0.05 * self.diameters[cur_label]))
                precisions[obj_name]["ad_10"].append(float(ad_error < 0.1 * self.diameters[cur_label]))
                # deg, cm
                precisions[obj_name]["rete_2"].append(float(r_error < 2 and t_error < 0.02))
                precisions[obj_name]["rete_5"].append(float(r_error < 5 and t_error < 0.05))
                precisions[obj_name]["rete_10"].append(float(r_error < 10 and t_error < 0.1))

                precisions[obj_name]["re_2"].append(float(r_error < 2))
                precisions[obj_name]["re_5"].append(float(r_error < 5))
                precisions[obj_name]["re_10"].append(float(r_error < 10))

                precisions[obj_name]["te_2"].append(float(t_error < 0.02))
                precisions[obj_name]["te_5"].append(float(t_error < 0.05))
                precisions[obj_name]["te_10"].append(float(t_error < 0.1))
                # px
                precisions[obj_name]["proj_2"].append(float(proj_2d_error < 2))
                precisions[obj_name]["proj_5"].append(float(proj_2d_error < 5))
                precisions[obj_name]["proj_10"].append(float(proj_2d_error < 10))

        # summarize
        obj_names = sorted(list(precisions.keys()))
        header = ["objects"] + obj_names + [f"Avg({len(obj_names)})"]
        big_tab = [header]
        for metric_name in metric_names:
            line = [metric_name]
            this_line_res = []
            for obj_name in obj_names:
                res = precisions[obj_name][metric_name]
                if len(res) > 0:
                    line.append(f"{100 * np.mean(res):.2f}")
                    this_line_res.append(np.mean(res))
                else:
                    line.append(0.0)
                    this_line_res.append(0.0)
            # mean
            if len(obj_names) > 0:
                line.append(f"{100 * np.mean(this_line_res):.2f}")
            big_tab.append(line)

        for error_name in ["re", "te"]:
            line = [error_name]
            this_line_res = []
            for obj_name in obj_names:
                res = errors[obj_name][error_name]
                if len(res) > 0:
                    line.append(f"{np.mean(res):.2f}")
                    this_line_res.append(np.mean(res))
                else:
                    line.append(float("nan"))
                    this_line_res.append(float("nan"))
            # mean
            if len(obj_names) > 0:
                line.append(f"{np.mean(this_line_res):.2f}")
            big_tab.append(line)
        ### log big table
        print(big_tab)
        self._logger.info("precisions")
        res_log_tab_str = tabulate(
            big_tab,
            tablefmt="plain",
            # floatfmt=floatfmt
        )
        self._logger.info("\n{}".format(res_log_tab_str))
        errors_cache_path = osp.join(self._output_dir, f"{method_name}_{self.dataset_name}_errors.pkl")
        recalls_cache_path = osp.join(self._output_dir, f"{method_name}_{self.dataset_name}_precisions.pkl")
        self._logger.info(f"{errors_cache_path}")
        self._logger.info(f"{recalls_cache_path}")
        mmcv.dump(errors, errors_cache_path)
        mmcv.dump(precisions, recalls_cache_path)

        dump_tab_name = osp.join(self._output_dir, f"{method_name}_{self.dataset_name}_tab_precisions.txt")
        with open(dump_tab_name, "w") as f:
            f.write("{}\n".format(res_log_tab_str))
        if self._distributed:
            self._logger.warning("\n The current evaluation on multi-gpu is not correct, run with single-gpu instead.")
        return {}
