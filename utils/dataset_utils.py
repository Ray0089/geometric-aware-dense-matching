import copy
import logging
import numpy as np
import operator
import pickle
import random
import mmcv
import torch
import torch.multiprocessing as mp
import torch.utils.data as data
from torch.utils.data import dataloader
import json
import cv2
logger = logging.getLogger(__name__)


def flat_dataset_dicts(dataset_dicts):
    """
    flatten the dataset dicts of detectron2 format
    original: list of dicts, each dict contains some image-level infos
              and an "annotations" field for instance-level infos of multiple instances
    => flat the instance level annotations
    flat format:
        list of dicts,
            each dict includes the image/instance-level infos
            an `inst_id` of a single instance,
            `inst_infos` includes only one instance
    """
    new_dicts = []
    for dataset_dict in dataset_dicts:
        img_infos = {_k: _v for _k, _v in dataset_dict.items() if _k not in ["annotations"]}
        if "annotations" in dataset_dict:
            for inst_id, anno in enumerate(dataset_dict["annotations"]):
                rec = {"inst_id": inst_id, "inst_infos": anno}
                rec.update(img_infos)
                new_dicts.append(rec)
        else:
            rec = img_infos
            new_dicts.append(rec)
    return new_dicts


def filter_invalid_in_dataset_dicts(dataset_dicts, visib_thr=0.0):
    """
    filter invalid instances in the dataset_dicts (for train)
    Args:
        visib_thr:
    """
    num_filtered = 0
    new_dicts = []
    for dataset_dict in dataset_dicts:
        new_dict = {_k: _v for _k, _v in dataset_dict.items() if _k not in ["annotations"]}
        if "annotations" in dataset_dict:
            new_annos = []
            for inst_id, anno in enumerate(dataset_dict["annotations"]):
                if anno.get("visib_fract", 1.0) > visib_thr:
                    new_annos.append(anno)
                else:
                    num_filtered += 1
            if len(new_annos) == 0:
                continue
            new_dict["annotations"] = new_annos

        new_dicts.append(new_dict)
    if num_filtered > 0:
        logger.warning(f"filtered out {num_filtered} instances with visib_fract <= {visib_thr}")
    return new_dicts


def trivial_batch_collator(batch):
    """A batch collator that does nothing.

    https://github.com/pytorch/fairseq/issues/1171
    """
    dataloader._use_shared_memory = False
    return batch


def filter_empty_dets(dataset_dicts):
    """
    Filter out images with empty detections
    NOTE: here we assume detections are in "annotations"
    Args:
        dataset_dicts (list[dict]): annotations in Detectron2 Dataset format.

    Returns:
        list[dict]: the same format, but filtered.
    """
    num_before = len(dataset_dicts)

    def valid(anns):
        if len(anns) > 0:
            return True
        # for ann in anns:
        #     if ann.get("iscrowd", 0) == 0:
        #         return True
        return False

    dataset_dicts = [x for x in dataset_dicts if valid(x["annotations"])]
    num_after = len(dataset_dicts)
    if num_after < num_before:
        logger = logging.getLogger(__name__)
        logger.warning(
            "Removed {} images with empty detections. {} images left.".format(num_before - num_after, num_after)
        )
    return dataset_dicts


def crop_resize_by_warp_affine(img, center, scale, output_size, rot=0, interpolation=cv2.INTER_LINEAR):
    """
    output_size: int or (w, h)
    NOTE: if img is (h,w,1), the output will be (h,w)
    """
    if isinstance(scale, (int, float)):
        scale = (scale, scale)
    if isinstance(output_size, int):
        output_size = (output_size, output_size)
    trans = get_affine_transform(center, scale, rot, output_size)

    dst_img = cv2.warpAffine(img, trans, (int(output_size[0]), int(output_size[1])), flags=interpolation)

    return dst_img


def get_affine_transform(center, scale, rot, output_size, shift=np.array([0, 0], dtype=np.float32), inv=False):
    """
    adapted from CenterNet: https://github.com/xingyizhou/CenterNet/blob/master/src/lib/utils/image.py
    center: ndarray: (cx, cy)
    scale: (w, h)
    rot: angle in deg
    output_size: int or (w, h)
    """
    if isinstance(center, (tuple, list)):
        center = np.array(center, dtype=np.float32)

    if isinstance(scale, (int, float)):
        scale = np.array([scale, scale], dtype=np.float32)

    if isinstance(output_size, (int, float)):
        output_size = (output_size, output_size)

    scale_tmp = scale
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.0], dtype=np.float32).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result
