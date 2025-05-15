# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from typing import List

from pycocotools.coco import COCO

from mmdet.registry import DATASETS
from .base_det_dataset import BaseDetDataset


def convert_phrase_ids(phrase_ids: list) -> list:
    unique_elements = sorted(set(phrase_ids))
    element_to_new_label = {
        element: label
        for label, element in enumerate(unique_elements)
    }
    phrase_ids = [element_to_new_label[element] for element in phrase_ids]
    return phrase_ids


def merge_tokens(tokens_positive):
    """
    合并连续或相邻的token范围
    
    Args:
        tokens_positive: 列表of[start, end]范围
        
    Returns:
        合并后的[start, end]范围
    """
    if not tokens_positive:
        return []
    
    # 按开始位置排序
    sorted_tokens = sorted(tokens_positive, key=lambda x: x[0])
    
    # 初始化结果为第一个范围
    merged = [sorted_tokens[0][0], sorted_tokens[0][1]]
    
    # 遍历并合并重叠或相邻的范围
    for start, end in sorted_tokens[1:]:
        # 如果当前范围与合并范围相邻或重叠
        if start <= merged[1] + 1:
            merged[1] = max(merged[1], end)
        else:
            # 如果不相邻，说明有了间隔
            break
    
    return merged

@DATASETS.register_module()
class RefDroneFlickr30kDataset(BaseDetDataset):
    """Flickr30K Dataset."""

    def load_data_list(self) -> List[dict]:

        self.coco = COCO(self.ann_file)

        self.ids = sorted(list(self.coco.imgs.keys()))

        data_list = []
        for img_id in self.ids:
            if isinstance(img_id, str):
                ann_ids = self.coco.getAnnIds(imgIds=[img_id], iscrowd=None)
            else:
                ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=None)

            coco_img = self.coco.loadImgs(img_id)[0]

            caption = coco_img['caption']
            file_name = coco_img['file_name']
            img_path = osp.join(self.data_prefix['img'], file_name)
            width = coco_img['width']
            height = coco_img['height']
            annos = self.coco.loadAnns(ann_ids)
            tokens_positive = annos[0]['tokens_positive']
            tokens_positive = merge_tokens(tokens_positive)
            # tokens_positive = coco_img['tokens_positive_eval']
            phrases = [caption[tokens_positive[0]:tokens_positive[1]]]
            phrase_ids = []

            instances = []
            for anno in annos:
                instance = {
                    'bbox': [
                        anno['bbox'][0], anno['bbox'][1],
                        anno['bbox'][0] + anno['bbox'][2],
                        anno['bbox'][1] + anno['bbox'][3]
                    ],
                    'bbox_label':
                    anno['category_id'],
                    'ignore_flag':
                    anno['iscrowd']
                }
                phrase_ids.append(0)
                instances.append(instance)

            phrase_ids = convert_phrase_ids(phrase_ids)

            data_list.append(
                dict(
                    img_path=img_path,
                    img_id=img_id,
                    height=height,
                    width=width,
                    instances=instances,
                    text=caption,
                    phrase_ids=phrase_ids,
                    tokens_positive=[[tokens_positive]],
                    phrases=phrases,
                ))

        return data_list
