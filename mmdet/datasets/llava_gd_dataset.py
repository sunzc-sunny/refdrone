# Copyright (c) OpenMMLab. All rights reserved.
import json
import os.path as osp
from typing import List, Optional

from mmengine.fileio import get_local_path

from mmdet.registry import DATASETS
from .base_det_dataset import BaseDetDataset

@DATASETS.register_module()
class LLavaVGDataset(BaseDetDataset):
    """object detection and visual grounding dataset."""

    def __init__(self,
                 *args,
                 data_root: str = '',
                 label_map_file: Optional[str] = None,
                 need_text: bool = True,
                 **kwargs) -> None:
        self.dataset_mode = 'VG'
        self.need_text = need_text

        super().__init__(*args, data_root=data_root, **kwargs)
        assert self.return_classes is True

    def load_data_list(self) -> List[dict]:
        with get_local_path(
                self.ann_file, backend_args=self.backend_args) as local_path:
            with open(local_path, 'r') as f:
                data_list = [json.loads(line) for line in f]

        out_data_list = []
        for data in data_list:
            data_info = {}
            img_path = osp.join(self.data_prefix['img'], data['filename'])
            data_info['img_path'] = img_path
            data_info['height'] = data['height']
            data_info['width'] = data['width']
            anno = data['grounding']
            # qs = "<image>\n[refer] Give me the location of <p> " +  anno['caption'] + " </p>"
            # conv = conv_templates["v1"].copy()
            # conv.append_message(conv.roles[0], qs)
            # conv.append_message(conv.roles[1], None)
            # source = conv.get_prompt()
            data_info['text'] = anno['caption']
            # data_info {'img_path': '/mnt/public/usr/sunzhichao/VisDrone2019/all_image/9999962_00000_d_0000123.jpg', 'height': 1050, 'width': 1400, 'text': 'The car parked on the left side of the street.'}
            regions = anno['regions']
            # qs = data_info['text'].lower()
            # qs = "<image>\n[refer] Give me the location of <p> " + data_info['text'] + " </p>"

            instances = []
            phrases = {}
            for i, region in enumerate(regions):
                bbox = region['bbox']
                phrase = region['phrase']
                tokens_positive = region['tokens_positive']
                if not isinstance(bbox[0], list):
                    bbox = [bbox]
                for box in bbox:
                    instance = {}
                    x1, y1, x2, y2 = box
                    inter_w = max(0, min(x2, data['width']) - max(x1, 0))
                    inter_h = max(0, min(y2, data['height']) - max(y1, 0))
                    if inter_w * inter_h == 0:
                        continue
                    if (x2 - x1) < 1 or (y2 - y1) < 1:
                        continue
                    instance['ignore_flag'] = 0
                    instance['bbox'] = box
                    instance['bbox_label'] = i
                    phrases[i] = {
                        'phrase': phrase,
                        'tokens_positive': tokens_positive
                    }
                    instances.append(instance)
            data_info['instances'] = instances
            data_info['phrases'] = phrases
            data_info['dataset_mode'] = self.dataset_mode
            out_data_list.append(data_info)
        del data_list
        return out_data_list
