from mmengine.runner import Runner
from mmengine.registry import init_default_scope

init_default_scope('mmdet')

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os

def visualize_detection(data_batch, save_dir='./work_dirs/vis_results'):
    os.makedirs(save_dir, exist_ok=True)

    inputs = data_batch['inputs']
    data_samples = data_batch['data_samples']
    
    for img_idx, (img_tensor, data_sample) in enumerate(zip(inputs, data_samples)):

        img_np = img_tensor.cpu().permute(1, 2, 0).numpy().astype(np.uint8)
        

        img_pil = Image.fromarray(img_np)
        draw = ImageDraw.Draw(img_pil)
        
        try:

            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()
        

        text = data_sample.text
        bboxes = data_sample.gt_instances.bboxes.tensor.cpu().numpy()
        image_path = data_sample.img_path

        for bbox in bboxes:
            x1, y1, x2, y2 = bbox.astype(int)
            draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=3)
        

        img_width, img_height = img_pil.size
        text_position = (10, img_height - 40)
        
        text_bg = Image.new('RGBA', (img_width, 40), (0, 0, 0, 128))
        img_pil.paste(text_bg, (0, img_height - 40), text_bg)
        

        draw.text(text_position, text, fill=(255, 255, 255), font=font)


        save_path = os.path.join(save_dir, f'{img_idx}_img_{os.path.basename(image_path)}.jpg')
        img_pil.convert('RGB').save(save_path)
        print(f'Saved visualization to {save_path}')



test_pipeline = [
    dict(
        type='LoadImageFromFile', backend_args=None,
        imdecode_backend='pillow'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'text', 'custom_entities',
                   'tokens_positive'))
]

test_dataset = dict(
    type='MDETRStyleRefCocoDataset',
    ann_file='datasets/RefDrone_test_mdetr.json',
    data_prefix=dict(img='datasets/VisDrone2019/all_image/'),
    test_mode=True,
    return_classes=True,
    pipeline=test_pipeline,
    backend_args=None)


dataloader_cfg = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=test_dataset)

# Build dataloader
test_data_loader = Runner.build_dataloader(dataloader_cfg)

# Test the dataloader
for batch_idx, data_batch in enumerate(test_data_loader):
    visualize_detection(data_batch)
    exit()
