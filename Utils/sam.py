import os
import time

import torch
import numpy as np

from PIL import Image

from rembg import remove
from segment_anything import sam_model_registry, SamPredictor


def sam_init(sam_ckpt_path):
    sam_checkpoint = sam_ckpt_path
    model_type = "vit_h"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint).to(device=f"cuda")
    predictor = SamPredictor(sam)
    return predictor

def sam_segment(predictor, input_image, *bbox_coords):
    bbox = np.array(bbox_coords)
    image = np.asarray(input_image)

    start_time = time.time()
    predictor.set_image(image)

    masks_bbox, scores_bbox, logits_bbox = predictor.predict(
        box=bbox,
        multimask_output=True
    )

    print(f"SAM Time: {time.time() - start_time:.3f}s")
    out_image = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
    out_image[:, :, :3] = image
    out_image_bbox = out_image.copy()
    out_image_bbox[:, :, 3] = masks_bbox[-1].astype(np.uint8) * 255

    del predictor
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

    return Image.fromarray(out_image_bbox, mode='RGBA'), masks_bbox

def remove_bg(sam_predictor, img):
    image_rem = img.convert('RGBA') #
    image_nobg = remove(image_rem, alpha_matting=True)
    arr = np.asarray(image_nobg)[:,:,-1]
    x_nonzero = np.nonzero(arr.sum(axis=0))
    y_nonzero = np.nonzero(arr.sum(axis=1))
    x_min = int(x_nonzero[0].min())
    y_min = int(y_nonzero[0].min())
    x_max = int(x_nonzero[0].max())
    y_max = int(y_nonzero[0].max())
    masked_image, mask = sam_segment(sam_predictor, img.convert('RGB'), x_min, y_min, x_max, y_max)
    
    return masked_image
