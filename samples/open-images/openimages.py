import glob
from collections import defaultdict
import pandas as pd
import numpy as np
import skimage
import sys
import os
from mrcnn.config import Config
        
import base64
from pycocotools import _mask as coco_mask
import typing as t
import zlib

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR) 

from mrcnn.utils import Dataset
from mrcnn import utils

class OpenImagesConfig(Config):
    """
    """
    # Give the configuration a recognizable name
    NAME = "openimages"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 602  # background + 3 shapes

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 1024
    IMAGE_MAX_DIM = 1024

class OpenImageDataset(Dataset):
        
    def load_dataset(self, path, subset):
        self._path = path
        self._subset = subset
        
        # load classes
        class_descrs = pd.read_csv(
            os.path.join(path, 'class-descriptions-boxable.csv'),
            names=['class_id','class_name'])
        for index, row in class_descrs.iterrows(): 
            self.add_class('openimages', row['class_id'], row['class_name'])
   
            
        # load annotations
        self._annotations = defaultdict(list)
        try:
            with open(os.path.join(path, subset, subset+'-annotations-object-segmentation.csv')) as fid:
                fid.readline()
                row = fid.readline()
                while row: 
                    mask_path, image_id, label, box_id, xmin, xmax, ymin, ymax, _, _ = row.split(',')
                    if image_id.startswith('0'): # for testing
                        self._annotations[image_id].append(
                            {'label': label, 
                             'mask_path': mask_path,
                             'bounding_box': (xmin, xmax, ymin, ymax)})

                    row = fid.readline()
        except IOError:
            pass
                
             
        # load images
        image_paths = glob.glob(os.path.join(path, subset, 'images', '*.jpg')) 
        for image_path in image_paths:
            _, filename = os.path.split(image_path)
            image_id, _ = os.path.splitext(filename)
            self.add_image('openimages', image_id, image_path)
                
    def load_image(self, image_id):
        image = super().load_image(image_id)
        self.image_info[image_id]['size'] = image.shape[:2]
        return image
        
                
    def load_mask(self, i):
        image_id = self.image_info[i]['id']
        size = self.image_info[i]['size']
        annotations = self._annotations[image_id]
        masks = []
        classes = []
        for mask in annotations:
            mask_path = os.path.join(self._path, self._subset, 'masks', mask['mask_path'])
            mask_image = skimage.io.imread(mask_path)
            mask_image = skimage.transform.resize(mask_image, size)
            masks.append(mask_image)
            classes.append(self.map_source_class_id('openimages.'+mask['label']))
        if not masks:
            return np.array([[]]), np.array([])
        masks = np.stack(masks, axis=2).astype(bool)
        return masks, np.array(classes)
        

def encode_binary_mask(mask: np.ndarray) -> t.Text:
 """Converts a binary mask into OID challenge encoding ascii text."""

 # check input mask --
 if mask.dtype != np.bool:
   raise ValueError(
       "encode_binary_mask expects a binary mask, received dtype == %s" %
       mask.dtype)

 mask = np.squeeze(mask)
 if len(mask.shape) != 2:
   raise ValueError(
       "encode_binary_mask expects a 2d mask, received shape == %s" %
       mask.shape)

 # convert input mask to expected COCO API input --
 mask_to_encode = mask.reshape(mask.shape[0], mask.shape[1], 1)
 mask_to_encode = mask_to_encode.astype(np.uint8)
 mask_to_encode = np.asfortranarray(mask_to_encode)

 # RLE encode mask --
 encoded_mask = coco_mask.encode(mask_to_encode)[0]["counts"]

 # compress and base64 encoding --
 binary_str = zlib.compress(encoded_mask, zlib.Z_BEST_COMPRESSION)
 base64_str = base64.b64encode(binary_str)
 return base64_str

def unmold_mask(mask, bbox, image_size):
    "convert mask to original dimensions"
    nx, ny = image_size
    ratio = nx / ny
    new_nx, new_ny = mask.shape
    if ratio < 1:
        crop_x = int((new_nx - ratio * new_ny)/2)
        cropped_mask = mask[crop_x:-crop_x, :]
        crop_bbox = [max(bbox[0] - crop_x, 0), bbox[1],
                     max(bbox[2] - crop_x, 0), bbox[3]]

    else:
        crop_y = int((new_ny - new_nx / ratio)/2)
        cropped_mask = mask[:, crop_y:-crop_y]
        crop_bbox = [bbox[0], max(bbox[1] - crop_y, 0),
                     bbox[2], bbox[3] - crop_y]

    true_mask = utils.resize(cropped_mask, image_size).astype(bool)
    true_bbox = (np.array(crop_bbox) * image_size[0] / cropped_mask.shape[0]).astype(int)
    return true_mask, true_bbox

