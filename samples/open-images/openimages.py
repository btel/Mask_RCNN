import glob
from collections import defaultdict
import pandas as pd
import numpy as np
import skimage
import sys
import os

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR) 

from mrcnn.utils import Dataset

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
                
             
        # load images
        image_paths = glob.glob(os.path.join(path, subset, 'images', '*.jpg')) 
        for image_path in image_paths:
            _, filename = os.path.split(image_path)
            image_id, _ = os.path.splitext(filename)
            if image_id in self._annotations:
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
        masks = np.stack(masks, axis=2).astype(bool)
        return masks, np.array(classes)
        
        
        