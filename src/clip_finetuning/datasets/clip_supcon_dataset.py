import torch
from torch.utils.data import Dataset
from PIL import Image
import logging 

logger = logging.getLogger(__name__)

class ImageLabelDataset(Dataset):
    def __init__(self, preprocess_fn, image_path_dict):
        self.preprocess = preprocess_fn
        self.image_paths_by_class = image_path_dict
        self._image_list = [(idx, p) for idx, paths in image_path_dict.items() for p in paths]
        self.total_images = len(self._image_list)

    def __len__(self):
        return self.total_images
    
    def __getitem__(self, idx):
        actual_idx = idx%self.total_images
        class_idx, image_path = self._image_list[actual_idx]
        try:
            with Image.open(image_path) as img:
                image = img.convert("RGB")
        except Exception as img_e:
            logger.error(f"[Dataset] PIL Load/Convert Failed: {image_path}. Error: {img_e}")

        try:
            processed_image = self.preprocess(image)
        except Exception as prep_e:
                logger.error(f"[Dataset] Preprocessing failed for {image_path}: {prep_e}")
                return None
        
        return processed_image, torch.tensor(class_idx, dtype=torch.long)
    