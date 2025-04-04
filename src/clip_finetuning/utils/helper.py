import numpy as np
import torch.nn.functional as F
import torch
import logging
import random 
import os
import math
import glob

logger = logging.getLogger(__name__)

def split_data(data_dir, validation_split=0.0, seed=42):
    # (Implementation remains the same)
    # ...
    logger.info(f"Splitting data from: {data_dir} (Val split: {validation_split:.2f}, Seed: {seed})")
    random.seed(seed); train_paths = {}; val_paths = {}
    if not os.path.isdir(data_dir): logger.error(f"Data dir not found: {data_dir}"); raise FileNotFoundError(...)
    try: potential_classes = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    except OSError as e: logger.error(f"Listdir error in {data_dir}: {e}"); raise
    logger.info(f"Found {len(potential_classes)} potential class dirs. Processing...")
    valid_train_classes = 0; valid_val_classes = 0
    for i, cls_name in enumerate(potential_classes):
        class_dir = os.path.join(data_dir, cls_name)
        try: paths = glob.glob(os.path.join(class_dir, '*.jpg')) + glob.glob(os.path.join(class_dir, '*.png')) + glob.glob(os.path.join(class_dir, '*.jpeg'))
        except Exception as e: logger.warning(f"Glob error in {class_dir}: {e}. Skipping."); continue
        if len(paths) < 2: logger.warning(f"  - Skipping class '{cls_name}' (idx {i}) - needs >= 2 images, found {len(paths)}."); continue
        random.shuffle(paths)
        if validation_split > 0:
            min_train_needed = 2
            max_val_count = max(1, math.floor(len(paths) * validation_split))
            split_idx = len(paths) - max_val_count
            if split_idx < min_train_needed and len(paths) > min_train_needed: split_idx = min_train_needed
            current_train_paths = paths[:split_idx]; current_val_paths = paths[split_idx:]
            if len(current_train_paths) >= 1: train_paths[i] = current_train_paths
            else: logger.warning(f"  - Class '{cls_name}' (idx {i}) -> 0 training images.")
            if len(current_val_paths) >= 1: val_paths[i] = current_val_paths
            else: logger.warning(f"  - Class '{cls_name}' (idx {i}) -> 0 validation images.")
        else: train_paths[i] = paths
    valid_train_classes = len(train_paths); valid_val_classes = len(val_paths)
    total_train_images = sum(len(p) for p in train_paths.values()); total_val_images = sum(len(p) for p in val_paths.values())
    logger.info(f"Data split complete:")
    logger.info(f"  - Training: {valid_train_classes} classes, {total_train_images} images (Avg: {total_train_images / valid_train_classes if valid_train_classes else 0:.2f})")
    logger.info(f"  - Validation: {valid_val_classes} classes, {total_val_images} images (Avg: {total_val_images / valid_val_classes if valid_val_classes else 0:.2f})")
    if valid_train_classes == 0: logger.warning("No classes suitable for training.")
    if valid_val_classes == 0 and validation_split > 0: logger.warning("Validation requested, but no validation data.")
    return train_paths, val_paths

def normalize(features, axis=1):
    norms = np.linalg.norm(features, ord=2, axis=axis, keepdims=True)
    return features / norms

def create_mask_with_version1(labels, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    labels = labels.contiguous().view(-1, 1)
    print("labels", labels)
    mask = torch.eq(labels, labels.T).float().to(device)
    print("original mask", mask)
    self_mask = torch.scatter(
        torch.ones_like(mask),
        1,
        torch.arange(4).view(-1, 1).to(device),
        0
    )
    print("self mask", self_mask)
    mask = mask * self_mask
    print("final mask", mask)
    return mask

def create_mask_with_version2(labels):
    mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    mask.fill_diagonal_(0)
    return mask

def create_positive_mask(labels):
    mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    mask.fill_diagonal_(0)
    return mask

if __name__ == "__main__":
    vectors = [[0.800, 0.600], [0.936, 0.351], [-0.555, 0.832], [-0.707, 0.707]]
    vectors = np.array(vectors)
    normalized = normalize(vectors)

    a = torch.tensor([[0.800, 0.600], [0.936, 0.351], [-0.555, 0.832], [-0.707, 0.707]])
    s = F.normalize(a, dim=1)

    # similarity 
    res = a @ a.T

    similarity = torch.matmul(a, a.T)
    labels = torch.tensor([0, 0, 1, 1])

    positive_mask = create_positive_mask(labels)
    print(positive_mask)

    labels = labels.contiguous().view(-1, 1)
    mask = torch.eq(labels, labels.T).float()
    self_mask = torch.scatter(
        torch.ones_like(mask),
        1,
        torch.arange(4).view(-1, 1),
        0
    )
    mask = mask * self_mask
    logit_max, _ = torch.max(similarity, dim=1, keepdim=True)
    logits = similarity - logit_max.detach()
    exp_logits_all = torch.exp(logits) * self_mask
    exp_logits = torch.exp(logits) * self_mask
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)
    print("NUM POSITIVE PAIRS", mask.sum(1))

