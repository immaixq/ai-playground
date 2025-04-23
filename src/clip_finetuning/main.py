# main.py
import torch
import torch.optim as optim
import torch.nn as nn
import clip
import logging
import os
import random
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

import configs.clip_supercon_config as config
from datasets.clip_supcon_dataset import ImageLabelDataset
from models.supcon_clip import (
    SupConClipModel,
    unfreeze_visual_encoder_layers,
    contrastive_loss,
)
from utils.helper import split_data

os.makedirs(config.OUTPUT_DIR, exist_ok=True)

log_file_path = os.path.join(config.OUTPUT_DIR, config.LOG_FILENAME)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.FileHandler(log_file_path, mode="w"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)  # Main script logger

torch.manual_seed(config.SEED)
random.seed(config.SEED)
np.random.seed(config.SEED)
device = torch.device(config.DEVICE)
logger.info(f"Using device: {device}")
logger.info(f"Using random seed: {config.SEED}")


def collate_fn_skip_none(batch):
    """Collate function that filters out None values."""
    batch = list(filter(lambda x: x is not None, batch))
    if not batch:
        return None
    return torch.utils.data.dataloader.default_collate(batch)


# --- Main Training Function ---
def main():
    logger.info(f"Loading CLIP model: {config.CLIP_MODEL_NAME}")
    try:
        clip_model, preprocess = clip.load(config.CLIP_MODEL_NAME, device=device)
        visual_model = clip_model.visual

        try:
            encoder_output_dim = visual_model.output_dim

        except AttributeError:
            try:
                encoder_output_dim = visual_model.ln_post.normalized_shape[0]
            except AttributeError:
                encoder_output_dim = 512
                logger.warning("VERIFY encoder_output_dim!")

        logger.info(f"CLIP model loaded. Encoder output dim: {encoder_output_dim}")
    except Exception as e:
        logger.error(f"Failed to load CLIP: {e}", exc_info=True)
        return

    projector_hidden_dim = 512
    projector_output_dim = 128

    projector = nn.Sequential(
        nn.Linear(encoder_output_dim, projector_hidden_dim),
        nn.ReLU(),
        nn.Linear(projector_hidden_dim, projector_output_dim),
    ).to(device)

    model = SupConClipModel(visual_model, projector).to(device)

    unfreeze_visual_encoder_layers(visual_model, config.UNFREEZE_LAST_N_BLOCKS)

    logger.info("Preparing data...")
    try:
        train_paths, val_paths = split_data(
            config.IMG_DIR, config.VALIDATION_SPLIT, config.SEED
        )
        if not train_paths:
            raise ValueError("No training data.")

        train_dataset = ImageLabelDataset(preprocess, train_paths)
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=True,
            num_workers=config.NUM_WORKERS,
            pin_memory=True,
            drop_last=True,
            persistent_workers=bool(config.NUM_WORKERS > 0),
            collate_fn=collate_fn_skip_none,  # Use custom collate
        )
        logger.info(f"Train DataLoader: {len(train_loader)} steps/epoch.")

        val_loader = DataLoader(
            ImageLabelDataset(preprocess, val_paths),
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=config.NUM_WORKERS,
            pin_memory=True,
            drop_last=False,
            persistent_workers=bool(config.NUM_WORKERS > 0),
            collate_fn=collate_fn_skip_none,
        )

    except Exception as e:
        logger.error(f"Data prep failed: {e}", exc_info=True)
        return

    params_to_train = list(
        filter(lambda p: p.requires_grad, model.parameters())
    )  # Filter on combined model
    if not params_to_train:
        logger.error("No trainable parameters!")
        return
    logger.info(
        f"Number of trainable parameters: {sum(p.numel() for p in params_to_train)}"
    )

    optimizer = optim.AdamW(
        params_to_train, lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )
    num_training_steps = config.EPOCHS * len(train_loader)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_training_steps
    )
    criterion = contrastive_loss

    logger.info("\n--- Starting Training ---")
    for epoch in range(config.EPOCHS):
        train_pbar = tqdm(
            train_loader, desc=f"Epoch {epoch + 1}/{config.EPOCHS}", leave=False
        )
        for batch in train_pbar:
            if batch is None:
                continue
            images, labels = batch

            try:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()

                embeddings = model(images)

                loss = criterion(embeddings, labels, temperature=config.TEMPERATURE)
                logger.info(f"Loss: {loss}")
                if torch.isnan(loss) or torch.isinf(loss):
                    continue

                # Backward, Clip, Step
                loss.backward()
                if config.GRAD_CLIP_NORM > 0:
                    torch.nn.utils.clip_grad_norm_(
                        params_to_train, config.GRAD_CLIP_NORM
                    )
                optimizer.step()
                scheduler.step()

            except Exception as e:
                logger.error(f"Training error: {e}", exc_info=True)
                continue
    logger.info("--- Training Finished ---")

    logger.info("\n--- Saving Model ---")
    try:
        torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
        logger.info(f"Model saved to: {config.MODEL_SAVE_PATH}")
    except Exception as e:
        logger.error(f"Failed to save model: {e}", exc_info=True)
    logger.info("--- Model Saved ---")


if __name__ == "__main__":
    main()
