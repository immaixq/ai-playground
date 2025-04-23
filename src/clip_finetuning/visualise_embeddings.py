import torch
from torch.utils.data import DataLoader
from datasets.clip_supcon_dataset import ImageLabelDataset
from models.supcon_clip import SupConClipModel
import clip
from torch import nn
from sklearn.manifold import TSNE
import umap
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image as PILImage

# --- Configuration ---
IMG_DIR = "/Users/maixueqiao/Downloads/project/ai-playground/data/magazines"
CUSTOM_WEIGHT_PATH = "./output_supcon/clip_supcon_model.pth"
DEVICE = "cpu"
BATCH_SIZE = 32
SEED = 42
N_COMPONENTS = 2
REDUCTION_METHOD = "UMAP"
CLIP_MODEL_NAME = "ViT-B/32"

# --- New Data Point Configuration ---
NEW_IMAGE_PATH = "/Users/maixueqiao/Downloads/project/ai-playground/src/python_web_scraper/NY_covers/50/50-1.jpg"  # <--- REPLACE WITH PATH TO NEW IMAGE
NEW_IMAGE_LABEL_INDEX = 0  

# --- Load CLIP and Preprocess ---
pretrained, preprocess = clip.load(CLIP_MODEL_NAME, device=DEVICE)
visual_encoder = pretrained.visual

# --- Instantiate and Load Your Model ---
projector = nn.Sequential(nn.Linear(visual_encoder.output_dim, 512), nn.ReLU(), nn.Linear(512, 128)).to(DEVICE)
model = SupConClipModel(visual_encoder, projector, device=DEVICE)
model.load_state_dict(torch.load(CUSTOM_WEIGHT_PATH, map_location=DEVICE))
model.eval()

# --- Prepare Dataset and DataLoader ---
def get_image_paths_by_class(img_dir):
    image_paths_by_class = {}
    class_names = sorted(os.listdir(img_dir))
    for i, class_name in enumerate(class_names):
        class_path = os.path.join(img_dir, class_name)
        if os.path.isdir(class_path):
            image_paths_by_class[i] = [os.path.join(class_path, f) for f in os.listdir(class_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
    return image_paths_by_class

image_paths_by_class = get_image_paths_by_class(IMG_DIR)
dataset = ImageLabelDataset(preprocess, image_paths_by_class)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# --- Get Embeddings and Labels for Existing Data ---
all_embeddings = []
all_labels = []

with torch.no_grad():
    for images, labels in dataloader:
        images = images.to(DEVICE)
        embeddings = model(images)
        embeddings = embeddings.cpu().numpy()
        labels = labels.cpu().numpy()
        all_embeddings.append(embeddings)
        all_labels.append(labels)

all_embeddings = np.concatenate(all_embeddings, axis=0)
all_labels = np.concatenate(all_labels, axis=0)

# --- Reduce Dimensionality ---
if REDUCTION_METHOD == "TSNE":
    reducer = TSNE(n_components=N_COMPONENTS, random_state=SEED, n_iter=300)
elif REDUCTION_METHOD == "UMAP":
    reducer = umap.UMAP(n_components=N_COMPONENTS, random_state=SEED)
else:
    raise ValueError(f"Invalid reduction method: {REDUCTION_METHOD}. Choose 'TSNE' or 'UMAP'.")

reduced_embeddings = reducer.fit_transform(all_embeddings)

# --- Get Embedding for the New Data Point ---
try:
    new_image = PILImage.open(NEW_IMAGE_PATH).convert("RGB")
    processed_new_image = preprocess(new_image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        new_embedding = model(processed_new_image).cpu().numpy()
except Exception as e:
    print(f"Error processing new image: {e}")
    new_embedding = None

# --- Reduce Dimensionality of the New Embedding ---
if new_embedding is not None:
    reduced_new_embedding = reducer.transform(new_embedding)

    # --- Visualize Embeddings with the New Point ---
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=all_labels, cmap='viridis', s=20)

    # Plot the new point
    new_point_color = plt.cm.viridis(NEW_IMAGE_LABEL_INDEX / len(os.listdir(IMG_DIR))) # Get color based on its label index
    plt.scatter(reduced_new_embedding[:, 0], reduced_new_embedding[:, 1], c=[new_point_color], marker='*', s=100, label=f'New Point (Class {NEW_IMAGE_LABEL_INDEX})')

    plt.title(f'{REDUCTION_METHOD} Visualization of Embeddings with New Point')
    plt.xlabel(f'{REDUCTION_METHOD} Dimension 1')
    plt.ylabel(f'{REDUCTION_METHOD} Dimension 2')

    # Add a legend for the original classes
    class_names = sorted(os.listdir(IMG_DIR))
    legend1 = plt.legend(handles=scatter.legend_elements()[0], labels=class_names, title="Classes")
    plt.gca().add_artist(legend1)

    # Add a legend for the new point
    plt.legend()

    plt.show()

    # For 3D visualization (if N_COMPONENTS = 3):
    if N_COMPONENTS == 3 and new_embedding is not None:
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], reduced_embeddings[:, 2], c=all_labels, cmap='viridis', s=20)
        ax.scatter(reduced_new_embedding[:, 0], reduced_new_embedding[:, 1], reduced_new_embedding[:, 2], c=[new_point_color], marker='*', s=100, label=f'New Point (Class {NEW_IMAGE_LABEL_INDEX})')
        ax.set_xlabel(f'{REDUCTION_METHOD} Dimension 1')
        ax.set_ylabel(f'{REDUCTION_METHOD} Dimension 2')
        ax.set_zlabel(f'{REDUCTION_METHOD} Dimension 3')
        ax.set_title(f'{REDUCTION_METHOD} Visualization of Embeddings with New Point (3D)')

        legend1 = ax.legend(handles=scatter.legend_elements()[0], labels=class_names, title="Classes")
        ax.add_artist(legend1)
        plt.legend()
        plt.show()