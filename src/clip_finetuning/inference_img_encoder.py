import logging.config
from models.supcon_clip import SupConClipModel
from PIL import Image
import logging
import torch
import clip
from torch import nn
from torch.nn import functional as F

LOG_FILENAME = 'inference_log'
CUSTOM_WEIGHT_PATH = "/Users/maixueqiao/Downloads/project/ai-playground/src/clip_finetuning/output_supcon/clip_supcon_model.pth"
IMAGE_PATH = "/Users/maixueqiao/Downloads/project/ai-playground/src/python_web_scraper/NY_covers/0/1-1.jpg"
DEVICE = "cpu"  

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILENAME, mode='w'),
        logging.StreamHandler()
    ]
)

pretrained, _ = clip.load("ViT-B/32", device=DEVICE)
encoder = pretrained.visual
text_encoder = pretrained.encode_text
tokenizer = clip.tokenize

projector = nn.Sequential(
    nn.Linear(encoder.output_dim, 512),
    nn.ReLU(),
    nn.Linear(512, 128) 
)

image_projection = nn.Linear(
    in_features=128,
    out_features=512
).to(DEVICE)

model = SupConClipModel(encoder, projector, device=DEVICE)

try:
    model.load_state_dict(torch.load(CUSTOM_WEIGHT_PATH, map_location=DEVICE))
    logging.info(f"Loaded custom weights from: {CUSTOM_WEIGHT_PATH}")
except FileNotFoundError:
    logging.error(f"Custom weight file not found at: {CUSTOM_WEIGHT_PATH}")
    exit()
except Exception as e:
    logging.error(f"Error loading custom weights: {e}")
    exit()

image = Image.open(IMAGE_PATH).convert("RGB")

preprocess = clip.load("ViT-B/32")[1] # Get the preprocessing function
input_tensor = preprocess(image).unsqueeze(0).to(DEVICE)

model.eval()
with torch.no_grad():
    output_embedding = model(input_tensor)
    image_embeddings = F.normalize(output_embedding, dim=-1)
    projected_image_embeddings = image_projection(image_embeddings)
    projected_image_embeddings = F.normalize(projected_image_embeddings, dim=-1)

labels = ["the new yorker", "the parisianer", "the tokyoiter"]

text_inputs = tokenizer(labels).to(DEVICE)
with torch.no_grad():
    text_embeddings = text_encoder(text_inputs)
    text_embeddings = F.normalize(text_embeddings, p=2, dim=-1)

similarity_scores = torch.matmul(projected_image_embeddings, text_embeddings.T).squeeze(0)

closest_label_index = torch.argmax(similarity_scores).item()
closest_label = labels[closest_label_index]
similarity_score = similarity_scores[closest_label_index].item()

for label, embedding in zip(labels, text_embeddings.cpu().numpy()):
    print(f"{label}: {embedding}")
print(f"\nClosest label: {closest_label} with similarity score: {similarity_score:.4f}")

top_k = 1
top_similarity_scores, top_k_indices = torch.topk(similarity_scores, top_k)
top_k_labels = [labels[i] for i in top_k_indices.cpu().numpy()]
print(f"Top {top_k} closest labels: {top_k_labels} with scores: {top_similarity_scores.cpu().numpy()}")