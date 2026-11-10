import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from torchvision import transforms
from datasets import load_from_disk
from models.model_resnet50 import Model
import json
from tqdm import tqdm

CLASSES = [
    "letter", "memo", "email", "filefolder", "form",
    "handwritten", "invoice", "advertisement", "budget",
    "news article", "presentation", "scientific publication",
    "questionnaire", "resume", "scientific report", "specification"
]

DATASET_PATH = "/mnt/c/Users/aress/OneDrive/Escritorio/tfg_stamps/tfg_stamps/rvl_cdip_full"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")

# Load model
model = Model(device=device)
model.load_state_dict(torch.load(
    os.path.join(ROOT, "models", "rvl-resnet50.model"),
    map_location=device
))
model.eval()
model.to(device)
print("Model loaded")

# Preprocessing — same as standard ResNet
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),  # dataset is grayscale, ResNet needs 3ch
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Load dataset
print("Loading dataset...")
dataset = load_from_disk(DATASET_PATH)
print(f"Total images: {len(dataset)}")

# We only need a manageable subset — 500 per class max
# Filter correctly classified images, max 500 per class
correct_per_class = {i: [] for i in range(16)}
MAX_PER_CLASS = 500

print("Running inference — this will take a while...")
with torch.no_grad():
    for idx in tqdm(range(len(dataset))):
        # Stop early if we have enough for all classes
        if all(len(v) >= MAX_PER_CLASS for v in correct_per_class.values()):
            break

        sample = dataset[idx]
        true_label = sample["label"]

        # Skip if we already have enough for this class
        if len(correct_per_class[true_label]) >= MAX_PER_CLASS:
            continue

        try:
            img = sample["image"].convert("RGB")
            tensor = preprocess(img).unsqueeze(0).to(device)
            _, logits = model(tensor)
            probs = F.softmax(logits, dim=1)
            pred = logits.argmax(dim=1).item()
            confidence = probs[0, pred].item()

            if pred == true_label and confidence == 1.0: # ABANS ESTAVA A 0.95
                correct_per_class[true_label].append({
                    "idx": idx,
                    "label": true_label,
                    "class_name": CLASSES[true_label],
                    "confidence": round(confidence, 4),
                })
        except Exception as e:
            continue

# Summary
total_correct = sum(len(v) for v in correct_per_class.values())
print(f"\nCorrectly classified images found: {total_correct}")
for i, name in enumerate(CLASSES):
    print(f"  {name:30s}: {len(correct_per_class[i])} images")

# Save to JSON
out_path = os.path.join(ROOT, "correct_images_1.0.json")
with open(out_path, "w") as f:
    json.dump(correct_per_class, f, indent=2)
print(f"\nSaved to {out_path}")