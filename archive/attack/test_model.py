import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import load_from_disk

CLASSES = [
    "letter", "memo", "email", "filefolder", "form",
    "handwritten", "invoice", "advertisement", "budget",
    "news article", "presentation", "scientific publication",
    "questionnaire", "resume", "scientific report", "specification"
]

# Change this path to where your .arrow files are
DATASET_PATH = "/mnt/c/Users/aress/OneDrive/Escritorio/tfg_stamps/tfg_stamps/rvl_cdip_full"
dataset = load_from_disk(DATASET_PATH)
print(dataset)

# Look at one sample
sample = dataset[0]
print(f"Keys: {sample.keys()}")
print(f"Label: {sample['label']} → {CLASSES[sample['label']]}")
print(f"Image type: {type(sample['image'])}")
print(f"Image size: {sample['image'].size}")