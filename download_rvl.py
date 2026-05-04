from datasets import load_dataset

dataset = load_dataset("rvl_cdip", split="train")
dataset.save_to_disk("rvl_cdip_full")
