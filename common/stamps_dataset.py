from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class StampsDataset(Dataset):
    def __init__(self, root_dir: str, image_size: int = 64):
        self.paths = list(Path(root_dir).glob("**/*.png")) + \
                     list(Path(root_dir).glob("**/*.jpg")) + \
                     list(Path(root_dir).glob("**/*.jpeg"))

        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),           # [0, 1] float
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("L")   # force grayscale
        return self.transform(img), 0                    # dummy label


def get_dataloader(root_dir: str, batch_size: int, image_size: int = 64,
                   num_workers: int = 4, shuffle: bool = True) -> DataLoader:
    dataset = StampsDataset(root_dir, image_size)
    return DataLoader(dataset, batch_size=batch_size,
                      shuffle=shuffle, num_workers=num_workers,
                      pin_memory=True)