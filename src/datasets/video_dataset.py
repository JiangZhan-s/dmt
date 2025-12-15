import torch
from torch.utils.data import Dataset
import os
from PIL import Image
from torchvision import transforms
import glob

class Vimeo90kDataset(Dataset):
    """
    Standard dataset for Video Compression (Vimeo-90k).
    Assumes structure: /path/to/vimeo_septuplet/sequences/00001/0001/im1.png ...
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        # Resize to 256x256 to ensure compatibility with SPyNet (needs multiples of 32/64)
        self.transform = transform or transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])
        # This is a simplified loader. In reality, you'd parse the 'sep_trainlist.txt'
        self.sequences = []
        # Recursive search for directories containing images (simplified)
        # For real usage, parse the official train list text file.
        if os.path.exists(root_dir):
            # Just finding some folders for demo purposes
            self.sequences = glob.glob(os.path.join(root_dir, "sequences", "*", "*"))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq_path = self.sequences[idx]
        # Load frame 1 (reference) and frame 2 (current)
        # Vimeo90k has 7 frames per folder usually.
        img1_path = os.path.join(seq_path, "im1.png")
        img2_path = os.path.join(seq_path, "im2.png")
        
        img1 = Image.open(img1_path).convert('RGB')
        img2 = Image.open(img2_path).convert('RGB')
        
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            
        return img2, img1 # Return (Current, Reference)

class RandomDummyDataset(Dataset):
    """
    Dataset for testing the code without downloading 80GB of data.
    Generates random video frames.
    """
    def __init__(self, length=100, height=256, width=256):
        self.length = length
        self.height = height
        self.width = width

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Generate random tensors [3, H, W]
        x_ref = torch.rand(3, self.height, self.width)
        # Make x_cur slightly different to simulate motion
        x_cur = x_ref + torch.randn(3, self.height, self.width) * 0.1
        x_cur = torch.clamp(x_cur, 0, 1)
        return x_cur, x_ref
