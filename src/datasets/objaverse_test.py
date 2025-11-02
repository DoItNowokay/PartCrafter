# Save this file as: src/datasets/objaverse_part_eval.py

import os
import json
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class ObjaversePartEvalDataset(Dataset):
    def __init__(self, configs, mode='test'):
        self.configs = configs
        
        json_config_path = self.configs['dataset']['config'][0]
        
        with open(json_config_path, 'r') as f:
            metadata = json.load(f)

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        self.data = []
        print("Pre-loading dataset into memory (this may take a moment)...")
        for item_info in tqdm(metadata, desc="Loading data"):
            num_parts = item_info['num_parts']
            
            image_path = item_info['image_path']
            object_image = Image.open(image_path).convert("RGB")

            part_surfaces = []
            for i in range(num_parts):
                part_pc_filename = f"part_{i:02d}.npy"
                base_surface_dir = os.path.dirname(item_info['surface_path'])
                part_pc_path = os.path.join(base_surface_dir, "part_pointcloud", part_pc_filename)
                
                if os.path.exists(part_pc_path):
                    part_pc = np.load(part_pc_path)
                    part_surfaces.append(torch.from_numpy(part_pc).float())
                else:
                    part_surfaces.append(torch.zeros((1, 3)))

            self.data.append({
                "image": image_path,
                "part_surfaces": part_surfaces,
                "num_parts": num_parts
            })
        print("Dataset pre-loading complete.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        # image_tensor = self.transform(item['image'])
        
        return {
            "image": item['image'],
            "part_surfaces": item['part_surfaces'],
            "num_parts": item['num_parts']
        }

def collate_fn_eval(batch):

    # images = torch.stack([item['image'] for item in batch])
    images = [item['image'] for item in batch]
    part_surfaces = [item['part_surfaces'] for item in batch]
    num_parts = [item['num_parts'] for item in batch]
    
    return {
        "images": images,
        "part_surfaces": part_surfaces,
        "num_parts": num_parts
    }