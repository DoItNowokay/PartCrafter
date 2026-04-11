# Save this file as: src/datasets/objaverse_part_eval.py

import os
import json
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from src.utils.data_utils import load_surface, load_surfaces
import random

class ObjaversePartEvalDataset(Dataset):
    def __init__(self, configs, mode='test'):
        self.configs = configs
        self.shuffle_parts = configs['dataset']['shuffle_parts']
        
        self.min_num_parts = configs['dataset']['min_num_parts']
        self.max_num_parts = configs['dataset']['max_num_parts']
        
        json_config_path = self.configs['dataset']['config'][0]
        max_number_of_samples = self.configs['dataset'].get('max_num_samples', None)
        
        with open(json_config_path, 'r') as f:
            etadata = json.load(f)
        metadata = etadata
        # Filter by num_parts
        metadata = [item for item in metadata if self.min_num_parts <= item['num_parts'] <= self.max_num_parts]
        # for i in range(len(etadata)):
        #     if etadata[i]['num_parts'] > 1:
        #         if metadata is None:
        #             metadata = []
        #         metadata.append(etadata[i])
        if max_number_of_samples is not None:
            metadata = np.random.choice(
                metadata,
                size=min(max_number_of_samples, len(metadata)),
                replace=False
            ).tolist()


        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        self.data = []
        print("Pre-loading dataset into memory (this may take a moment)...")
        for item_info in tqdm(metadata, desc="Loading data"):
            if not item_info.get("valid", True):
                continue
            
            num_parts = item_info['num_parts']
            
            image_path = item_info['image_path']
            object_image = Image.open(image_path).convert("RGB")

            if 'surface_path' in item_info:
                surface_path = item_info['surface_path']
                surface_data = np.load(surface_path, allow_pickle=True).item()
                # If parts is empty, the object is the only part
                part_surfaces = surface_data['parts'] if len(surface_data['parts']) > 0 else [surface_data['object']]
                if self.shuffle_parts:
                    random.shuffle(part_surfaces)
                part_surfaces = load_surfaces(part_surfaces) # [N, P, 6]
            else:
                part_surfaces = []
                for surface_path in item_info['surface_paths']:
                    surface_data = np.load(surface_path, allow_pickle=True).item()
                    part_surfaces.append(load_surface(surface_data))
                part_surfaces = torch.stack(part_surfaces, dim=0) # [N, P, 6]

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