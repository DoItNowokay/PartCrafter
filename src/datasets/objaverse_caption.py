from src.utils.typing_utils import *

import json
import os
import random

import torch
from torchvision import transforms
import numpy as np
import pandas as pd 
from PIL import Image
from tqdm import tqdm

from src.utils.data_utils import load_surface, load_surfaces

class ObjaverseCaptionDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        configs: DictConfig,
        training: bool = True,
        mode: str = 'val'
        ):
        super().__init__()
        self.configs = configs
        self.training = training

        caption_file = configs['dataset']['caption_file']
        # --- Load Captions ---
        if caption_file is None:
            raise ValueError("Please provide the path to the caption CSV file.")
        if not os.path.exists(caption_file):
            raise FileNotFoundError(f"Caption file not found at: {caption_file}")

        # Create a dictionary mapping from object ID to caption
        caption_df = pd.read_csv(caption_file)
        self.caption_map = pd.Series(caption_df.caption.values, index=caption_df.id).to_dict()
        # --- End of Caption Loading ---

        self.min_num_parts = configs['dataset']['min_num_parts']
        self.max_num_parts = configs['dataset']['max_num_parts']

        if not self.training:
            if mode not in configs:
                raise KeyError(f"Missing key '{mode}' in config file.")
            self.eval_min_num_parts = configs[mode]['min_num_parts']
            self.eval_max_num_parts = configs[mode]['max_num_parts']

        self.max_iou_mean = configs['dataset'].get('max_iou_mean', None)
        self.max_iou_max = configs['dataset'].get('max_iou_max', None)

        self.shuffle_parts = configs['dataset']['shuffle_parts']
        self.training_ratio = configs['dataset']['training_ratio']
        self.balance_object_and_parts = configs['dataset'].get('balance_object_and_parts', False)

        self.rotating_ratio = configs['dataset'].get('rotating_ratio', 0.0)
        self.rotating_degree = configs['dataset'].get('rotating_degree', 10.0)
        self.transform = transforms.Compose([
            transforms.RandomRotation(degrees=(-self.rotating_degree, self.rotating_degree), fill=(255, 255, 255)),
        ])

        if isinstance(configs['dataset']['config'], ListConfig):
            data_configs = []
            for config in configs['dataset']['config']:
                with open(config) as f:
                    local_data_configs = json.load(f)

                if self.balance_object_and_parts:
                    if self.training:
                        local_data_configs = local_data_configs[:int(len(local_data_configs) * self.training_ratio)]
                    else:
                        local_data_configs = local_data_configs[int(len(local_data_configs) * self.training_ratio):]
                        local_data_configs = [c for c in local_data_configs if self.eval_min_num_parts <= c['num_parts'] <= self.eval_max_num_parts]
                data_configs += local_data_configs
        else:
            with open(configs['dataset']['config']) as f:
                data_configs = json.load(f)

        data_configs = [config for config in data_configs if config['valid']]

        # --- MODIFIED SECTION: Filter data to only include items with captions ---
        original_count = len(data_configs)
        data_configs = [
            config for config in data_configs
            if self._get_id_from_config(config) in self.caption_map
        ]
        print(f"Filtered dataset: Kept {len(data_configs)} out of {original_count} items with available captions.")
        # --- END OF MODIFIED SECTION ---

        data_configs = [config for config in data_configs if self.min_num_parts <= config['num_parts'] <= self.max_num_parts]
        if self.max_iou_mean is not None and self.max_iou_max is not None:
            data_configs = [config for config in data_configs if config['iou_mean'] <= self.max_iou_mean]
            data_configs = [config for config in data_configs if config['iou_max'] <= self.max_iou_max]

        if not self.balance_object_and_parts:
            if self.training:
                data_configs = data_configs[:int(len(data_configs) * self.training_ratio)]
            else:
                data_configs = data_configs[int(len(data_configs) * self.training_ratio):]
                data_configs = [config for config in data_configs if self.eval_min_num_parts <= config['num_parts'] <= self.eval_max_num_parts]

        self.data_configs = data_configs
        self.image_size = (512, 512)

    def __len__(self) -> int:
        return len(self.data_configs)

    def _get_id_from_config(self, data_config):
        # Assumes the ID is the filename without extension from the surface_path
        # You might need to adjust this based on your file naming convention
        if 'surface_path' in data_config:
            return os.path.splitext(os.path.basename(data_config['surface_path']))[0]
        # Add other ways to get ID if needed, e.g., from image_path
        return None

    def _get_data_by_config(self, data_config):
        object_id = self._get_id_from_config(data_config)
        caption = self.caption_map.get(object_id, "") # Get caption, default to empty string if not found

        if 'surface_path' in data_config:
            surface_path = data_config['surface_path']
            surface_data = np.load(surface_path, allow_pickle=True).item()
            part_surfaces = surface_data['parts'] if len(surface_data['parts']) > 0 else [surface_data['object']]
            if self.shuffle_parts:
                random.shuffle(part_surfaces)
            part_surfaces = load_surfaces(part_surfaces)
        else:
            part_surfaces = []
            for surface_path in data_config['surface_paths']:
                surface_data = np.load(surface_path, allow_pickle=True).item()
                part_surfaces.append(load_surface(surface_data))
            part_surfaces = torch.stack(part_surfaces, dim=0)

        image_path = data_config['image_path']
        image = Image.open(image_path).resize(self.image_size)
        if self.training and random.random() < self.rotating_ratio:
            image = self.transform(image)
        image = np.array(image)
        image = torch.from_numpy(image).to(torch.uint8)
        images = torch.stack([image] * part_surfaces.shape[0], dim=0)
        
        return {
            "images": images,
            "part_surfaces": part_surfaces,
            "caption": caption, # <-- ADDED caption to the return dictionary
        }

    def __getitem__(self, idx: int):
        data_config = self.data_configs[idx]
        data = self._get_data_by_config(data_config)
        return data

class BatchedObjaverseCaptionDataset(ObjaversePartDataset):
    def __init__(
        self,
        configs: DictConfig,
        batch_size: int,
        is_main_process: bool = False,
        shuffle: bool = True,
        training: bool = True,
    ):
        assert training
        assert batch_size > 1
        super().__init__(configs, training)
        self.batch_size = batch_size
        self.is_main_process = is_main_process
        if batch_size < self.max_num_parts:
            self.data_configs = [config for config in self.data_configs if config['num_parts'] <= batch_size]

        if shuffle:
            random.shuffle(self.data_configs)

        self.object_configs = [config for config in self.data_configs if config['num_parts'] == 1]
        self.parts_configs = [config for config in self.data_configs if config['num_parts'] > 1]

        self.object_ratio = configs['dataset']['object_ratio']
        self.object_configs = self.object_configs[:int(len(self.parts_configs) * self.object_ratio)]

        dropped_data_configs = self.parts_configs + self.object_configs
        if shuffle:
            random.shuffle(dropped_data_configs)

        self.data_configs = self._get_batched_configs(dropped_data_configs, batch_size)

    def _get_batched_configs(self, data_configs, batch_size):
        batched_data_configs = []
        progress_bar = tqdm(
            total=len(data_configs),
            desc="Batching Dataset",
            ncols=125,
            disable=not self.is_main_process,
        )
        current_batch = []
        current_batch_parts = 0
        
        while data_configs:
            config = data_configs.pop(0)
            num_parts = config['num_parts']
            if current_batch_parts + num_parts <= batch_size:
                current_batch.append(config)
                current_batch_parts += num_parts
                progress_bar.update(1)
            else:
                # Pad and add the current batch
                if current_batch:
                    padding_needed = batch_size - len(current_batch)
                    if padding_needed > 0:
                         current_batch.extend([{}] * padding_needed)
                    batched_data_configs.extend(current_batch)
                # Start a new batch
                current_batch = [config]
                current_batch_parts = num_parts
                progress_bar.update(1)

        # Add the last batch if it exists
        if current_batch:
             padding_needed = batch_size - len(current_batch)
             if padding_needed > 0:
                  current_batch.extend([{}] * padding_needed)
             batched_data_configs.extend(current_batch)

        progress_bar.close()
        return batched_data_configs

    def __getitem__(self, idx: int):
        data_config = self.data_configs[idx]
        if not data_config:
            return {}
        return self._get_data_by_config(data_config)

    def collate_fn(self, batch):
        batch = [data for data in batch if data]
        if not batch:
            return None
            
        images = torch.cat([data['images'] for data in batch], dim=0)
        surfaces = torch.cat([data['part_surfaces'] for data in batch], dim=0)
        num_parts = torch.LongTensor([data['part_surfaces'].shape[0] for data in batch])
        
        captions = [data['caption'] for data in batch]

        return {
            "images": images,
            "part_surfaces": surfaces,
            "num_parts": num_parts,
            "captions": captions, 
        }