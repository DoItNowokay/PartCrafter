from src.utils.typing_utils import *

import json
import os
import random

import accelerate
import torch
from torchvision import transforms
import numpy as np
from PIL import Image
from tqdm import tqdm
import pandas as pd

from src.utils.data_utils import load_surface, load_surfaces

class ObjaverseCaptionDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        configs: DictConfig,
        training: bool = True,
    ):
        super().__init__()
        self.configs = configs
        self.training = training

        self.min_num_parts = configs['dataset']['min_num_parts']
        self.max_num_parts = configs['dataset']['max_num_parts']
        self.val_min_num_parts = configs['val']['min_num_parts']
        self.val_max_num_parts = configs['val']['max_num_parts']

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

        caps3d_csv_path = configs['dataset']['caps3d_csv_path']
        try:
            caps3d_df = pd.read_csv(caps3d_csv_path)
            self.uid_to_caption = dict(zip(caps3d_df["uid"], caps3d_df["text"]))
        except FileNotFoundError:
            print(f"Warning: Caps3D CSV file not found at: {caps3d_csv_path}")
            self.uid_to_caption = {}

        if isinstance(configs['dataset']['config'], ListConfig):
            data_configs = []
            for config in configs['dataset']['config']:
                local_data_configs = json.load(open(config))
                if self.balance_object_and_parts:
                    if self.training:
                        local_data_configs = local_data_configs[:int(len(local_data_configs) * self.training_ratio)]
                    else:
                        local_data_configs = local_data_configs[int(len(local_data_configs) * self.training_ratio):]
                        local_data_configs = [config for config in local_data_configs if self.val_min_num_parts <= config['num_parts'] <= self.val_max_num_parts]
                data_configs += local_data_configs
        else:
            data_configs = json.load(open(configs['dataset']['config']))

        data_configs = [config for config in data_configs if config['valid']]
        data_configs = [config for config in data_configs if self.min_num_parts <= config['num_parts'] <= self.max_num_parts]
        
        # --- MODIFICATION FOR UID EXTRACTION ---
        # Add a 'uid' key to each config by extracting it from the 'file' key
        for config in data_configs:
            config['uid'] = os.path.splitext(config['file'])[0]

        # Filter out objects that do not have a corresponding caption
        data_configs = [config for config in data_configs if config["uid"] in self.uid_to_caption]

        if self.max_iou_mean is not None and self.max_iou_max is not None:
            data_configs = [config for config in data_configs if config['iou_mean'] <= self.max_iou_mean]
            data_configs = [config for config in data_configs if config['iou_max'] <= self.max_iou_max]
        if not self.balance_object_and_parts:
            if self.training:
                data_configs = data_configs[:int(len(data_configs) * self.training_ratio)]
            else:
                data_configs = data_configs[int(len(data_configs) * self.training_ratio):]
                data_configs = [config for config in data_configs if self.val_min_num_parts <= config['num_parts'] <= self.val_max_num_parts]

        self.data_configs = data_configs
        self.image_size = (512, 512)

    def __len__(self) -> int:
        return len(self.data_configs)

    def _get_data_by_config(self, data_config):
        if 'surface_path' in data_config:
            surface_path = data_config['surface_path']
            surface_data = np.load(surface_path, allow_pickle=True).item()
            part_surfaces = surface_data['parts'] if len(surface_data['parts']) > 0 else [surface_data['object']]
            if self.shuffle_parts:
                random.shuffle(part_surfaces)
            part_surfaces = load_surfaces(part_surfaces) # [N, P, 6]
        else:
            part_surfaces = []
            for surface_path in data_config['surface_paths']:
                surface_data = np.load(surface_path, allow_pickle=True).item()
                part_surfaces.append(load_surface(surface_data))
            part_surfaces = torch.stack(part_surfaces, dim=0) # [N, P, 6]

        image_path = data_config['image_path']
        image = Image.open(image_path).resize(self.image_size)
        if random.random() < self.rotating_ratio:
            image = self.transform(image)
        image = np.array(image)
        image = torch.from_numpy(image).to(torch.uint8) # [H, W, 3]
        images = torch.stack([image] * part_surfaces.shape[0], dim=0) # [N, H, W, 3]

        uid = data_config["uid"]
        caption = self.uid_to_caption.get(uid, "")

        return {
            "images": images,
            "part_surfaces": part_surfaces,
            "caption": caption, 
        }

    def __getitem__(self, idx: int):
        data_config = self.data_configs[idx]
        data = self._get_data_by_config(data_config)
        return data

class BatchedObjaverseCaptionDataset(ObjaverseCaptionDataset):
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
        num_data_configs = len(data_configs)
        progress_bar = tqdm(
            range(len(data_configs)),
            desc="Batching Dataset",
            ncols=125,
            disable=not self.is_main_process,
        )
        while len(data_configs) > 0:
            temp_batch = []
            temp_num_parts = 0
            unchosen_configs = []
            while temp_num_parts < batch_size and len(data_configs) > 0:
                config = data_configs.pop() 
                num_parts = config['num_parts']
                if temp_num_parts + num_parts <= batch_size:
                    temp_batch.append(config)
                    temp_num_parts += num_parts
                    progress_bar.update(1)
                else:
                    unchosen_configs.append(config)
            data_configs = data_configs + unchosen_configs 
            if temp_num_parts == batch_size:
                if len(temp_batch) < batch_size:
                    temp_batch += [{}] * (batch_size - len(temp_batch))
                batched_data_configs += temp_batch
        progress_bar.close()
        return batched_data_configs

    def __getitem__(self, idx: int):
        data_config = self.data_configs[idx]
        if len(data_config) == 0:
            return {}
        data = self._get_data_by_config(data_config)
        return data

    def collate_fn(self, batch):
        batch = [data for data in batch if len(data) > 0]
        images = torch.cat([data['images'] for data in batch], dim=0)
        surfaces = torch.cat([data['part_surfaces'] for data in batch], dim=0) 
        num_parts = torch.LongTensor([data['part_surfaces'].shape[0] for data in batch])
        
        captions = [data['caption'] for data in batch]

        assert images.shape[0] == surfaces.shape[0] == num_parts.sum() == self.batch_size
        batch = {
            "images": images,
            "part_surfaces": surfaces,
            "num_parts": num_parts,
            "captions": captions, 
        }
        return batch