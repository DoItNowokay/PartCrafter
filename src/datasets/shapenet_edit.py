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


import sys
# Make sure this path is correct for your setup
try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from src.utils.data_utils import load_surface, load_surfaces
    from src.utils.typing_utils import *
except ImportError:
    print("Warning: Could not import from src.utils.data_utils or typing_utils.")
    print("Please ensure the sys.path.append points to the correct root directory.")
    # Define dummy functions if not found, to allow script to run
    def load_surface(x): raise ImportError("Missing data_utils")
    def load_surfaces(x): raise ImportError("Missing data_utils")
    class DictConfig: pass
    class ListConfig: pass

from omegaconf import ListConfig, DictConfig

# --- NEW UTILITY FUNCTION ---
def downsample_pc(pc, num_points):
    """
    Downsamples a point cloud to a specific number of points.
    pc: torch.Tensor of shape [N, C]
    num_points: int
    """
    if pc.shape[0] == num_points:
        return pc
    elif pc.shape[0] > num_points:
        # Randomly sample indices
        indices = torch.randperm(pc.shape[0], device=pc.device)[:num_points]
        return pc[indices]
    else:
        # Randomly repeat points (sample with replacement)
        indices = torch.randint(0, pc.shape[0], (num_points,), device=pc.device)
        return pc[indices]

class ShapeNetLatentEditing(torch.utils.data.Dataset):
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
        
        # --- ADDED ---
        self.num_source_points = configs['dataset'].get('num_source_points', 16384)

        if isinstance(configs['dataset']['config'], ListConfig):
            data_configs = []
            for config_path in configs['dataset']['config']:
                if not os.path.exists(config_path):
                    print(f"Warning: Config file not found {config_path}")
                    continue
                local_data_configs = json.load(open(config_path))
                if self.balance_object_and_parts:
                    if self.training:
                        local_data_configs = local_data_configs[:int(len(local_data_configs) * self.training_ratio)]
                    else:
                        local_data_configs = local_data_configs[int(len(local_data_configs) * self.training_ratio):]
                        local_data_configs = [config for config in local_data_configs if self.val_min_num_parts <= config['num_parts'] <= self.val_max_num_parts]
                data_configs += local_data_configs
        else:
            config_path = configs['dataset']['config']
            if not os.path.exists(config_path):
                 print(f"Error: Config file not found {config_path}")
                 self.data_configs = []
                 return
            data_configs = json.load(open(config_path))

        data_configs = [config for config in data_configs if config.get('valid', False)]
        
        if not self.balance_object_and_parts:
            if self.training:
                data_configs = data_configs[:int(len(data_configs) * self.training_ratio)]
            else:
                data_configs = data_configs[int(len(data_configs) * self.training_ratio):]
                data_configs = [config for config in data_configs if self.val_min_num_parts <= config.get('num_parts', 0) <= self.val_max_num_parts]
        
        self.uid_to_dataconfig = {}
        for config in data_configs:
            if 'file' not in config:
                continue
            uid = os.path.splitext(config['file'])[0]
            config['uid'] = uid 
            self.uid_to_dataconfig[uid] = config
            
        print(f"Loaded {len(self.uid_to_dataconfig)} objects into lookup table.")
        shapetalk_csv_path = configs['dataset']['shapetalk_csv_path']
        try:
            shapetalk_df = pd.read_csv(shapetalk_csv_path)
        except FileNotFoundError:
            print(f"Error: ShapeTalk CSV file not found at: {shapetalk_csv_path}")
            self.data_configs = []
            return
        except Exception as e:
            print(f"Error reading {shapetalk_csv_path}: {e}")
            self.data_configs = []
            return

        new_data_configs = []
        for _, row in tqdm(shapetalk_df.iterrows(), total=len(shapetalk_df), desc="Processing ShapeTalk Pairs"):
            source_uid = row.get('source_model_name')
            target_uid = row.get('target_model_name')
            utterance = row.get('utterance')

            if not all([source_uid, target_uid, utterance is not None]):
                continue
            if source_uid in self.uid_to_dataconfig.keys() and target_uid in self.uid_to_dataconfig.keys():
                source_config = self.uid_to_dataconfig[source_uid]
                target_config = self.uid_to_dataconfig[target_uid]
                # print(f"Found pair: source={source_uid}, target={target_uid}")
                num_parts = target_config.get('num_parts', 0)
                if not (self.min_num_parts <= num_parts <= self.max_num_parts):
                    continue
                
                if not self.training:
                    if not (self.val_min_num_parts <= num_parts <= self.val_max_num_parts):
                        continue
                new_data_configs.append({
                    'utterance': str(utterance),
                    'source_config': source_config,
                    'target_config': target_config,
                    'num_parts': num_parts 
                })

        self.data_configs = new_data_configs
        # print(f"Created {len(self.data_configs)} valid data items from {shapetalk_csv_path}.")
        if len(self.data_configs) == 0:
            print("WARNING: Dataset is empty. Check paths and filters.")
        # else:
            # print(f"kdsnflksdnflk Example data item: {self.data_configs[0]}")

        self.image_size = (512, 512)

    def __len__(self) -> int:
        return len(self.data_configs)

    def _get_data_by_config(self, data_item):
        if not data_item:
            return {}

        source_config = data_item['source_config']
        target_config = data_item['target_config']
        caption = data_item['utterance']

        # --- 1. Load TARGET part surfaces (to get num_parts) ---
        try:
            if 'surface_path' in target_config:
                surface_path = target_config['surface_path']
                if not os.path.exists(surface_path): return {}
                surface_data = np.load(surface_path, allow_pickle=True).item()
                target_part_surfaces_data = surface_data['parts'] if len(surface_data['parts']) > 0 else [surface_data['object']]
                if self.shuffle_parts:
                    random.shuffle(target_part_surfaces_data)
                target_part_surfaces = load_surfaces(target_part_surfaces_data) # [N_target, P_target, 6]
            else:
                target_part_surfaces_list = []
                if 'surface_paths' not in target_config: return {}
                for surface_path in target_config['surface_paths']:
                    if not os.path.exists(surface_path): continue
                    surface_data = np.load(surface_path, allow_pickle=True).item()
                    target_part_surfaces_list.append(load_surface(surface_data))
                if not target_part_surfaces_list: return {}
                target_part_surfaces = torch.stack(target_part_surfaces_list, dim=0) # [N_target, P_target, 6]
        except Exception:
            return {} 
        
        num_parts = target_part_surfaces.shape[0] # This is N_target
        if num_parts == 0:
            return {}

        # --- 2. Load SOURCE part surfaces (concat and downsample) ---
        try:
            if 'surface_path' in source_config:
                surface_path = source_config['surface_path']
                if not os.path.exists(surface_path): return {}
                surface_data = np.load(surface_path, allow_pickle=True).item()
                source_part_surfaces_data = surface_data['parts'] if len(surface_data['parts']) > 0 else [surface_data['object']]
                source_part_surfaces = load_surfaces(source_part_surfaces_data) # [N_source, P_source, 6]
            else:
                source_part_surfaces_list = []
                if 'surface_paths' not in source_config: return {}
                for surface_path in source_config['surface_paths']:
                    if not os.path.exists(surface_path): continue
                    surface_data = np.load(surface_path, allow_pickle=True).item()
                    source_part_surfaces_list.append(load_surface(surface_data))
                if not source_part_surfaces_list: return {}
                source_part_surfaces = torch.stack(source_part_surfaces_list, dim=0) # [N_source, P_source, 6]
            
            # Concat and downsample
            source_pc_global = source_part_surfaces.view(-1, 6) # [N_source * P_source, 6]
            downsampled_source_pc = downsample_pc(source_pc_global, self.num_source_points) # [num_source_points, 6]
            
        except Exception as e:
            print(f"Warning: Error loading source surfaces for {source_config.get('uid')}: {e}")
            return {} 

        # Decide if we rotate this item
        apply_rotation = random.random() < self.rotating_ratio

        # --- 3. Load SOURCE Image ---
        source_image_path = source_config.get('image_path')
        if not source_image_path or not os.path.exists(source_image_path):
            return {}
        try:
            source_image = Image.open(source_image_path).convert("RGB").resize(self.image_size)
            if apply_rotation:
                source_image = self.transform(source_image)
            source_image = np.array(source_image)
            source_image = torch.from_numpy(source_image).to(torch.uint8) # [H, W, 3]
        except Exception as e:
            print(f"Warning: Error loading source image {source_image_path}: {e}")
            return {}
            
        # --- 4. Load TARGET Image ---
        target_image_path = target_config.get('image_path')
        if not target_image_path or not os.path.exists(target_image_path):
            return {}
        try:
            target_image = Image.open(target_image_path).convert("RGB").resize(self.image_size)
            if apply_rotation:
                target_image = self.transform(target_image) # Apply same transform
            target_image = np.array(target_image)
            target_image = torch.from_numpy(target_image).to(torch.uint8) # [H, W, 3]
        except Exception as e:
            print(f"Warning: Error loading target image {target_image_path}: {e}")
            return {}

        # --- 5. Stack tensors ---
        source_images = torch.stack([source_image] * num_parts, dim=0) # [N_target, H, W, 3]
        target_images = torch.stack([target_image] * num_parts, dim=0) # [N_target, H, W, 3]
        captions = [caption] * num_parts
        source_part_surfaces = torch.stack([downsampled_source_pc] * num_parts, dim=0) # [N_target, num_source_points, 6]


        if caption == "":
            print(f"Warning: Empty caption for source={source_config['uid']}, target={target_config['uid']}")

        return {
            "source_images": source_images,
            "target_images": target_images, 
            "source_part_surfaces": source_part_surfaces, # <-- ADDED
            "target_part_surfaces": target_part_surfaces,
            "captions": captions,
        }

    def __getitem__(self, idx: int):
        data_config = self.data_configs[idx]
        data = self._get_data_by_config(data_config)
        return data

class BatchedShapeNetLatentEditing(ShapeNetLatentEditing):
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
        data_configs_list = list(data_configs) # Use a copy to pop from
        
        progress_bar = tqdm(
            total=len(data_configs_list),
            desc="Batching Dataset",
            ncols=125,
            disable=not self.is_main_process,
        )
        
        while len(data_configs_list) > 0:
            temp_batch = []
            temp_num_parts = 0
            unchosen_configs = []
            
            while temp_num_parts < batch_size and len(data_configs_list) > 0:
                config = data_configs_list.pop(0) 
                
                num_parts = config.get('num_parts', 0)
                if num_parts == 0: # Skip items with 0 parts
                    progress_bar.update(1)
                    continue
                
                if temp_num_parts + num_parts <= batch_size:
                    temp_batch.append(config)
                    temp_num_parts += num_parts
                    progress_bar.update(1)
                else:
                    unchosen_configs.append(config)
            
            data_configs_list = unchosen_configs + data_configs_list
            
            # --- MODIFIED: Save last incomplete batch ---
            if temp_num_parts > 0:
                if len(temp_batch) < batch_size:
                    temp_batch += [{}] * (batch_size - len(temp_batch))
                batched_data_configs.extend(temp_batch)

            if temp_num_parts == 0 and len(data_configs_list) > 0:
                # Discard next item if it's too large
                data_configs_list.pop(0)
                progress_bar.update(1)
                
        progress_bar.close()
        return batched_data_configs


    def __getitem__(self, idx: int):
        if idx >= len(self.data_configs):
             raise IndexError("Index out of bounds")
        data_config = self.data_configs[idx]
        if not data_config: # Check for empty dict {}
            return {}
        data = self._get_data_by_config(data_config) 
        return data

    def collate_fn(self, batch):
        batch = [data for data in batch if data] # Filter out empty {}
        if not batch:
            return None 
            
        source_images = torch.cat([data['source_images'] for data in batch], dim=0)
        target_images = torch.cat([data['target_images'] for data in batch], dim=0) 
        source_surfaces = torch.cat([data['source_part_surfaces'] for data in batch], dim=0) # <-- ADDED
        target_surfaces = torch.cat([data['target_part_surfaces'] for data in batch], dim=0)
        num_parts = torch.LongTensor([data['target_part_surfaces'].shape[0] for data in batch])

        captions = [data['captions'] for data in batch]
        captions = sum(captions, []) 
        
        total_parts = num_parts.sum().item()

        # --- MODIFIED: Removed strict batch_size check ---
        assert source_images.shape[0] == total_parts
        assert target_images.shape[0] == total_parts
        assert source_surfaces.shape[0] == total_parts # <-- ADDED
        assert target_surfaces.shape[0] == total_parts
        # --- END MODIFIED ---

        batch = {
            "source_images": source_images,
            "target_images": target_images, 
            "source_part_surfaces": source_surfaces, # <-- ADDED
            "target_part_surfaces": target_surfaces,
            "num_parts": num_parts,
            "captions": captions,
        }
        return batch
    

import plotly.graph_objects as go
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
if __name__ == "__main__":
    
    # !!! UPDATE THIS PATH !!!
    config_path = "/storage/ab_anoushkrit/github/PartCrafter/configs/mp8_nt512_edit.yaml"
    
    if not os.path.exists(config_path):
         print(f"Error: Config file not found at {config_path}")
         exit()
         
    configs = OmegaConf.load(config_path)
    # Add the new key for testing
    if 'dataset' not in configs: configs.dataset = {}
    if 'num_source_points' not in configs.dataset:
        print("Adding default 'num_source_points: 2048' to config for testing")
        configs.dataset.num_source_points = 2048
        
    os.makedirs("plot_outputs", exist_ok=True)

    print("Initializing dataset...")
    dataset = BatchedShapeNetLatentEditing(
        configs, 
        batch_size=8, 
        is_main_process=True, 
        shuffle=False, # Set to False for stable, reproducible testing
        training=False
    )
    val_dataset = ShapeNetLatentEditing(
        configs, 
        is_main_process=True, 
        shuffle=False, 
        training=False
    )
    print(f"Dataset size (number of batched items): {len(dataset)}")
    
    if len(dataset) == 0:
        print("Dataset is empty. Exiting.")
        exit()

    print("Starting dataset iteration using DataLoader...")

    data_loader = DataLoader(
        dataset,
        batch_size=dataset.batch_size, 
        shuffle=False, 
        num_workers=0, 
        collate_fn=dataset.collate_fn 
    )

    items_processed = 0
    for i, data in enumerate(data_loader):
        
        if data is None:
            print(f"Batch {i}: Skipped (collate_fn returned None or load error)")
            continue

        print(f"\n--- Processing Batch {i} ---")
        
        for k, v in data.items():
            if isinstance(v, torch.Tensor):
                print(f"  {k}: {v.shape}")
            else:
                if k == 'captions' and isinstance(v, list):
                    print(f"  {k}: (list of {len(v)})")
                    print(f"    Sample: {v[0]}" if v else "")
                else:
                    print(f"  {k}: {v}")

        # --- SAVE INPUT IMAGE (source) ---
        if 'source_images' in data:
            try:
                image_tensor = data['source_images'][0] 
                image_np = image_tensor.cpu().numpy()
                pil_image = Image.fromarray(image_np)
                img_filename = os.path.join("plot_outputs", f"source_image_{i}.png")
                pil_image.save(img_filename)
                print(f"  Saved source image: {img_filename}")
            except Exception as e:
                print(f"  Error saving source image: {e}")

        # --- SAVE TARGET IMAGE ---
        if 'target_images' in data:
            try:
                image_tensor = data['target_images'][0] 
                image_np = image_tensor.cpu().numpy()
                pil_image = Image.fromarray(image_np)
                img_filename = os.path.join("plot_outputs", f"target_image_{i}.png") 
                pil_image.save(img_filename)
                print(f"  Saved target image: {img_filename}")
            except Exception as e:
                print(f"  Error saving target image: {e}")

        # --- PLOTTING LOGIC (SOURCE point cloud) ---
        if 'source_part_surfaces' in data:
            print(f"  Plotting source_part_surfaces for batch {i}...")
            # All source PCs in the batch are identical, so just plot the first one
            surfaces_tensor = data['source_part_surfaces'][0] # Shape [num_source_points, 6]
            surfaces_np = surfaces_tensor.cpu().numpy()
            
            fig = go.Figure()
            part_points = surfaces_np[:, :3] # [num_source_points, 3]
            fig.add_trace(go.Scatter3d(
                x=part_points[:, 0], y=part_points[:, 1], z=part_points[:, 2],
                mode='markers',
                marker=dict(size=2, opacity=0.8),
                name='Source Point Cloud'
            ))
            
            plot_title = f"Batch {i} - Source Shape"
            fig.update_layout(
                title=plot_title,
                scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z', aspectmode='data')
            )
            html_filename = os.path.join("plot_outputs", f"source_part_surfaces_{i}.html")
            fig.write_html(html_filename)
            print(f"  Saved HTML plot: {html_filename}")

        # --- PLOTTING LOGIC (TARGET part surfaces) ---
        # --- FIXED BUG: key was 'part_surfaces' ---
        if 'target_part_surfaces' in data: 
            print(f"  Plotting target_part_surfaces for batch {i}...")
            surfaces_tensor = data['target_part_surfaces']
            surfaces_np = surfaces_tensor.cpu().numpy()
            n_parts, n_points, n_dims = surfaces_np.shape

            fig = go.Figure()
            for part_idx in range(n_parts):
                part_points = surfaces_np[part_idx, :, :3]
                fig.add_trace(go.Scatter3d(
                    x=part_points[:, 0], y=part_points[:, 1], z=part_points[:, 2],
                    mode='markers',
                    marker=dict(size=2, opacity=0.8),
                    name=f'Part {part_idx}'
                ))
            
            plot_title = f"Batch {i} - {n_parts} Parts (Target Shape)"
            fig.update_layout(
                title=plot_title,
                scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z', aspectmode='data')
            )
            html_filename = os.path.join("plot_outputs", f"target_part_surfaces_{i}.html")
            fig.write_html(html_filename)
            print(f"  Saved HTML plot: {html_filename}")

        items_processed += 1
        if items_processed >= 3: 
            print("\nProcessed 3 batches. Exiting loop.")
            break
            
    print("Script finished.")
