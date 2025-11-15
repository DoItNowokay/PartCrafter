# import os
# import json
# import torch
# import random
# import numpy as np
# import pandas as pd
# from torch.utils.data import Dataset
# from omegaconf import ListConfig

# import sys
# sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
# from src.utils.data_utils import load_surfaces, load_surface

# class ShapeNetEditingDataset(Dataset):
#     """
#     ShapeTalkDataset-style loader for ShapeNet editing.
#     Uses:
#         - JSON configs (with surface_path, image_path)
#         - CSV describing (source_id, target_id, utterance)
#     """

#     def __init__(self, configs, training=True):
#         super().__init__()
#         self.configs = configs
#         self.training = training

#         csv_path = configs["dataset"]["shapetalk_csv_path"]
#         json_list = configs["dataset"]["config"]        # list of JSON files
#         self.min_num_parts = configs["dataset"].get("min_num_parts", 1)
#         self.max_num_parts = configs["dataset"].get("max_num_parts", 32)

#         # ---------------------------------------------------
#         # 1. LOAD ALL SHAPENET JSON CONFIGS (fast)
#         # ---------------------------------------------------
#         all_json = []
#         # Case 1: ListConfig (multiple JSON files)
#         if isinstance(json_list, ListConfig) or isinstance(json_list, list):
#             for cfg_file in json_list:
#                 with open(cfg_file, "r") as f:
#                     all_json += json.load(f)

#         # Case 2: single JSON string
#         elif isinstance(json_list, str):
#             with open(json_list, "r") as f:
#                 all_json = json.load(f)

#         else:
#             raise TypeError(f"Unsupported type for dataset.config: {type(json_list)}")

#         # keep only valid items
#         all_json = [c for c in all_json if c["valid"]]

#         # Add UID = filename without extension
#         for c in all_json:
#             c["uid"] = os.path.splitext(c["file"])[0]

#         # Map uid → config
#         self.uid_to_config = {c["uid"]: c for c in all_json}

#         # ---------------------------------------------------
#         # 2. LOAD CSV (editing pairs)
#         # ---------------------------------------------------
#         df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
#         df.columns = [c.strip() for c in df.columns]

#         needed = ["source_model_name", "target_model_name", "utterance"]
#         for n in needed:
#             if n not in df.columns:
#                 raise RuntimeError(f"CSV missing {n}")

#         # normalize UIDs like "chair_f5be22e..." -> "f5be22e..."
#         df["source_model_name"] = df["source_model_name"].apply(lambda x: x.split("_")[-1])
#         df["target_model_name"] = df["target_model_name"].apply(lambda x: x.split("_")[-1])

#         # ---------------------------------------------------
#         # 3. Build editing configs using JSONs only
#         # ---------------------------------------------------
#         data_configs = []
#         for _, row in df.iterrows():
#             src_uid = row["source_model_name"]
#             tgt_uid = row["target_model_name"]

#             if src_uid not in self.uid_to_config:
#                 continue
#             if tgt_uid not in self.uid_to_config:
#                 continue

#             src_cfg = self.uid_to_config[src_uid]
#             tgt_cfg = self.uid_to_config[tgt_uid]

#             # filter by part count (like objaverse)
#             if not (self.min_num_parts <= src_cfg["num_parts"] <= self.max_num_parts):
#                 continue
#             if not (self.min_num_parts <= tgt_cfg["num_parts"] <= self.max_num_parts):
#                 continue

#             data_configs.append({
#                 "src": src_cfg,
#                 "tgt": tgt_cfg,
#                 "caption": row["utterance"],
#                 "num_parts": tgt_cfg["num_parts"],
#             })

#         self.data_configs = data_configs
#         print(f"[ShapeNetEditingDataset] Loaded {len(self.data_configs)} editing pairs.")

#     def __len__(self):
#         return len(self.data_configs)

#     # ---------------------------------------------------
#     # Load surfaces from surface_path
#     # ---------------------------------------------------
#     def _load_parts(self, surface_path):
#         d = np.load(surface_path, allow_pickle=True).item()
#         if "parts" in d and len(d["parts"]) > 0:
#             return load_surfaces(d["parts"])
#         return load_surface(d["object"]).unsqueeze(0)
    
#     # ---- Add to ShapeNetEditingDataset class ----
#     def _get_data_by_config(self, cfg):
#         """
#         Helper: load a single sample given the `cfg` dict produced during init.
#         Returns the same dict shape as __getitem__ normally returns.
#         """
#         src_cfg = cfg["src"]
#         tgt_cfg = cfg["tgt"]

#         # load surfaces (uses your existing _load_parts)
#         src = self._load_parts(src_cfg["surface_path"])
#         tgt = self._load_parts(tgt_cfg["surface_path"])

#         caption = cfg.get("caption", "")
#         captions = [caption] * (tgt.shape[0])

#         # uids: try uid key, fallback to filename without extension
#         # src_uid = src_cfg.get("uid") or os.path.splitext(src_cfg.get("file", ""))[0]
#         # tgt_uid = tgt_cfg.get("uid") or os.path.splitext(tgt_cfg.get("file", ""))[0]
        
#         def make_white_image(H=512, W=512):
#             arr = np.ones((H, W, 3), dtype=np.uint8) * 255   # white RGB
#             return torch.from_numpy(arr)

#         img = make_white_image()
#         # repeat image for each part (src + tgt)
#         images = torch.stack([img] * (tgt.shape[0]), dim=0)  # [T, H, W, 3]

#         return {
#             "images": images, #plain white image of 512x512 ,
#             "source_part_surfaces": src,
#             "part_surfaces": tgt,
#             "num_parts": tgt.shape[0],
#             "captions": captions
#         }


#     def __getitem__(self, idx):
#         cfg = self.data_configs[idx]

#         src_cfg = cfg["src"]
#         tgt_cfg = cfg["tgt"]

#         src = self._load_parts(src_cfg["surface_path"])
#         tgt = self._load_parts(tgt_cfg["surface_path"])

#         caption = cfg["caption"]
#         captions = [caption] * (tgt.shape[0])
#         def make_white_image(H=512, W=512):
#             arr = np.ones((H, W, 3), dtype=np.uint8) * 255   # white RGB
#             return torch.from_numpy(arr)

#         img = make_white_image()
#         # repeat image for each part (src + tgt)
#         images = torch.stack([img] * (tgt.shape[0]), dim=0)  # [S+T, H, W, 3]

#         return {
#             "images": images,
#             "source_part_surfaces": src,
#             "part_surfaces": tgt,
#             "num_parts": tgt.shape[0],
            
#             "captions": captions
#         }

# class BatchedShapeNetEditingDataset(ShapeNetEditingDataset):
#     def __init__(self, configs, batch_size, shuffle=True, training=True, is_main_process=False):
#         super().__init__(configs, training)
#         self.batch_size = batch_size
#         self.is_main_process = is_main_process

#         if shuffle:
#             random.shuffle(self.data_configs)

#         # keep only configs with ≤ batch_size parts
#         self.data_configs = [
#             c for c in self.data_configs if c["num_parts"] <= batch_size
#         ]

#         self.data_configs = self._batch_configs(self.data_configs, batch_size)

#     def _batch_configs(self, data, batch_size):
#         out = []
#         data = data.copy()

#         from tqdm import tqdm
#         pbar = tqdm(total=len(data), disable=not self.is_main_process)

#         while len(data) > 0:
#             temp, nparts, unchosen = [], 0, []

#             while len(data) > 0 and nparts < batch_size:
#                 c = data.pop()
#                 if nparts + c["num_parts"] <= batch_size:
#                     temp.append(c)
#                     nparts += c["num_parts"]
#                     pbar.update(1)
#                 else:
#                     unchosen.append(c)

#             data += unchosen

#             if nparts == batch_size:
#                 out.append(temp)

#         pbar.close()
#         return out

#     # ---- Replace BatchedShapeNetEditingDataset.__getitem__ with this ----
#     def __getitem__(self, idx):
#         """
#         idx indexes a *batch* (a list of per-sample cfgs).
#         Return a list of per-sample dicts (each produced by _get_data_by_config).
#         """
#         batch_cfgs = self.data_configs[idx]  # this is a list of cfg dicts
#         out = []
#         for cfg in batch_cfgs:
#             if cfg == {}:
#                 continue
#             out.append(self._get_data_by_config(cfg))
#         return out

#     # ---- Replace / use this collate_fn in BatchedShapeNetEditingDataset ----
#     def collate_fn(self, batch):
#         """
#         DataLoader will produce: batch = [ list_of_samples ] when DataLoader.batch_size=1.
#         We unwrap and then concat surfaces across the list.
#         """
#         # if DataLoader used batch_size>1 you'd need an outer loop; our main/test uses batch_size=1
#         if isinstance(batch, list) and len(batch) == 1 and isinstance(batch[0], list):
#             samples = batch[0]
#         else:
#             # fallback: flatten
#             samples = []
#             for b in batch:
#                 if isinstance(b, list):
#                     samples += b
#                 elif isinstance(b, dict):
#                     samples.append(b)

#         if len(samples) == 0:
#             return {}

#         # concat surfaces
#         source_part_surfaces = torch.cat([s["source_part_surfaces"] for s in samples], dim=0)
#         part_surfaces = torch.cat([s["part_surfaces"] for s in samples], dim=0)

#         captions = sum([s["captions"] for s in samples], [])
        
#         images = torch.cat([s["images"] for s in samples], dim=0)
        

#         return {
#             "images": images,
#             "source_part_surfaces": source_part_surfaces,
#             "part_surfaces": part_surfaces,
#             "num_parts": torch.LongTensor(
#                 [s["part_surfaces"].shape[0] for s in samples]
#             ),
#             "captions": captions
#         }


import os
import json
import torch
import random
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from omegaconf import ListConfig

# Assuming these utils are in your project
# You might need to adjust the import path
from src.utils.data_utils import load_surfaces, load_surface 

class ShapeNetEditingDataset(Dataset):
    """
    A standard PyTorch Dataset for ShapeNet editing.
    - Abandons part-packing.
    - __getitem__ returns one (source, target) object pair.
    - A custom collate_fn will be used to pad and batch.
    """

    def __init__(self, configs, training=True):
        super().__init__()
        self.configs = configs
        self.training = training

        csv_path = configs["dataset"]["shapetalk_csv_path"]
        json_list = configs["dataset"]["config"]
        
        # This is the GLOBAL max parts your model can handle.
        # We will pad/truncate everything to this size.
        self.max_num_parts = configs["dataset"].get("max_num_parts", 32)

        # ---------------------------------------------------
        # 1. LOAD ALL SHAPENET JSON CONFIGS
        # ---------------------------------------------------
        all_json = []
        if isinstance(json_list, (ListConfig, list)):
            for cfg_file in json_list:
                with open(cfg_file, "r") as f:
                    all_json += json.load(f)
        elif isinstance(json_list, str):
            with open(json_list, "r") as f:
                all_json = json.load(f)
        else:
            raise TypeError(f"Unsupported type for dataset.config: {type(json_list)}")

        all_json = [c for c in all_json if c["valid"]]
        for c in all_json:
            c["uid"] = os.path.splitext(c["file"])[0]

        self.uid_to_config = {c["uid"]: c for c in all_json}

        # ---------------------------------------------------
        # 2. LOAD CSV (editing pairs)
        # ---------------------------------------------------
        df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
        df.columns = [c.strip() for c in df.columns]
        
        df["source_model_name"] = df["source_model_name"].apply(lambda x: x.split("_")[-1])
        df["target_model_name"] = df["target_model_name"].apply(lambda x: x.split("_")[-1])

        # ---------------------------------------------------
        # 3. Build editing configs
        # ---------------------------------------------------
        data_configs = []
        for _, row in df.iterrows():
            src_uid = row["source_model_name"]
            tgt_uid = row["target_model_name"]

            if src_uid not in self.uid_to_config or tgt_uid not in self.uid_to_config:
                continue

            src_cfg = self.uid_to_config[src_uid]
            tgt_cfg = self.uid_to_config[tgt_uid]

            data_configs.append({
                "src": src_cfg,
                "tgt": tgt_cfg,
                "caption": row["utterance"],
            })

        self.data_configs = data_configs
        print(f"[ShapeNetEditingDataset] Loaded {len(self.data_configs)} editing pairs.")

    def __len__(self):
        return len(self.data_configs)

    def _load_parts(self, surface_path):
        d = np.load(surface_path, allow_pickle=True).item()
        if "parts" in d and len(d["parts"]) > 0:
            return load_surfaces(d["parts"])
        return load_surface(d["object"]).unsqueeze(0)
    
    def _make_white_image(self, H=512, W=512):
        arr = np.ones((H, W, 3), dtype=np.uint8) * 255
        return torch.from_numpy(arr)

    def __getitem__(self, idx):
        cfg = self.data_configs[idx]
        src_cfg = cfg["src"]
        tgt_cfg = cfg["tgt"]

        src = self._load_parts(src_cfg["surface_path"])
        tgt = self._load_parts(tgt_cfg["surface_path"])
        caption = cfg["caption"]

        # --- FIX ---
        # Get the actual number of parts BEFORE padding
        num_source_parts = src.shape[0]
        num_target_parts = tgt.shape[0]
        # --- END FIX ---

        captions = [caption] * num_target_parts # This is wrong, should be 1 caption per object
        img = self._make_white_image()
        images = torch.stack([img] * num_target_parts, dim=0) # This is also wrong

        # --- RECOMMENDED FIX for __getitem__ ---
        return {
            "image": img, # Return ONE image, shape [H, W, 3]
            "source_part_surfaces": src, # [Num_Source, N_Points, 6]
            "part_surfaces": tgt,        # [Num_Target, N_Points, 6]
            "caption": caption,          # Return ONE caption string
            "num_source_parts": num_source_parts, # Return part counts
            "num_target_parts": num_target_parts
        }


# Add this class to your shapenet_editing_dataset.py file
# It REPLACES BatchedShapeNetEditingDataset

class EditingCollator:
    """
    This is the NEW collate_fn.
    It takes a list of samples from ShapeNetEditingDataset and
    pads them into a fixed-size "object-level" batch.
    """
    def __init__(self, max_num_parts):
        self.max_parts = max_num_parts
        print(f"[EditingCollator] Initialized. Padding all objects to {self.max_parts} parts.")

    def __call__(self, batch_list):
        batch_images = []
        batch_source_surfaces = []
        batch_target_surfaces = []
        batch_captions = []
        batch_source_mask = []
        batch_target_mask = []
        
        # --- THIS IS THE FIX ---
        batch_num_source_parts = []
        batch_num_target_parts = []
        # --- END FIX ---

        # Get point/channel dims from the first sample
        N_POINTS = batch_list[0]["source_part_surfaces"].shape[1]
        N_CHANNELS = batch_list[0]["source_part_surfaces"].shape[2]

        for sample in batch_list:
            # 1. Image and Caption
            batch_images.append(sample["image"])
            batch_captions.append(sample["caption"])
            
            batch_num_source_parts.append(sample["num_source_parts"])
            batch_num_target_parts.append(sample["num_target_parts"])
            # --- 2. Source Surfaces (Padding) ---
            src_surfaces = sample["source_part_surfaces"]
            num_source_parts = src_surfaces.shape[0]

            if num_source_parts > self.max_parts:
                padded_source = src_surfaces[:self.max_parts]
                source_mask = torch.ones(self.max_parts, dtype=torch.bool)
            else:
                source_padding_size = self.max_parts - num_source_parts
                source_pad = torch.zeros((source_padding_size, N_POINTS, N_CHANNELS), dtype=src_surfaces.dtype)
                padded_source = torch.cat([src_surfaces, source_pad], dim=0)
                source_mask = torch.cat([
                    torch.ones(num_source_parts, dtype=torch.bool),
                    torch.zeros(source_padding_size, dtype=torch.bool)
                ], dim=0)
            
            batch_source_surfaces.append(padded_source)
            batch_source_mask.append(source_mask)

            # --- 3. Target Surfaces (Padding) ---
            tgt_surfaces = sample["part_surfaces"]
            num_target_parts = tgt_surfaces.shape[0]

            if num_target_parts > self.max_parts:
                padded_target = tgt_surfaces[:self.max_parts]
                target_mask = torch.ones(self.max_parts, dtype=torch.bool)
            else:
                target_padding_size = self.max_parts - num_target_parts
                target_pad = torch.zeros((target_padding_size, N_POINTS, N_CHANNELS), dtype=tgt_surfaces.dtype)
                padded_target = torch.cat([tgt_surfaces, target_pad], dim=0)
                target_mask = torch.cat([
                    torch.ones(num_target_parts, dtype=torch.bool),
                    torch.zeros(target_padding_size, dtype=torch.bool)
                ], dim=0)

            batch_target_surfaces.append(padded_target)
            batch_target_mask.append(target_mask)

        # 4. Stack everything into a final "object-level" batch
        return {
            "images": torch.stack(batch_images, dim=0),
            "source_part_surfaces": torch.stack(batch_source_surfaces, dim=0),
            "part_surfaces": torch.stack(batch_target_surfaces, dim=0),
            "captions": batch_captions, # List of strings
            "source_mask": torch.stack(batch_source_mask, dim=0),
            "target_mask": torch.stack(batch_target_mask, dim=0),
            "num_parts": torch.tensor(batch_num_target_parts, dtype=torch.long)
        }