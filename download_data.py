import os
import objaverse
import multiprocessing
import random
import shutil
random.seed(42)
import csv
import requests
from tqdm import tqdm
import tarfile
import argparse
import pandas as pd

from huggingface_hub import snapshot_download

target_classes = {
        'air_conditioner', 'airplane', 'alarm_clock', 'ambulance', 'ant', 'apple', 'armchair', 'axe', 'backpack', 'banana',
        'bandage', 'barn', 'baseball_bat', 'basketball', 'bat_(animal)', 'bathtub', 'bear', 'bed', 'bee', 'bell', 'belt',
        'bench', 'bicycle', 'binoculars', 'bird', 'boat', 'book', 'bookshelf', 'boot', 'bottle', 'bowl', 'bridge', 'broom',
        'bus', 'butterfly', 'cabinet', 'cake', 'calculator', 'camera', 'candle', 'car', 'carrot', 'cat', 'chair',
        'chessboard', 'clock', 'coat', 'coconut', 'coffee_maker', 'cookie', 'cow', 'crab', 'crane_(machine)', 'crocodile',
        'crown', 'cup', 'deer', 'desk', 'dog', 'dolphin', 'door', 'doughnut', 'dress', 'drum', 'duck', 'eagle', 'elephant',
        'envelope', 'eyeglasses', 'fan', 'fire_extinguisher', 'fire_truck', 'fish', 'flag', 'flower', 'fork', 'fox',
        'frying_pan', 'giraffe', 'glove', 'goat', 'grapes', 'guitar', 'hamburger', 'hammer', 'handbag', 'hat', 'headphones',
        'helicopter', 'horse', 'hot_dog', 'house', 'iron_(for_clothing)', 'jacket', 'key', 'keyboard', 'kite', 'knife',
        'ladder', 'lamp', 'laptop', 'leaf', 'lemon', 'lion', 'lipstick', 'lizard', 'microphone', 'microwave', 'monkey',
        'motorcycle', 'mountain', 'mouse_(computer)', 'mushroom', 'orange_(fruit)', 'oven', 'owl', 'pants', 'parrot', 'pen',
        'pencil', 'penguin', 'piano', 'pig', 'pillow', 'pineapple', 'pizza', 'plate', 'police_car', 'potato', 'pumpkin',
        'rabbit', 'refrigerator', 'remote_control', 'rhinoceros', 'rocket', 'sailboat', 'sandwich', 'scissors', 'scorpion',
        'screwdriver', 'shark', 'sheep', 'ship', 'shirt', 'shoe', 'skateboard', 'snake', 'sofa', 'spider', 'spoon'
    }

def download_abo(download_path="./", remove_tar_after=False):
    abo_dir = os.path.join(download_path, 'ABO')
    os.makedirs(abo_dir, exist_ok=True)
    
    url = "https://amazon-berkeley-objects.s3.amazonaws.com/archives/abo-3dmodels.tar"
    tar_path = os.path.join(abo_dir, "abo-3dmodels.tar")
    
    # 2. Download the file
    print(f"Downloading ABO dataset to {tar_path}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status() # Check for download errors
        
        # Get total file size for progress bar
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024 # 1KB
        
        with open(tar_path, 'wb') as file, tqdm(
            desc=tar_path,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(block_size):
                size = file.write(data)
                bar.update(size)
                
        print("Download complete.")

    except Exception as e:
        print(f"Error during download: {e}")
        return

    # 3. Extract the tar file
    print(f"Extracting {tar_path}...")
    try:
        with tarfile.open(tar_path, "r:") as tar:
            # The archive might already contain a top-level folder (e.g., '3dmodels').
            # We extract it directly into the 'ABO' directory.
            tar.extractall(path=abo_dir)
            
        print(f"Extraction complete. Files are located in {abo_dir}")
        
        if remove_tar_after:
            os.remove(tar_path)
            print("Cleaned up tar archive.")
            
    except Exception as e:
        print(f"Error during extraction: {e}")

def download_cc3m(tsv_path, split_name, percentage=100, output_dir="/s3/DATA/CC3M"):
    split_dir = os.path.join(output_dir, split_name)
    os.makedirs(split_dir, exist_ok=True)

    with open(tsv_path, 'r', encoding='utf-8') as f:
        all_rows = list(csv.reader(f, delimiter='\t'))

    total = len(all_rows)
    limit = int((percentage / 100) * total)

    for idx, row in enumerate(tqdm(all_rows[:limit], desc=f"Downloading {split_name}")):
        if len(row) < 2:
            continue 

        _, url = row[0], row[1]

        # ext = os.path.splitext(url.split("/")[-1])[-1].split("?")[0]
        # if ext.lower() not in [".jpg", ".jpeg", ".png"]:
        #     ext = ".jpg"  
        ext = ".jpg"

        img_filename = f"{split_name}_{idx}{ext}"
        img_path = os.path.join(split_dir, img_filename)

        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                with open(img_path, 'wb') as f:
                    f.write(response.content)

            else:
                # print(f"Failed to download {url}")
                pass
        except Exception as e:
            # print(f"Error downloading {url}: {e}")
            pass

def download_objaverse_v1(download_path="./", num_processes=1, num_objects=50000):
    # Update the internal path variables
    objaverse.BASE_PATH = os.path.join(download_path, 'objaverse')
    objaverse._VERSIONED_PATH = os.path.join(objaverse.BASE_PATH, 'hf-objaverse-v1')

    uids = objaverse.load_uids()
    random_object_uids = random.sample(uids, num_objects)

    objects = objaverse.load_objects(
        uids=random_object_uids,
        download_processes=num_processes
    )
    
def download_objaverse_LGM(csv_path, download_path="./", num_processes=1):

    objaverse.BASE_PATH = os.path.join(download_path, 'objaverse')
    objaverse._VERSIONED_PATH = os.path.join(objaverse.BASE_PATH, 'hf-objaverse-LGM')
    
    print(f"Downloading to: {objaverse.BASE_PATH}")

    try:
        df = pd.read_csv(csv_path, header=None)
        uids = df[1].tolist() 
        print(f"Found {len(uids)} objects in {csv_path}")
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    objects = objaverse.load_objects(
        uids=uids,
        download_processes=num_processes
    )

def download_shapenet(download_path="./"):
    snapshot_download(
        repo_id="ShapeNet/shapenetcore-glb",
        repo_type="dataset",
        local_dir=os.path.join(download_path, "ShapeNetGLB"),
        cache_dir=os.path.join(download_path, "ShapeNetGLB/hf_cache"),  
        local_dir_use_symlinks=False
    )

def download_objaverse_catg(target_classes, download_path="./", num_processes=1):
    objaverse.BASE_PATH = os.path.join(download_path, "objaverse/categorized_objaverse")
    objaverse._VERSIONED_PATH = os.path.join(objaverse.BASE_PATH, 'hf-objaverse-v1')

    print("Loading LVIS annotations...")
    lvis_annotations = objaverse.load_lvis_annotations()
    print("LVIS annotations loaded.")

    uids_to_download = []
    uid_to_class = {}
    if len(target_classes) != 0:
        for category in target_classes:
            if category in lvis_annotations:
                uids = lvis_annotations[category]
                uids_to_download.extend(uids)
                for uid in uids:
                    uid_to_class[uid] = category
    else:
        for category, uids in lvis_annotations.items():
            uids_to_download.extend(uids)
            for uid in uids:
                uid_to_class[uid] = category

    uids_to_download = list(set(uids_to_download))

    print(f"\nTotal unique objects to download: {len(uids_to_download)}")

    print(f"Using {num_processes} download processes.")
    print("Starting download...")
    objaverse.load_objects(uids=uids_to_download, download_processes=num_processes)
    print("Download complete.")

    print("\nCategorizing objects...")
    downloaded_glbs_dir = os.path.join(objaverse._VERSIONED_PATH, 'glbs')
    categorized_output_dir = objaverse._VERSIONED_PATH

    os.makedirs(categorized_output_dir, exist_ok=True)

    # Iterate through the downloaded object files and move them
    for root, _, files in os.walk(downloaded_glbs_dir):
        for filename in files:
            if filename.endswith('.glb'):
                uid = filename.replace('.glb', '')
                
                if uid in uid_to_class:
                    category = uid_to_class[uid]
                    
                    # Create the category-specific folder if it doesn't exist
                    category_path = os.path.join(categorized_output_dir, category)
                    os.makedirs(category_path, exist_ok=True)
                    
                    # Move the file
                    src_path = os.path.join(root, filename)
                    dest_path = os.path.join(category_path, filename)
                    
                    try:
                        shutil.move(src_path, dest_path)
                        # print(f"Moved {filename} to '{category}' folder.")
                    except FileNotFoundError:
                        print(f"File not found: {src_path}. Skipping.")

    print("Categorization complete.")
    print(f"Objects are now categorized in: {categorized_output_dir}")

# download_cc3m("data/CC3M/train.tsv", "train", percentage=10) 
# download_cc3m("data/CC3M/val.tsv", "val") 
# if __name__ == "__main__":
#     # download_objaverse_catg(target_classes={}, download_path="/scratch/anoushkrit.scee.iitmandi/DATA", num_processes=20)
#     download_objaverse_v1(download_path="/scratch/ab_anoushkrit/DATA", num_objects=50000)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download various 3D datasets.")
    
    # Core arguments
    parser.add_argument(
        "--dataset", 
        type=str, 
        required=True, 
        choices=["abo", "cc3m", "objaverse_v1", "objaverse_lgm", "shapenet", "objaverse_catg"],
        help="The dataset to download."
    )
    parser.add_argument(
        "--output_path", 
        type=str, 
        required=True, 
        help="Directory path where the dataset will be downloaded."
    )
    
    # Optional arguments shared by multiple functions
    parser.add_argument(
        "--num_processes", 
        type=int, 
        default=multiprocessing.cpu_count(),
        help="Number of processes for parallel downloads."
    )
    
    # Objaverse V1 specific
    parser.add_argument(
        "--num_objects", 
        type=int, 
        default=50000, 
        help="Number of random objects to download (for objaverse_v1)."
    )
    
    # LGM / CC3M specific
    parser.add_argument(
        "--csv_path", 
        type=str, 
        help="Path to CSV file (required for objaverse_lgm)."
    )
    parser.add_argument(
        "--tsv_path", 
        type=str, 
        help="Path to TSV file (required for cc3m)."
    )
    parser.add_argument(
        "--split_name", 
        type=str, 
        default="train", 
        help="Split name for CC3M (e.g., train, val)."
    )
    parser.add_argument(
        "--percentage", 
        type=float, 
        default=100.0, 
        help="Percentage of CC3M dataset to download."
    )

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path, exist_ok=True)

    print(f"--- Starting Download for {args.dataset} ---")
    print(f"Output Path: {args.output_path}")

    if args.dataset == "abo":
        download_abo(download_path=args.output_path)

    elif args.dataset == "cc3m":
        download_cc3m(
            tsv_path=args.tsv_path, 
            split_name=args.split_name, 
            percentage=args.percentage, 
            output_dir=args.output_path
        )

    elif args.dataset == "objaverse_v1":
        download_objaverse_v1(
            download_path=args.output_path, 
            num_processes=args.num_processes, 
            num_objects=args.num_objects
        )

    elif args.dataset == "objaverse_lgm":
        download_objaverse_LGM(
            csv_path=args.csv_path, 
            download_path=args.output_path, 
            num_processes=args.num_processes
        )

    elif args.dataset == "shapenet":
        download_shapenet(download_path=args.output_path)

    elif args.dataset == "objaverse_catg":
        download_objaverse_catg(
            target_classes=target_classes, 
            download_path=args.output_path, 
            num_processes=args.num_processes
        )