import os
import json
import shutil
from pathlib import Path

def prune_folders(root_directory, threshold=4, dry_run=True):
    """
    Scans subdirectories in the root_directory. 
    If a folder contains 'num_parts.json' and the 'num_parts' key is > threshold,
    the folder is deleted.
    
    Args:
        root_directory (str): The path to the folder containing the hash IDs (e.g., 'shapenet/chair')
        threshold (int): The limit for num_parts. Folders with values > this are deleted.
        dry_run (bool): If True, only prints what would be deleted. If False, actually deletes.
    """
    
    root_path = Path(root_directory)
    
    if not root_path.exists():
        print(f"Error: Directory '{root_directory}' does not exist.")
        return

    print(f"Scanning directory: {root_path.resolve()}")
    print(f"Mode: {'DRY RUN (No files will be deleted)' if dry_run else 'LIVE (Files WILL be deleted)'}")
    print("-" * 50)

    deleted_count = 0
    error_count = 0
    kept_count = 0

    # Iterate over all items in the root directory
    for item in root_path.iterdir():
        if item.is_dir():
            json_path = item / 'num_parts.json'
            
            if json_path.exists():
                try:
                    with open(json_path, 'r') as f:
                        data = json.load(f)
                    
                    # Check if the key exists and matches condition
                    if 'num_parts' in data:
                        num_parts = data['num_parts']
                        
                        if num_parts > threshold:
                            print(f"[DELETE] Folder: {item.name} (num_parts: {num_parts})")
                            
                            if not dry_run:
                                try:
                                    shutil.rmtree(item)
                                    print(f"    -> Successfully deleted.")
                                    deleted_count += 1
                                except Exception as e:
                                    print(f"    -> Failed to delete: {e}")
                                    error_count += 1
                            else:
                                deleted_count += 1 # Counting potential deletions
                        else:
                            # print(f"[KEEP]   Folder: {item.name} (num_parts: {num_parts})")
                            kept_count += 1
                    else:
                        print(f"[SKIP]   Folder: {item.name} (Key 'num_parts' missing in JSON)")
                        
                except json.JSONDecodeError:
                    print(f"[ERROR]  Folder: {item.name} (Invalid JSON file)")
                    error_count += 1
                except Exception as e:
                    print(f"[ERROR]  Folder: {item.name} (Unexpected error: {e})")
                    error_count += 1
            else:
                # Optional: Print if you want to know about folders missing the JSON file entirely
                # print(f"[SKIP]   Folder: {item.name} (num_parts.json not found)")
                pass

    print("-" * 50)
    if dry_run:
        print(f"Summary (Dry Run):")
        print(f"Would be deleted: {deleted_count}")
        print(f"Would be kept:    {kept_count}")
    else:
        print(f"Summary (Live):")
        print(f"Deleted: {deleted_count}")
        print(f"Kept:    {kept_count}")
        print(f"Errors:  {error_count}")

if __name__ == "__main__":
    # --- CONFIGURATION ---
    
    # Update this path to point to your specific category folder (e.g., 'preprocessed_data/shapenet/chair')
    TARGET_DIRECTORY = "preprocessed_data/shapenet/chair"
    
    # Set this to False ONLY when you are ready to actually delete files
    DRY_RUN_MODE = False 
    
    prune_folders(TARGET_DIRECTORY, threshold=4, dry_run=DRY_RUN_MODE)