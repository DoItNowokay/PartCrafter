import os
import json
import argparse
from tqdm import tqdm

def generate_configs(args):
    """
    Scans a preprocessed data directory, generates per-class JSON configs,
    and then aggregates them into one main config file.
    """
    
    # --- 1. Find all class folders to process ---
    try:
        class_names = [d for d in os.listdir(args.input_dir) if os.path.isdir(os.path.join(args.input_dir, d))]
        if not class_names:
            print(f"Error: No class subdirectories found in {args.input_dir}")
            return
    except FileNotFoundError:
        print(f"Error: Input directory not found: {args.input_dir}")
        return
    except Exception as e:
        print(f"Error reading {args.input_dir}: {e}")
        return

    print(f"Found {len(class_names)} classes: {class_names}")
    
    all_configs = []
    
    # --- 2. Process each class folder ---
    class_bar = tqdm(class_names, desc="Processing Classes")
    for class_name in class_bar:
        class_path = os.path.join(args.input_dir, class_name)
        
        # Find all object ID folders (e.g., '1a8bbf2994788e2743e99e0cae970928')
        try:
            object_ids = [d for d in os.listdir(class_path) if os.path.isdir(os.path.join(class_path, d))]
        except Exception as e:
            print(f"Warning: Could not read {class_path}. Skipping. Error: {e}")
            continue

        if not object_ids:
            print(f"Warning: No object folders found in {class_path}. Skipping.")
            continue
            
        class_configs = []
        
        # --- 3. Scan each object folder ---
        object_bar = tqdm(object_ids, desc=f"Scanning {class_name}", leave=False)
        for object_id in object_bar:
            mesh_folder_path = os.path.join(class_path, object_id)
            
            # Define paths to the expected files
            num_parts_path = os.path.join(mesh_folder_path, 'num_parts.json')
            surface_path = os.path.join(mesh_folder_path, 'points.npy')
            iou_path = os.path.join(mesh_folder_path, 'iou.json')
            
            # Check for images (prefer 'rmbg' but fall back to 'rendering.png')
            image_path_plain = os.path.join(mesh_folder_path, 'rendering.png')
            
            # --- 4. Build the config object (similar to your original script) ---
            config = {
                "dataset": args.dataset_name,
                "file": f"{object_id}.glb",  # Reconstruct original file name
                "folder": class_name,
                "num_parts": 0,
                "valid": False,
                "surface_path": None,
                "image_path": None,
                "iou_mean": 0.0,
                "iou_max": 0.0
            }
            
            try:
                # Load num_parts.json
                with open(num_parts_path, 'r') as f:
                    config["num_parts"] = json.load(f).get('num_parts', 0)
                
                # Load iou.json (if it exists)
                if os.path.exists(iou_path):
                    with open(iou_path, 'r') as f:
                        iou_config = json.load(f)
                        config['iou_mean'] = iou_config.get('iou_mean', 0.0)
                        config['iou_max'] = iou_config.get('iou_max', 0.0)
                
                # Check for surface points
                if not os.path.exists(surface_path):
                    raise FileNotFoundError(f"Missing points.npy in {mesh_folder_path}")
                config['surface_path'] = surface_path
                
                if os.path.exists(image_path_plain):
                    config['image_path'] = image_path_plain
                else:
                    raise FileNotFoundError(f"Missing rendering.png/rendering_rmbg.png in {mesh_folder_path}")
                
                # If all checks passed
                config['valid'] = True
                class_configs.append(config)
                
            except Exception as e:
                # tqdm.write(f"Skipping {mesh_folder_path}: {e}")
                continue # Skip this object if it's incomplete

        # --- 5. Save the per-class JSON config ---
        if class_configs:
            class_configs_path = os.path.join(class_path, 'object_part_configs.json')
            try:
                with open(class_configs_path, 'w') as f:
                    json.dump(class_configs, f, indent=4)
                # tqdm.write(f"Saved {class_configs_path} with {len(class_configs)} valid objects.")
                
                # Add this class's configs to the master list
                all_configs.extend(class_configs)
            except Exception as e:
                tqdm.write(f"Error writing {class_configs_path}: {e}")

    # --- 6. Save the final aggregated JSON config ---
    if all_configs:
        final_configs_path = os.path.join(args.input_dir, args.output_file)
        try:
            with open(final_configs_path, 'w') as f:
                json.dump(all_configs, f, indent=4)
            print(f"\n🎉 Success! Aggregated {len(all_configs)} configs from {len(class_names)} classes.")
            print(f"Final aggregated config file saved to: {final_configs_path}")
        except Exception as e:
            print(f"\nError writing final aggregated file {final_configs_path}: {e}")
    else:
        print("\nNo valid objects were found to aggregate.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate JSON configs from a preprocessed dataset directory.")
    
    parser.add_argument('--input_dir', 
                        type=str, 
                        required=True, 
                        help='Base directory of preprocessed class folders (e.g., "preprocessed_data/shapenet")')
    
    parser.add_argument('--output_file', 
                        type=str, 
                        default='object_part_configs.json', 
                        help='Name for the final aggregated JSON file (saved in input_dir)')
    
    parser.add_argument('--dataset_name', 
                        type=str, 
                        default='ShapeNet', 
                        help='Name of the dataset to write into the JSON (e.g., ShapeNet, Objaverse)')

    args = parser.parse_args()
    
    generate_configs(args)