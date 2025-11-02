import os
import json
import argparse
import time
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

print('Init done')

# --- ShapeNet to Objaverse Class Name Mapping ---
# This dictionary controls which classes will be processed by default.
SHAPENET_MAP = {
    "airplane": "airplane", "basket": "basket", "bathtub": "bathtub", "bed": "bed",
    "bench": "bench", "birdhouse": "birdhouse", "bookshelf": "bookcase", "bottle": "bottle",
    "bowl": "bowl", "bus": "bus_(vehicle)", "cabinet": "cabinet", "camera": "camera",
    "can": "can", "cap": "cap_(headwear)", "car": "car_(automobile)", "chair": "chair",
    "clock": "clock", "dishwasher": "dishwasher", "earphone": "earphone", "faucet": "faucet",
    "guitar": "guitar", "helmet": "helmet", "jar": "jar", "knife": "knife",
    "lamp": "lamp", "laptop": "laptop_computer", "mailbox": "mailbox_(at_home)",
    "microphone": "microphone", "microwave": "microwave_oven", "motorcycle": "motorcycle",
    "mug": "mug", "piano": "piano", "pillow": "pillow", "pistol": "pistol",
    "pot": "pot", "printer": "printer", "skateboard": "skateboard", "sofa": "sofa",
    "stove": "stove", "table": "table", "telephone": "telephone", "train": "train_(railroad_vehicle)"
}
# SHAPENET_MAP = {
#     "chair": "chair"
# }

def preprocess(args):
    input_path = args['input']
    output_path = args['output']
    class_name = args['class_name']
    dataset_name = args.get('dataset_name', 'Objaverse')
    # device = args.get('device', 'cuda:0')

    assert os.path.exists(input_path), f'{input_path} does not exist'

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    source_meshes = {f for f in os.listdir(input_path) if f.endswith('.glb')}
    processed_folders = {f for f in os.listdir(output_path) if os.path.isdir(os.path.join(output_path, f))}
    processed_meshes_base = {folder for folder in processed_folders}
    meshes_to_process = {mesh for mesh in source_meshes if mesh.replace('.glb', '') not in processed_meshes_base}

    if args['limit'] == -1:
        meshes_to_process = list(meshes_to_process)
    else:
        meshes_to_process = list(meshes_to_process)[:min(args['limit'], len(meshes_to_process))]
    if not meshes_to_process:
        print(f"✅ Class '{class_name}' is already fully processed. Skipping.")
    else:
        print(f"Found {len(meshes_to_process)} unprocessed meshes in '{class_name}'.")
        for mesh_name in tqdm(meshes_to_process, desc=f"Processing {class_name}"):
            mesh_path = os.path.join(input_path, mesh_name)
            mesh_path_quoted = f'"{mesh_path}"'
            output_path_quoted = f'"{output_path}"'

            os.system(f"python3 datasets/preprocess/mesh_to_point.py --input {mesh_path_quoted} --output {output_path_quoted}")
            os.system(f"python3 datasets/preprocess/render.py --input {mesh_path_quoted} --output {output_path_quoted}")
            export_mesh_folder = os.path.join(output_path, mesh_name.replace('.glb', ''))
            export_rendering_path = os.path.join(export_mesh_folder, 'rendering.png')
            export_rendering_path_quoted = f'"{export_rendering_path}"'
            os.system(f"python3 datasets/preprocess/rmbg.py --input {export_rendering_path_quoted} --output {output_path_quoted}")
            os.system(f"python3 datasets/preprocess/calculate_iou.py --input {mesh_path_quoted} --output {output_path_quoted}")
            time.sleep(1)

    configs = []
    for mesh_name in source_meshes:
        mesh_folder_path = os.path.join(output_path, mesh_name.replace('.glb', ''))
        num_parts_path = os.path.join(mesh_folder_path, 'num_parts.json')
        surface_path = os.path.join(mesh_folder_path, 'points.npy')
        image_path = os.path.join(mesh_folder_path, 'rendering_rmbg.png')
        iou_path = os.path.join(mesh_folder_path, 'iou.json')

        config = {
            "dataset": dataset_name, "file": mesh_name, "folder": class_name,
            "num_parts": 0, "valid": False, "mesh_path": os.path.join(input_path, mesh_name),
            "surface_path": None, "image_path": None, "iou_mean": 0.0, "iou_max": 0.0
        }
        try:
            config["num_parts"] = json.load(open(num_parts_path))['num_parts']
            iou_config = json.load(open(iou_path))
            config['iou_mean'] = iou_config['iou_mean']
            config['iou_max'] = iou_config['iou_max']
        except Exception:
            continue

        try:
            assert os.path.exists(surface_path)
            config['surface_path'] = surface_path
            assert os.path.exists(image_path)
            config['image_path'] = image_path
            config['valid'] = True
            configs.append(config)
        except Exception:
            continue

    configs_path = os.path.join(output_path, 'object_part_configs.json')
    with open(configs_path, 'w') as f:
        json.dump(configs, f, indent=4)

    return configs_path


if __name__ == '__main__':
    # print(13)
    parser = argparse.ArgumentParser(description="Run preprocessing for 3D models.")
    parser.add_argument('--input', type=str, default="/scratch/anoushkrit.scee.iitmandi/DATA/objaverse/categorized_objaverse/hf-objaverse-v1/glbs", help='Base directory of class folders.')
    parser.add_argument('--output', type=str, default="preprocessed_data", help='Base directory for saving output.')
    parser.add_argument('--dataset_name', type=str, default='Objaverse', help='Name of the dataset.')

    # --- NEW ARGUMENTS ---
    parser.add_argument('--sequential', action='store_true', help='Run processing sequentially instead of in parallel.')
    parser.add_argument('--classes', nargs='+', type=str, help='A list of specific class names (ShapeNet names) to process.')
    parser.add_argument('--workers', type=int, default=int(cpu_count() * 0.75), help='Number of parallel workers (if not sequential).')
    parser.add_argument('--limit', type=int, default=-1, help='Limit the number of meshes to process per class.')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use for processing (e.g., cuda:0 or cpu).')
    parser.add_argument('--shapenet', action='store_true', help='Use ShapeNet classes Objaverse.')

    args = parser.parse_args()

    tasks = []
    print("Preparing tasks...")

    # --- NEW LOGIC: Determine which classes to process ---
    if args.shapenet:
        if args.classes:
            # User has provided a specific list of classes
            class_map_to_process = {c: SHAPENET_MAP.get(c) for c in args.classes if c in SHAPENET_MAP}
            print(f"Processing user-specified classes: {list(class_map_to_process.keys())}")
        else:
            # No specific classes provided, use the full default map
            class_map_to_process = SHAPENET_MAP
            print("Processing all default classes defined in SHAPENET_MAP.")
    else:
        if args.classes:
            # User has provided a specific list of classes without ShapeNet mapping
            class_map_to_process = {c: c for c in args.classes}
            print(f"Processing user-specified classes without ShapeNet mapping: {list(class_map_to_process.keys())}")
        else:
            # No specific classes provided, process all folders in input directory
            available_folders = [f for f in os.listdir(args.input) if os.path.isdir(os.path.join(args.input, f))]
            class_map_to_process = {f: f for f in available_folders}
            print("Processing all available class folders in the input directory.")
            print(class_map_to_process)
        

    for objaverse_name in class_map_to_process.values():
        class_input_path = os.path.join(args.input, objaverse_name)
        if os.path.isdir(class_input_path):
            tasks.append({
                'input': class_input_path,
                'output': os.path.join(args.output, objaverse_name),
                'class_name': objaverse_name,
                'dataset_name': args.dataset_name,
                'limit': args.limit,
                'device': args.device
            })
        else:
            print(f"⚠️  Warning: Source folder '{objaverse_name}' not found. Skipping class '{objaverse_name}'.")

    if not tasks:
        print("Error: No valid class folders found to process.")
        exit(1)

    print(f"Found {len(tasks)} valid classes to process.")

    results = []
    # --- NEW LOGIC: Choose between parallel or sequential execution ---
    if args.sequential:
        print("Running in sequential mode.")
        for task in tqdm(tasks, desc="Overall Progress"):
            result = preprocess(task)
            results.append(result)
    else:
        print(f"Running in parallel mode with up to {args.workers} workers.")
        with Pool(processes=args.workers) as pool:
            results = list(tqdm(pool.imap_unordered(preprocess, tasks), total=len(tasks), desc="Overall Progress"))

    # --- Aggregation step remains the same ---
    print("\nAggregating all class-specific configuration files...")
    all_configs = []
    for config_path in results:
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                all_configs.extend(json.load(f))

    final_configs_path = os.path.join(args.output, 'object_part_configs.json')
    with open(final_configs_path, 'w') as f:
        json.dump(all_configs, f, indent=4)

    print(f"\n🎉 All classes have been processed. Final aggregated config file saved to: {final_configs_path}")