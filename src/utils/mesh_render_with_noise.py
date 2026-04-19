import argparse
import os
import sys

import numpy as np
import trimesh
from PIL import Image

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from src.utils.data_utils import get_colored_mesh_composition
from src.utils.render_utils import (
    export_renderings,
    make_grid_for_images_or_videos,
    render_normal_views_around_mesh,
    render_views_around_mesh,
)


def add_noise_to_geometry(
    geometry: trimesh.Trimesh,
    noise_type: str = "gaussian",
    noise_amount: float = 0.01,
) -> trimesh.Trimesh:
    if noise_type != "gaussian":
        raise ValueError(f"Unsupported noise type: {noise_type}")
    noise = np.random.normal(0.0, noise_amount, geometry.vertices.shape)
    geometry.vertices += noise
    return geometry


def render_mesh_with_noise(
    mesh,
    save_dir,
    prefix="rendering",
    input_image_pil=None,
    num_views=36,
    radius=4.0,
    fps=18,
):
    os.makedirs(save_dir, exist_ok=True)

    rendered_images = render_views_around_mesh(mesh, num_views=num_views, radius=radius)
    rendered_normals = render_normal_views_around_mesh(mesh, num_views=num_views, radius=radius)

    grids = [rendered_images, rendered_normals]
    if input_image_pil is not None:
        grids.insert(0, [input_image_pil] * num_views)

    rendered_grids = make_grid_for_images_or_videos(grids, nrow=3 if input_image_pil else 2)

    export_renderings(rendered_images, os.path.join(save_dir, f"{prefix}.gif"), fps=fps)
    export_renderings(rendered_normals, os.path.join(save_dir, f"{prefix}_normal.gif"), fps=fps)
    export_renderings(rendered_grids, os.path.join(save_dir, f"{prefix}_grid.gif"), fps=fps)

    rendered_images[0].save(os.path.join(save_dir, f"{prefix}.png"))
    rendered_normals[0].save(os.path.join(save_dir, f"{prefix}_normal.png"))
    rendered_grids[0].save(os.path.join(save_dir, f"{prefix}_grid.png"))


def main():
    parser = argparse.ArgumentParser(description="Add noise to merged step meshes and render them.")
    parser.add_argument("--folder", required=True, help="Root folder to process")
    parser.add_argument("--noise_type", default="gaussian", choices=["gaussian"], help="Type of noise")
    parser.add_argument("--noise_amount", type=float, default=0.01, help="Noise amount")
    parser.add_argument("--num_views", type=int, default=36, help="Number of rendered views")
    parser.add_argument("--radius", type=float, default=4.0, help="Camera radius")
    parser.add_argument("--fps", type=int, default=18, help="GIF FPS")
    args = parser.parse_args()

    for root, _, files in os.walk(args.folder):
        if not os.path.basename(root).startswith("step"):
            continue

        print(f"Processing folder: {root}")
        part_files = sorted(f for f in files if f.startswith("part_") and f.endswith(".glb"))
        if not part_files:
            print(f"No part_*.glb files in {root}")
            continue

        input_image_path = os.path.join(root, "input_image.png")
        input_image_pil = Image.open(input_image_path) if os.path.exists(input_image_path) else None

        outputs = []
        for part_file in part_files:
            glb_path = os.path.join(root, part_file)
            try:
                loaded = trimesh.load(glb_path, process=False)
                if isinstance(loaded, trimesh.Scene):
                    loaded = loaded.to_geometry()
                if isinstance(loaded, trimesh.Trimesh):
                    noisy_part = add_noise_to_geometry(loaded.copy(), args.noise_type, args.noise_amount)
                    outputs.append(noisy_part)
                else:
                    print(f"Skipping unsupported geometry in {glb_path}: {type(loaded)}")
            except Exception as exc:
                print(f"Error loading {glb_path}: {exc}")

        if not outputs:
            print(f"No valid meshes in {root}")
            continue

        # Match inference flow: merged_mesh = get_colored_mesh_composition(outputs)
        merged_mesh = get_colored_mesh_composition(outputs)

        render_mesh_with_noise(
            merged_mesh,
            root,
            prefix="rendering",
            input_image_pil=input_image_pil,
            num_views=args.num_views,
            radius=args.radius,
            fps=args.fps,
        )


if __name__ == "__main__":
    main()