import warnings
warnings.filterwarnings("ignore")
import diffusers.utils.logging as diffusion_logging
diffusion_logging.set_verbosity_error()

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.typing_utils import *
from src.utils.render_utils import render_views_around_mesh, render_normal_views_around_mesh, export_renderings, make_grid_for_images_or_videos
from src.utils.data_utils import get_colored_mesh_composition

import argparse
import logging
from collections import defaultdict
import trimesh
from PIL import Image
import numpy as np
from tqdm import tqdm
import torch
import accelerate
from accelerate import Accelerator
from accelerate.logging import get_logger as get_accelerate_logger
from torchvision import transforms
import traceback # <<< Added for logging tracebacks

from transformers import BitImageProcessor, Dinov2Model
from src.schedulers import RectifiedFlowScheduler
from src.models.autoencoders import TripoSGVAEModel
from src.models.transformers import PartCrafterDiTModel
from src.pipelines.pipeline_partcrafter import PartCrafterPipeline
from src.utils.train_utils import get_configs
from src.utils.metric_utils import compute_cd_and_f_score_cuda
from src.models.briarmbg import BriaRMBG
from src.utils.image_utils import prepare_image
from huggingface_hub import snapshot_download

from src.datasets import ObjaversePartEvalDataset, collate_fn_eval
from torch.utils.data import DataLoader


@torch.no_grad()
def run_evaluation(
    dataloader: DataLoader,
    pipeline: PartCrafterPipeline,
    accelerator: Accelerator,
    logger: logging.Logger,
    args: argparse.Namespace,
    configs: Dict,
    eval_dir: str,
    rmbg_net: BriaRMBG  # <<< MODIFICATION: Accept rmbg_net >>>
):
    if args.seed >= 0:
        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)
    else:
        generator = None

    metrics_summary = defaultdict(lambda: defaultdict(list))

    progress_bar = tqdm(
        enumerate(dataloader),
        total=args.max_test_steps,
        desc="Evaluating",
        disable=not accelerator.is_main_process
    )

    for step, batch in progress_bar:
        # <<< --- FIX: ADDED TRY/EXCEPT BLOCK --- >>>
        try:
            if step >= args.max_test_steps:
                break

            if configs["test"]["batch_size_per_gpu"] != 1:
                logger.warning("Warning: Evaluation is designed for a batch size of 1.")

            image_path = batch["images"][0]
            # Un-normalize tensor back to PIL image
            # raw_image_pil = transforms.ToPILImage()(image_tensor * 0.5 + 0.5)

            # <<< MODIFICATION: Apply background removal >>>
            # Use .module if using accelerator (DDP)
            _rmbg_net = accelerator.unwrap_model(rmbg_net)
            image_pil = prepare_image(image_path, bg_color=np.array([1.0, 1.0, 1.0]), rmbg_net=_rmbg_net)
            # <<< END MODIFICATION >>>
            
            # Fallback if prepare_image fails and returns None (e.g., from empty contour fix)
            if image_pil is None:
                logger.warning(f"Step: {step:04d} | SKIPPING. Image preprocessing failed (e.g., empty mask).")
                continue

            gt_part_surfaces = batch["part_surfaces"][0]
            num_parts = batch["num_parts"][0]
            
            # <<< --- FIX: ADDED SAFETY CHECK --- >>>
            # This limit is from inference_partcrafter.py (MAX_NUM_PARTS = 16)
            MAX_MODEL_PARTS = 16 
            if num_parts > MAX_MODEL_PARTS:
                logger.warning(f"Step: {step:04d} | SKIPPING. num_parts ({num_parts}) > max ({MAX_MODEL_PARTS}).")
                continue  # Skip this batch and go to the next one
            # <<< --- END FIX --- >>>
            
            # Log the num_parts being used for this item
            logger.info(f"Step: {step:04d} | Processing with num_parts: {num_parts}")
            
            input_image_pil = image_pil 

            for guidance_scale in sorted(args.test_guidance_scales):
                # _pipeline = accelerator.unwrap_model(pipeline)
                import time
                start_time = time.time()
                pred_part_meshes = pipeline(
                    [image_pil] * num_parts,
                    num_inference_steps=configs['test']['num_inference_steps'],
                    num_tokens=configs['model']['vae']['num_tokens'],
                    guidance_scale=guidance_scale,
                    attention_kwargs={"num_parts": num_parts},
                    generator=generator,
                    max_num_expanded_coords=configs['test']['max_num_expanded_coords'],
                    use_flash_decoder=configs['test']['use_flash_decoder'],
                ).meshes
                end_time = time.time()
                logger.info(f"Step: {step:04d} | GS: {guidance_scale:<4.1f} | Generation Time: {end_time - start_time:.2f} seconds")

                batch_cds, batch_f_scores = [], []
                for i in range(num_parts):
                    # ... (rest of the metric calculation loop remains identical) ...
                    pred_mesh = pred_part_meshes[i]
                    gt_surface = gt_part_surfaces[i].to(accelerator.device, dtype=torch.float32) 

                    if pred_mesh is None or len(pred_mesh.vertices) == 0:
                        if accelerator.is_main_process:
                            logger.warning(f"Step: {step:04d} | GS: {guidance_scale:<4.1f} | Part: {i:02d} | SKIPPED: Predicted mesh is None or empty. Using default metrics.")
                        
                        part_cd = torch.tensor(configs['test']['metric']['default_cd'])
                        part_f = torch.tensor(configs['test']['metric']['default_f1'])
                    else:
                        cd, f_score = compute_cd_and_f_score_cuda(
                            gt_surface, pred_mesh,
                            num_samples=configs['test']['metric']['cd_num_samples'],
                            threshold=configs['test']['metric']['f1_score_threshold'],
                        )
                        part_cd = cd.cpu() if isinstance(cd, torch.Tensor) else torch.tensor(cd)
                        part_f = f_score.cpu() if isinstance(f_score, torch.Tensor) else torch.tensor(f_score)
                    
                    if accelerator.is_main_process:
                        logger.info(f"Step: {step:04d} | GS: {guidance_scale:<4.1f} | Part: {i:02d} | CD: {part_cd.item():.4f} | F1: {part_f.item():.4f}")

                    batch_cds.append(part_cd.item())
                    batch_f_scores.append(part_f.item())
                
                metrics_summary[guidance_scale]["chamfer"].extend(batch_cds)
                metrics_summary[guidance_scale]["f1_score"].extend(batch_f_scores)

                if accelerator.is_main_process and args.save_visuals:
                    # ... (rest of the visualization logic remains identical) ...
                    local_eval_dir = os.path.join(eval_dir, f"gs_{guidance_scale:.1f}", f"step_{step:04d}")
                    os.makedirs(local_eval_dir, exist_ok=True)
                    
                    input_image_pil.save(os.path.join(local_eval_dir, "input_image.png"))
                    for i, mesh in enumerate(pred_part_meshes):
                        if mesh: 
                            mesh.export(os.path.join(local_eval_dir, f"part_{i:02d}.glb"))

                    valid_meshes = [mesh for mesh in pred_part_meshes if mesh is not None]
                    
                    if valid_meshes:
                        merged_mesh = get_colored_mesh_composition(valid_meshes)
                        merged_mesh.export(os.path.join(local_eval_dir, "object.glb"))

                        render_cfg = configs['test']['rendering']
                        rendered_images = render_views_around_mesh(merged_mesh, num_views=render_cfg.get('num_views', 36), radius=render_cfg.get('radius', 4.0))
                        rendered_normals = render_normal_views_around_mesh(merged_mesh, num_views=render_cfg.get('num_views', 36), radius=render_cfg.get('radius', 4.0))
                        rendered_grids = make_grid_for_images_or_videos([[input_image_pil] * len(rendered_images), rendered_images, rendered_normals], nrow=3)

                        export_renderings(rendered_images, os.path.join(local_eval_dir, "rendering.gif"), fps=render_cfg.get('fps', 18))
                        rendered_images[0].save(os.path.join(local_eval_dir, "rendering.png"))
                    else:
                        logger.warning(f"Step: {step:04d} | GS: {guidance_scale:<4.1f} | No valid meshes to merge for visualization.")

        except Exception as e:
            logger.error(f"Step: {step:04d} | FAILED WITH UNCAUGHT ERROR. SKIPPING.")
            logger.error(f"Error: {e}")
            # logger.error(traceback.format_exc()) # Logs the full traceback
            continue # Go to the next step
        # <<< --- END FIX --- >>>
            
    if accelerator.is_main_process:
        logger.info("\n" + "="*60 + "\n                 Evaluation Results Summary\n" + "="*60)
        report_path = os.path.join(eval_dir, "results.txt")
        with open(report_path, "w") as f:
            f.write("Evaluation Results Summary\n" + "="*60 + "\n")
            for guidance_scale, metrics in sorted(metrics_summary.items()):
                avg_cd = np.mean(metrics["chamfer"])
                avg_f1 = np.mean(metrics["f1_score"])
                log_msg = f"Guidance Scale: {guidance_scale:<4.1f} | Avg Chamfer Distance: {avg_cd:.4f} | Avg F1-Score: {avg_f1:.4f}"
                logger.info(log_msg)
                f.write(log_msg + "\n")
        logger.info(f"\nResults saved to {report_path}")
        
def main():
    parser = argparse.ArgumentParser(description="Evaluate a PartCrafter model.")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file.")
    parser.add_argument("--checkpoint_path", type=str, required=False, help="Path to the *custom transformer checkpoint* folder.")
    parser.add_argument("--output_dir", type=str, default="evaluation", help="Directory to save results.")
    parser.add_argument("--tag", type=str, default="test_run", help="A specific tag for this run.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--max_test_steps", type=int, default=None, help="Max number of batches to evaluate.")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of workers. Set to 0 to avoid I/O bottlenecks.")
    parser.add_argument("--test_guidance_scales", type=float, nargs="+", default=[7.0], help="List of CFG scales to test.")
    parser.add_argument("--save_visuals", action='store_true', help="Save generated meshes and renders.")
    args, extras = parser.parse_known_args()
    configs = get_configs(args.config, extras)

    eval_dir = os.path.join(args.output_dir, args.tag)
    os.makedirs(eval_dir, exist_ok=True)
    
    accelerator = Accelerator()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
    logger = get_accelerate_logger(__name__, log_level="INFO")
    
    if accelerator.is_main_process:
        fh = logging.FileHandler(os.path.join(eval_dir, "log.txt"))
        fh.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
        logger.logger.addHandler(fh)

    if args.seed >= 0:
        accelerate.utils.set_seed(args.seed)

    # <<< MODIFICATION: Match inference_partcrafter.py loading >>>
    logger.info("Downloading base models and RMBG...")
    partcrafter_weights_dir = "pretrained_weights/PartCrafter"
    rmbg_weights_dir = "pretrained_weights/RMBG-1.4"
    snapshot_download(repo_id="wgsxm/PartCrafter", local_dir=partcrafter_weights_dir)
    snapshot_download(repo_id="briaai/RMBG-1.4", local_dir=rmbg_weights_dir)

    # init rmbg model for background removal
    logger.info("Loading RMBG model...")
    rmbg_net = BriaRMBG.from_pretrained(rmbg_weights_dir).to(accelerator.device)
    rmbg_net.eval() 

    logger.info(f"Loading base models from: {partcrafter_weights_dir}")
    logger.info(f"Loading custom transformer from: {args.checkpoint_path}")
    weight_dtype = torch.float16

    # Load base components from base weights
    vae = TripoSGVAEModel.from_pretrained(partcrafter_weights_dir, subfolder="vae", torch_dtype=weight_dtype)
    scheduler = RectifiedFlowScheduler.from_pretrained(partcrafter_weights_dir, subfolder="scheduler")
    feature_extractor = BitImageProcessor.from_pretrained(partcrafter_weights_dir, subfolder="feature_extractor_dinov2")
    image_encoder = Dinov2Model.from_pretrained(partcrafter_weights_dir, subfolder="image_encoder_dinov2", torch_dtype=weight_dtype)


    if args.checkpoint_path is None:
        transformer = PartCrafterDiTModel.from_pretrained(partcrafter_weights_dir, subfolder="transformer", torch_dtype=weight_dtype)
    else:
        transformer = PartCrafterDiTModel.from_pretrained(args.checkpoint_path, torch_dtype=weight_dtype)


    pipeline = PartCrafterPipeline(vae=vae, transformer=transformer, scheduler=scheduler, feature_extractor_dinov2=feature_extractor, image_encoder_dinov2=image_encoder)
    pipeline.to(accelerator.device, weight_dtype)
    pipeline.set_progress_bar_config(disable=True)
    
    logger.info("Loading test dataset...")
    test_dataset = ObjaversePartEvalDataset(configs=configs, mode='test')
    test_loader = DataLoader(
        test_dataset,
        batch_size=configs["test"]["batch_size_per_gpu"],
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn_eval
    )
    
    # <<< MODIFICATION >>>
    # Prepare pipeline and rmbg_net with accelerator
    test_loader, rmbg_net, pipeline = accelerator.prepare(
        test_loader, rmbg_net, pipeline
    )
    # <<< END MODIFICATION >>>

    if args.max_test_steps is None:
        args.max_test_steps = len(test_loader)

    logger.info(f"Loaded {len(test_dataset)} test samples. Evaluating for {args.max_test_steps} steps.")

    run_evaluation(
        dataloader=test_loader,
        pipeline=pipeline,
        accelerator=accelerator,
        logger=logger,
        args=args,
        configs=configs,
        eval_dir=eval_dir,
        rmbg_net=rmbg_net  # <<< MODIFICATION: Pass rmbg_net >>>
    )

if __name__ == "__main__":
    main()