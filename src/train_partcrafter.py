import itertools
import warnings
from xml.parsers.expat import model
warnings.filterwarnings("ignore")  # ignore all warnings
import diffusers.utils.logging as diffusion_logging
diffusion_logging.set_verbosity_error()  # ignore diffusers warnings

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.typing_utils import *

import os
import argparse
import logging
import time
import math
import gc
from packaging import version

import trimesh
from PIL import Image
import numpy as np
import wandb
from tqdm import tqdm

import torch
import torch.nn.functional as tF
import accelerate
from accelerate import Accelerator
from accelerate.logging import get_logger as get_accelerate_logger
from accelerate import DataLoaderConfiguration, DeepSpeedPlugin
from diffusers.training_utils import (
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3
)

from transformers import (
    BitImageProcessor,
    Dinov2Model,
    CLIPTextModel,
    CLIPTokenizer
)
from src.schedulers import RectifiedFlowScheduler
from src.models.autoencoders import TripoSGVAEModel
from src.models.transformers import PartCrafterDiTModel
from src.pipelines.pipeline_partcrafter import PartCrafterPipeline
from src.models.condition_processor import ConditionProcessor

from src.datasets import (
    ObjaversePartDataset,
    BatchedObjaversePartDataset,
    ObjaverseCaptionDataset,
    BatchedObjaverseCaptionDataset,
    MultiEpochsDataLoader,
    yield_forever
)
from src.utils.data_utils import get_colored_mesh_composition
from src.utils.train_utils import (
    MyEMAModel,
    get_configs,
    get_optimizer,
    get_lr_scheduler,
    save_experiment_params,
    save_model_architecture,
)
from src.utils.render_utils import (
    render_views_around_mesh,
    render_normal_views_around_mesh,
    make_grid_for_images_or_videos,
    export_renderings
)
from src.utils.metric_utils import compute_cd_and_f_score_in_training

def main():
    PROJECT_NAME = "PartCrafter"

    parser = argparse.ArgumentParser(
        description="Train a diffusion model for 3D object generation",
    )

    # ... (all your arguments are correct, no changes needed here) ...
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the config file"
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Tag that refers to the current experiment"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Path to the output directory"
    )
    parser.add_argument(
        "--resume_from_iter",
        type=int,
        default=None,
        help="The iteration to load the checkpoint from"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for the PRNG"
    )
    parser.add_argument(
        "--offline_wandb",
        action="store_true",
        help="Use offline WandB for experiment tracking"
    )

    parser.add_argument(
        "--no_wandb",
        action="store_true",
        help="Disable WandB for experiment tracking"
    )

    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="The max iteration step for training"
    )
    parser.add_argument(
        "--max_val_steps",
        type=int,
        default=2,
        help="The max iteration step for validation"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=32,
        help="The number of processed spawned by the batch provider"
    )
    parser.add_argument(
        "--pin_memory",
        action="store_true",
        help="Pin memory for the data loader"
    )

    parser.add_argument(
        "--use_ema",
        action="store_true",
        help="Use EMA model for training"
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        help="Scale lr with total batch size (base batch size: 256)"
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.,
        help="Max gradient norm for gradient clipping"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass"
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
        help="Type of mixed precision training"
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help="Enable TF32 for faster training on Ampere GPUs"
    )

    parser.add_argument(
        "--val_guidance_scales",
        type=list,
        nargs="+",
        default=[7.0],
        help="CFG scale used for validation"
    )

    parser.add_argument(
        "--use_deepspeed",
        action="store_true",
        help="Use DeepSpeed for training"
    )
    parser.add_argument(
        "--zero_stage",
        type=int,
        default=1,
        choices=[1, 2, 3],  # https://huggingface.co/docs/accelerate/usage_guides/deepspeed
        help="ZeRO stage type for DeepSpeed"
    )

    parser.add_argument(
        "--from_scratch",
        action="store_true",
        help="Train from scratch"
    )
    parser.add_argument(
        "--load_pretrained_model",
        type=str,
        default=None,
        help="Tag of a pretrained PartCrafterDiTModel in this project"
    )
    parser.add_argument(
        "--load_pretrained_model_ckpt",
        type=int,
        default=-1,
        help="Iteration of the pretrained PartCrafterDiTModel checkpoint"
    )
    parser.add_argument(
        "--text_conditioning",
        type=str,
        default="none",
        choices=["none", "direct_text", "contrastive_text"],
        help="Whether to use text conditioning and which type"
    )
    parser.add_argument(
        "--editing",
        type=bool,
        default=False,
        help="Whether to perform editing"
    )

    # Parse the arguments
    args, extras = parser.parse_known_args()
    # Parse the config file
    configs = get_configs(args.config, extras)  # change yaml configs by `extras`

    args.val_guidance_scales = [float(x[0]) if isinstance(x, list) else float(x) for x in args.val_guidance_scales]
    if args.max_val_steps > 0:
        # If enable validation, the max_val_steps must be a multiple of nrow
        # Always keep validation batchsize 1
        divider = configs["val"]["nrow"]
        args.max_val_steps = max(args.max_val_steps, divider)
        if args.max_val_steps % divider != 0:
            args.max_val_steps = (args.max_val_steps // divider + 1) * divider

    # Create an experiment directory using the `tag`
    if args.tag is None:
        args.tag = time.strftime("%Y%m%d_%H_%M_%S")
    exp_dir = os.path.join(args.output_dir, args.tag)
    ckpt_dir = os.path.join(exp_dir, "checkpoints")
    eval_dir = os.path.join(exp_dir, "evaluations")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)

    # Initialize the logger
    logging.basicConfig(
        format="%(asctime)s - %(message)s",
        datefmt="%Y/%m/%d %H:%M:%S",
        level=logging.INFO
    )
    logger = get_accelerate_logger(__name__, log_level="INFO")
    file_handler = logging.FileHandler(os.path.join(exp_dir, "log.txt"))  # output to file
    file_handler.setFormatter(logging.Formatter(
        fmt="%(asctime)s - %(message)s",
        datefmt="%Y/%m/%d %H:%M:%S"
    ))
    logger.logger.addHandler(file_handler)
    logger.logger.propagate = True  # propagate to the root logger (console)

    # Set DeepSpeed config
    if args.use_deepspeed:
        deepspeed_plugin = DeepSpeedPlugin(
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            gradient_clipping=args.max_grad_norm,
            zero_stage=int(args.zero_stage),
            offload_optimizer_device="cpu",  # hard-coded here, TODO: make it configurable
        )
    else:
        deepspeed_plugin = None

    # Initialize the accelerator
    accelerator = Accelerator(
        project_dir=exp_dir,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        split_batches=False,  # batch size per GPU
        dataloader_config=DataLoaderConfiguration(non_blocking=args.pin_memory),
        deepspeed_plugin=deepspeed_plugin,
    )
    logger.info(f"Accelerator state:\n{accelerator.state}\n")

    # Set the random seed
    if args.seed >= 0:
        accelerate.utils.set_seed(args.seed)
        logger.info(f"You have chosen to seed([{args.seed}]) the experiment [{args.tag}]\n")

    # Enable TF32 for faster training on Ampere GPUs
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.text_conditioning:
        train_dataset = BatchedObjaverseCaptionDataset(
            configs=configs,
            batch_size=configs["train"]["batch_size_per_gpu"],
            is_main_process=accelerator.is_main_process,
            shuffle=True,
            training=True,
        )
        val_dataset = ObjaverseCaptionDataset(
            configs=configs,
            training=False,
        )
    else: # fallback to default (image conditioned training)
        train_dataset = BatchedObjaversePartDataset(
            configs=configs,
            batch_size=configs["train"]["batch_size_per_gpu"],
            is_main_process=accelerator.is_main_process,
            shuffle=True,
            training=True,
        )
        val_dataset = ObjaversePartDataset(
            configs=configs,
            training=False,
        )
    train_loader = MultiEpochsDataLoader(
        train_dataset,
        batch_size=configs["train"]["batch_size_per_gpu"],
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=args.pin_memory,
        collate_fn=train_dataset.collate_fn,
    )
    val_loader = MultiEpochsDataLoader(
        val_dataset,
        batch_size=configs["val"]["batch_size_per_gpu"],
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=args.pin_memory,
    )
    random_val_loader = MultiEpochsDataLoader(
        val_dataset,
        batch_size=configs["val"]["batch_size_per_gpu"],
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=args.pin_memory,
    )

    logger.info(f"Loaded [{len(train_dataset)}] training samples and [{len(val_dataset)}] validation samples\n")

    # Compute the effective batch size and scale learning rate
    total_batch_size = configs["train"]["batch_size_per_gpu"] * \
        accelerator.num_processes * args.gradient_accumulation_steps
    configs["train"]["total_batch_size"] = total_batch_size
    if args.scale_lr:
        configs["optimizer"]["lr"] *= (total_batch_size / 256)
        configs["lr_scheduler"]["max_lr"] = configs["optimizer"]["lr"]

    # Initialize the model
    logger.info("Initializing the model...")
    vae = TripoSGVAEModel.from_pretrained(
        configs["model"]["pretrained_model_name_or_path"],
        subfolder="vae"
    )
    feature_extractor_dinov2 = BitImageProcessor.from_pretrained(
        configs["model"]["pretrained_model_name_or_path"],
        subfolder="feature_extractor_dinov2"
    )
    image_encoder_dinov2 = Dinov2Model.from_pretrained(
        configs["model"]["pretrained_model_name_or_path"],
        subfolder="image_encoder_dinov2"
    )
    if args.text_conditioning:
        tokenizer = CLIPTokenizer.from_pretrained(
            configs["model"]["text_encoder_name"]
        )
        text_encoder = CLIPTextModel.from_pretrained(
            configs["model"]["text_encoder_name"]
        )
    if args.load_pretrained_model:
        # condition_processor = ConditionProcessor.from_pretrained(
        #     configs["model"]["pretrained_model_name_or_path"],
        #     subfolder="condition_processor"
        # )
        logger.info(f"Load pretrained ConditionProcessor to initialize from [{args.load_pretrained_model}] iteration [{args.load_pretrained_model_ckpt:06d}]\n")
        condition_processor = ConditionProcessor.from_pretrained(
            configs["model"]["pretrained_model_name_or_path"],
            subfolder="condition_processor",
            text_conditioning=args.text_conditioning
        )
    else:
        logger.info("Initialize ConditionProcessor from pretrained model\n")
        condition_processor = ConditionProcessor.from_config(
            os.path.join(
                configs["model"]["pretrained_model_name_or_path"],
                "condition_processor"
            ),
            text_conditioning=args.text_conditioning
        )

    enable_part_embedding = configs["model"]["transformer"].get("enable_part_embedding", True)
    enable_local_cross_attn = configs["model"]["transformer"].get("enable_local_cross_attn", True)
    enable_global_cross_attn = configs["model"]["transformer"].get("enable_global_cross_attn", True)
    global_attn_block_ids = configs["model"]["transformer"].get("global_attn_block_ids", None)
    if global_attn_block_ids is not None:
        global_attn_block_ids = list(global_attn_block_ids)
    global_attn_block_id_range = configs["model"]["transformer"].get("global_attn_block_id_range", None)
    if global_attn_block_id_range is not None:
        global_attn_block_id_range = list(global_attn_block_id_range)
    if args.from_scratch:
        logger.info(f"Initialize PartCrafterDiTModel from scratch\n")
        transformer = PartCrafterDiTModel.from_config(
            os.path.join(
                configs["model"]["pretrained_model_name_or_path"],
                "transformer"
            ),
            enable_part_embedding=enable_part_embedding,
            enable_local_cross_attn=enable_local_cross_attn,
            enable_global_cross_attn=enable_global_cross_attn,
            global_attn_block_ids=global_attn_block_ids,
            global_attn_block_id_range=global_attn_block_id_range,
        )
    elif args.load_pretrained_model is None:
        logger.info(f"Load pretrained TripoSGDiTModel to initialize PartCrafterDiTModel from [{configs['model']['pretrained_model_name_or_path']}]\n")
        transformer, loading_info = PartCrafterDiTModel.from_pretrained(
            configs["model"]["pretrained_model_name_or_path"],
            subfolder="transformer",
            low_cpu_mem_usage=False,
            output_loading_info=True,
            enable_part_embedding=enable_part_embedding,
            enable_local_cross_attn=enable_local_cross_attn,
            enable_global_cross_attn=enable_global_cross_attn,
            global_attn_block_ids=global_attn_block_ids,
            global_attn_block_id_range=global_attn_block_id_range,
        )
    else:
        logger.info(f"Load PartCrafterDiTModel EMA checkpoint from [{args.load_pretrained_model}] iteration [{args.load_pretrained_model_ckpt:06d}]\n")
        path = os.path.join(
            args.output_dir,
            args.load_pretrained_model,
            "checkpoints",
            f"{args.load_pretrained_model_ckpt:06d}"
        )
        transformer, loading_info = PartCrafterDiTModel.from_pretrained(
            path,
            subfolder="transformer_ema",
            low_cpu_mem_usage=False,
            output_loading_info=True,
            enable_part_embedding=enable_part_embedding,
            enable_local_cross_attn=enable_local_cross_attn,
            enable_global_cross_attn=enable_global_cross_attn,
            global_attn_block_ids=global_attn_block_ids,
            global_attn_block_id_range=global_attn_block_id_range,
        )
    if not args.from_scratch:
        for v in loading_info.values():
            if v and len(v) > 0:
                logger.info(f"Loading info of PartCrafterDiTModel: {loading_info}\n")
                break

    noise_scheduler = RectifiedFlowScheduler.from_pretrained(
        configs["model"]["pretrained_model_name_or_path"],
        subfolder="scheduler"
    )

    if args.use_ema:
        ema_transformer = MyEMAModel(
            transformer.parameters(),
            model_cls=PartCrafterDiTModel,
            model_config=transformer.config,
            **configs["train"]["ema_kwargs"]
        )

    # Freeze VAE, image encoder, and text encoder
    vae.requires_grad_(False)
    image_encoder_dinov2.requires_grad_(False)
    vae.eval()
    image_encoder_dinov2.eval()
    if args.text_conditioning != "none":
        text_encoder.requires_grad_(False)
        text_encoder.eval()

    trainable_modules = configs["train"].get("trainable_modules", None)
    if trainable_modules is None:
        transformer.requires_grad_(True)
    else:
        trainable_module_names = []
        transformer.requires_grad_(False)
        for name, module in transformer.named_modules():
            for module_name in tuple(trainable_modules.split(",")):
                if module_name in name:
                    for params in module.parameters():
                        params.requires_grad = True
                    trainable_module_names.append(name)
        logger.info(f"Trainable parameter names: {trainable_module_names}\n")

    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        def _subfolder_for_model(model):
            # map model type/name to a sensible subfolder
            name = type(model).__name__.lower()
            if "partcrafterdit" in name or "transformer" in name:
                return "transformer"
            if "conditionprocessor" in name or "condition_processor" in name:
                return "condition_processor"
            # fallback: use class name
            return name

        def save_model_hook(models, weights, output_dir):
            if not accelerator.is_main_process:
                return
            # save EMA separately
            if args.use_ema:
                try:
                    ema_transformer.save_pretrained(os.path.join(output_dir, "transformer_ema"))
                except Exception as e:
                    logger.error(f"Failed to save EMA transformer: {e}")

            # save each model to its own subfolder
            for model in models:
                subfolder = _subfolder_for_model(model)
                try:
                    model.save_pretrained(os.path.join(output_dir, subfolder))
                except Exception as e:
                    logger.error(f"Failed to save model {type(model).__name__} to {subfolder}: {e}")
                if weights:
                    # keep behavior of popping weights if present to free memory (preserve original intent)
                    try:
                        weights.pop()
                    except Exception:
                        pass

        def load_model_hook(models, input_dir):
            # load EMA if present
            if args.use_ema:
                ema_path = os.path.join(input_dir, "transformer_ema")
                if os.path.isdir(ema_path):
                    try:
                        load_model = MyEMAModel.from_pretrained(ema_path, PartCrafterDiTModel)
                        ema_transformer.load_state_dict(load_model.state_dict())
                        ema_transformer.to(accelerator.device)
                        del load_model
                    except Exception as e:
                        logger.error(f"Failed to load EMA transformer from {ema_path}: {e}")

            # load remaining models from their respective subfolders
            for _ in range(len(models)):
                model = models.pop()
                subfolder = _subfolder_for_model(model)
                model_path = os.path.join(input_dir, subfolder)
                if not os.path.isdir(model_path):
                    logger.warning(f"No saved weights at {model_path} for model {type(model).__name__}; skipping.")
                    continue
                try:
                    # Known type: PartCrafterDiTModel
                    if isinstance(model, PartCrafterDiTModel) or "transformer" in subfolder:
                        loaded = PartCrafterDiTModel.from_pretrained(model_path, subfolder=None)
                        model.register_to_config(**loaded.config)
                        model.load_state_dict(loaded.state_dict())
                        del loaded
                        continue

                    # Known type: ConditionProcessor
                    if type(model).__name__.lower().startswith("conditionprocessor") or "condition_processor" in subfolder:
                        try:
                            loaded = ConditionProcessor.from_pretrained(model_path, subfolder=None)
                            # ConditionProcessor may not have register_to_config; try best-effort
                            if hasattr(model, "register_to_config") and hasattr(loaded, "config"):
                                model.register_to_config(**loaded.config)
                            model.load_state_dict(loaded.state_dict())
                            del loaded
                            continue
                        except Exception:
                            # fallback to generic from_pretrained on the class if available
                            pass

                    # Generic fallback: try class.from_pretrained
                    loaded = None
                    if hasattr(type(model), "from_pretrained"):
                        loaded = type(model).from_pretrained(model_path, subfolder=None)
                    if loaded is not None:
                        if hasattr(model, "register_to_config") and hasattr(loaded, "config"):
                            model.register_to_config(**loaded.config)
                        model.load_state_dict(loaded.state_dict())
                        del loaded
                    else:
                        logger.warning(f"Unable to load model {type(model).__name__} from {model_path}")
                except Exception as e:
                    logger.error(f"Failed to load model {type(model).__name__} from {model_path}: {e}")

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    if configs["train"]["grad_checkpoint"]:
        transformer.enable_gradient_checkpointing()

    logger.info("Initializing the optimizer and learning rate scheduler...\n")
    name_lr_mult = configs["train"].get("name_lr_mult", None)
    lr_mult = configs["train"].get("lr_mult", 1.0)

    def add_params_with_lr_mult(model, name_lr_mult, params, params_lr_mult, names_lr_mult):
        for name, param in model.named_parameters():
            if name_lr_mult is not None:
                for k in name_lr_mult.split(","):
                    if k in name:
                        params_lr_mult.append(param)
                        names_lr_mult.append(name)
                        break  # Once it's found, no need to continue checking other keywords
                if name not in names_lr_mult:
                    params.append(param)
            else:
                params.append(param)

    params = []
    params_lr_mult = []
    names_lr_mult = []
    add_params_with_lr_mult(transformer, name_lr_mult, params, params_lr_mult, names_lr_mult)
    add_params_with_lr_mult(condition_processor, name_lr_mult, params, params_lr_mult, names_lr_mult)

    optimizer = get_optimizer(
        params=[
            {"params": params, "lr": configs["optimizer"]["lr"]},
            {"params": params_lr_mult, "lr": configs["optimizer"]["lr"] * lr_mult}
        ],
        **configs["optimizer"]
    )
    if name_lr_mult is not None:
        logger.info(f"Learning rate x [{lr_mult}] parameter names: {names_lr_mult}\n")

    configs["lr_scheduler"]["total_steps"] = configs["train"]["epochs"] * math.ceil(
        len(train_loader) // accelerator.num_processes / args.gradient_accumulation_steps)
    configs["lr_scheduler"]["total_steps"] *= accelerator.num_processes
    if "num_warmup_steps" in configs["lr_scheduler"]:
        configs["lr_scheduler"]["num_warmup_steps"] *= accelerator.num_processes
    lr_scheduler = get_lr_scheduler(optimizer=optimizer, **configs["lr_scheduler"])
    configs["lr_scheduler"]["total_steps"] //= accelerator.num_processes
    if "num_warmup_steps" in configs["lr_scheduler"]:
        configs["lr_scheduler"]["num_warmup_steps"] //= accelerator.num_processes

    transformer, condition_processor, optimizer, lr_scheduler, train_loader, val_loader, random_val_loader = accelerator.prepare(
        transformer, condition_processor, optimizer, lr_scheduler, train_loader, val_loader, random_val_loader
    )

    if args.use_ema:
        ema_transformer.to(accelerator.device)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    vae.to(accelerator.device, dtype=weight_dtype)
    image_encoder_dinov2.to(accelerator.device, dtype=weight_dtype)
    condition_processor.to(accelerator.device)
    if args.text_conditioning != "none":
        text_encoder.to(accelerator.device, dtype=weight_dtype)

    updated_steps_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
    total_updated_steps = configs["lr_scheduler"]["total_steps"]
    if args.max_train_steps is None:
        args.max_train_steps = total_updated_steps
    assert configs["train"]["epochs"] * updated_steps_per_epoch == total_updated_steps
    if accelerator.num_processes > 1 and accelerator.is_main_process:
        print()
    accelerator.wait_for_everyone()
    logger.info(f"Total batch size: [{total_batch_size}]")
    logger.info(f"Learning rate: [{configs['optimizer']['lr']}]")
    logger.info(f"Gradient Accumulation steps: [{args.gradient_accumulation_steps}]")
    logger.info(f"Total epochs: [{configs['train']['epochs']}]")
    logger.info(f"Total steps: [{total_updated_steps}]")
    logger.info(f"Steps for updating per epoch: [{updated_steps_per_epoch}]")
    logger.info(f"Steps for validation: [{len(val_loader)}]\n")

    global_update_step = 0
    if args.resume_from_iter is not None:
        if args.resume_from_iter < 0:
            args.resume_from_iter = int(sorted(os.listdir(ckpt_dir))[-1])
        logger.info(f"Load checkpoint from iteration [{args.resume_from_iter}]\n")
        if version.parse(torch.__version__) >= version.parse("2.4.0"):
            torch.serialization.add_safe_globals([
                int, list, dict,
                defaultdict,
                Any,
                DictConfig, ListConfig, Metadata, ContainerMetadata, AnyNode
            ])
        accelerator.load_state(os.path.join(ckpt_dir, f"{args.resume_from_iter:06d}"))
        global_update_step = int(args.resume_from_iter)

    if accelerator.is_main_process:
        exp_params = save_experiment_params(args, configs, exp_dir)
        save_model_architecture(accelerator.unwrap_model(transformer), exp_dir)

    if accelerator.is_main_process:
        if args.offline_wandb:
            os.environ["WANDB_MODE"] = "offline"
        wandb.init(
            project=PROJECT_NAME, name=args.tag,
            config=exp_params, dir=exp_dir,
            resume=True
        )
        arti_exp_info = wandb.Artifact(args.tag, type="exp_info")
        arti_exp_info.add_file(os.path.join(exp_dir, "params.yaml"))
        arti_exp_info.add_file(os.path.join(exp_dir, "model.txt"))
        arti_exp_info.add_file(os.path.join(exp_dir, "log.txt"))
        wandb.log_artifact(arti_exp_info)

    def get_sigmas(timesteps: Tensor, n_dim: int, dtype=torch.float32):
        sigmas = noise_scheduler.sigmas.to(dtype=dtype, device=accelerator.device)
        schedule_timesteps = noise_scheduler.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero()[0].item() for t in timesteps]
        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    if accelerator.is_main_process:
        print()
    logger.info(f"Start training into {exp_dir}\n")
    logger.logger.propagate = False
    progress_bar = tqdm(
        range(total_updated_steps),
        initial=global_update_step,
        desc="Training",
        ncols=125,
        disable=not accelerator.is_main_process
    )
    for batch in yield_forever(train_loader):

        if global_update_step == args.max_train_steps:
            progress_bar.close()
            logger.logger.propagate = True
            if accelerator.is_main_process and not args.no_wandb:
                wandb.finish()
            logger.info("Training finished!\n")
            return

        transformer.train()
        condition_processor.train()

        with accelerator.accumulate(transformer, condition_processor):

            images = batch["images"]
            with torch.no_grad():
                images = feature_extractor_dinov2(images=images, return_tensors="pt").pixel_values
            images = images.to(device=accelerator.device, dtype=weight_dtype)
            with torch.no_grad():
                image_embeds = image_encoder_dinov2(images).last_hidden_state
            negative_image_embeds = torch.zeros_like(image_embeds)

            # --- CORRECTED PART 2: Use the initialized tokenizer instance ---
            text_embeds = None
            if args.text_conditioning != "none":
                texts = batch["captions"]
                with torch.no_grad():
                    text_inputs = tokenizer(
                        texts,
                        padding="max_length",
                        max_length=tokenizer.model_max_length,
                        truncation=True,
                        return_tensors="pt",
                    )
                    text_inputs = {k: v.to(accelerator.device) for k, v in text_inputs.items()}
                    text_embeds = text_encoder(**text_inputs).last_hidden_state

                    uncond_input = tokenizer(
                        [""] * len(texts),
                        padding="max_length",
                        max_length=tokenizer.model_max_length,
                        truncation=True,
                        return_tensors="pt",
                    )
                    uncond_input = {k: v.to(accelerator.device) for k, v in uncond_input.items()}
                    negative_text_embeds = text_encoder(**uncond_input).last_hidden_state

            part_surfaces = batch["part_surfaces"]
            part_surfaces = part_surfaces.to(device=accelerator.device, dtype=weight_dtype)

            num_parts = batch["num_parts"]
            num_objects = num_parts.shape[0]

            with torch.no_grad():
                latents = vae.encode(
                    part_surfaces,
                    **configs["model"]["vae"]
                ).latent_dist.sample()

            noise = torch.randn_like(latents)
            u = compute_density_for_timestep_sampling(
                weighting_scheme=configs["train"]["weighting_scheme"],
                batch_size=num_objects,
                logit_mean=configs["train"]["logit_mean"],
                logit_std=configs["train"]["logit_std"],
                mode_scale=configs["train"]["mode_scale"],
            )
            indices = (u * noise_scheduler.config.num_train_timesteps).long()
            timesteps = noise_scheduler.timesteps[indices].to(accelerator.device)
            timesteps = timesteps.repeat_interleave(num_parts)

            sigmas = get_sigmas(timesteps, len(latents.shape), weight_dtype)
            latent_model_input = noisy_latents = (1. - sigmas) * latents + sigmas * noise
            
            if configs["train"]["cfg_dropout_prob"] > 0:
                dropout_mask = torch.rand(num_objects, device=accelerator.device) < configs["train"]["cfg_dropout_prob"]
                dropout_mask = dropout_mask.repeat_interleave(num_parts)
                if dropout_mask.any():
                    image_embeds[dropout_mask] = negative_image_embeds[dropout_mask]
                    if args.text_conditioning: text_embeds[dropout_mask] = negative_text_embeds[dropout_mask]

            # if not args.text_conditioning:
            #     image_embeds = condition_processor(image=image_embeds, text=None) # this can be used for both text and image
            # else:
            #     image_embeds = condition_processor(image=None, text=text_embeds)
            loss_contrastive, image_embeds = condition_processor(image=image_embeds, text=text_embeds, num_parts=num_parts)
            # print(loss_contrastive)
            model_pred = transformer(
                hidden_states=latent_model_input,
                timestep=timesteps,
                encoder_hidden_states=image_embeds,
                attention_kwargs={"num_parts": num_parts}
            ).sample

            if configs["train"]["training_objective"] == "x0":
                model_pred = model_pred * (-sigmas) + noisy_latents
                target = latents
            elif configs["train"]["training_objective"] == 'v':
                target = noise - latents
            elif configs["train"]["training_objective"] == '-v':
                # The training objective for TripoSG is the reverse of the flow matching objective. 
                # It uses "different directions", i.e., the negative velocity. 
                # This is probably a mistake in engineering, not very harmful. 
                # In TripoSG's rectified flow scheduler, prev_sample = sample + (sigma - sigma_next) * model_output
                # See TripoSG's scheduler https://github.com/VAST-AI-Research/TripoSG/blob/main/triposg/schedulers/scheduling_rectified_flow.py#L296
                # While in diffusers's flow matching scheduler, prev_sample = sample + (sigma_next - sigma) * model_output
                # See https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_flow_match_euler_discrete.py#L454
                target = latents - noise
            else:
                raise ValueError(f"Unknown training objective [{configs['train']['training_objective']}]")

            weighting = compute_loss_weighting_for_sd3(
                configs["train"]["weighting_scheme"],
                sigmas
            )

            loss = weighting * tF.mse_loss(model_pred.float(), target.float(), reduction="none")
            loss = loss.mean(dim=list(range(1, len(loss.shape))))
            loss = loss.mean()

            # accelerator.backward(loss.mean())
            if args.text_conditioning == "contrastive_text":
                loss = loss*0.2 + loss_contrastive*0.8
                # loss = loss + loss_contrastive
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                all_trainable_params = list(itertools.chain(condition_processor.parameters(), transformer.parameters()))
                accelerator.clip_grad_norm_(all_trainable_params, args.max_grad_norm)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        if accelerator.sync_gradients:
            loss = accelerator.gather(loss.detach()).mean()

            logs = {
                "loss": loss.item(),
                "lr": lr_scheduler.get_last_lr()[0]
            }
            if args.use_ema:
                ema_transformer.step(transformer.parameters())
                logs.update({"ema": ema_transformer.cur_decay_value})

            progress_bar.set_postfix(**logs)
            progress_bar.update(1)
            global_update_step += 1

            logger.info(
                f"[{global_update_step:06d} / {total_updated_steps:06d}] " +
                f"loss: {logs['loss']:.4f}, lr: {logs['lr']:.2e}" +
                f", ema: {logs['ema']:.4f}" if args.use_ema else ""
            )

            if (
                global_update_step % configs["train"]["log_freq"] == 0
                or global_update_step == 1
                or global_update_step % updated_steps_per_epoch == 0
            ):
                if accelerator.is_main_process and not args.no_wandb:
                    wandb.log({
                        "training/loss": logs["loss"],
                        "training/lr": logs["lr"],
                    }, step=global_update_step)
                    if args.use_ema:
                        wandb.log({
                            "training/ema": logs["ema"]
                        }, step=global_update_step)

            if (
                global_update_step % configs["train"]["save_freq"] == 0
                # or global_update_step % (configs["train"]["save_freq_epoch"] * updated_steps_per_epoch) == 0
                or global_update_step == total_updated_steps
            ):
                gc.collect()
                if accelerator.distributed_type == accelerate.utils.DistributedType.DEEPSPEED:
                    accelerator.save_state(os.path.join(ckpt_dir, f"{global_update_step:06d}"))
                elif accelerator.is_main_process:
                    accelerator.save_state(os.path.join(ckpt_dir, f"{global_update_step:06d}"))
                accelerator.wait_for_everyone()
                gc.collect()

            # Evaluate on the validation set
            if args.max_val_steps > 0 and (
                (global_update_step % configs["train"]["early_eval_freq"] == 0 and global_update_step < configs["train"]["early_eval"])  # 1. more frequently at the beginning
                or global_update_step % configs["train"]["eval_freq"] == 0  # 2. every `eval_freq` steps
                # or global_update_step % (configs["train"]["eval_freq_epoch"] * updated_steps_per_epoch) == 0  # 3. every `eval_freq_epoch` epochs
                or global_update_step == total_updated_steps # 4. last step of an epoch
                # or global_update_step == 1 # 5. first step
            ):  

                # Use EMA parameters for evaluation
                if args.use_ema:
                    # Store the Transformer parameters temporarily and load the EMA parameters to perform inference
                    ema_transformer.store(transformer.parameters())
                    ema_transformer.copy_to(transformer.parameters())

                transformer.eval()
                condition_processor.eval()

                log_validation(
                    val_loader, random_val_loader,
                    feature_extractor_dinov2, image_encoder_dinov2,
                    tokenizer, text_encoder,
                    vae, transformer, condition_processor,
                    global_update_step, eval_dir,
                    accelerator, logger,
                    args, configs
                )

                if args.use_ema:
                    # Switch back to the original Transformer parameters
                    ema_transformer.restore(transformer.parameters())

                torch.cuda.empty_cache()
                gc.collect()

def log_validation(
    dataloader, random_dataloader,
    feature_extractor_dinov2, image_encoder_dinov2,
    tokenizer, text_encoder,
    vae, transformer, condition_processor,
    global_step, eval_dir,
    accelerator, logger,  
    args, configs
):  

    
    val_noise_scheduler = RectifiedFlowScheduler.from_pretrained(
        configs["model"]["pretrained_model_name_or_path"],
        subfolder="scheduler"
    )

    pipeline = PartCrafterPipeline(
        vae=vae,
        transformer=accelerator.unwrap_model(transformer),
        condition_processor=accelerator.unwrap_model(condition_processor),
        scheduler=val_noise_scheduler,
        feature_extractor_dinov2=feature_extractor_dinov2,
        image_encoder_dinov2=image_encoder_dinov2,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
    )

    pipeline.set_progress_bar_config(disable=True)
    # pipeline.enable_xformers_memory_efficient_attention()

    if args.seed >= 0:
        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)
    else:
        generator = None
        

    val_progress_bar = tqdm(
        range(len(dataloader)) if args.max_val_steps is None else range(args.max_val_steps),
        desc=f"Validation [{global_step:06d}]",
        ncols=125,
        disable=not accelerator.is_main_process
    )

    medias_dictlist, metrics_dictlist = defaultdict(list), defaultdict(list)

    val_dataloder, random_val_dataloader = yield_forever(dataloader), yield_forever(random_dataloader)
    val_step = 0
    while val_step < args.max_val_steps:

        if val_step < args.max_val_steps // 2:
            # fix the first half
            batch = next(val_dataloder)
        else:
            # randomly sample the next batch
            batch = next(random_val_dataloader)
        
        images = batch["images"]
        if len(images.shape) == 5:
            images = images[0] # (1, N, H, W, 3) -> (N, H, W, 3)
        images = [Image.fromarray(image) for image in images.cpu().numpy()]
        N = len(images)
        captions = batch['captions']
        print(captions)
        assert (args.text_conditioning == "none") or (captions is not None), "Captions are required for text conditioning"
        if captions is not None:
            assert len(captions) == N, f"Number of captions {len(captions)} does not match number of images {N}"
        part_surfaces = batch["part_surfaces"].cpu().numpy()
        if len(part_surfaces.shape) == 4:
            part_surfaces = part_surfaces[0] # (1, N, P, 6) -> (N, P, 6)

        val_progress_bar.set_postfix(
            {"num_parts": N}
        )

        with torch.autocast("cuda", torch.float16):
            for guidance_scale in sorted(args.val_guidance_scales):
                pred_part_meshes = pipeline(
                    images,
                    captions,
                    num_inference_steps=configs['val']['num_inference_steps'],
                    num_tokens=configs['model']['vae']['num_tokens'],
                    guidance_scale=guidance_scale, 
                    attention_kwargs={"num_parts": N},
                    generator=generator,
                    max_num_expanded_coords=configs['val']['max_num_expanded_coords'],
                    use_flash_decoder=configs['val']['use_flash_decoder'],
                ).meshes

                # Save the generated meshes
                if accelerator.is_main_process:
                    local_eval_dir = os.path.join(eval_dir, f"{global_step:06d}", f"guidance_scale_{guidance_scale:.1f}")
                    os.makedirs(local_eval_dir, exist_ok=True)
                    rendered_images_list, rendered_normals_list = [], []
                    # 1. save the gt
                    images[0].save(os.path.join(local_eval_dir, f"{val_step:04d}.png"))
                    if captions is not None:
                        with open(os.path.join(local_eval_dir, f"{val_step:04d}_caption.txt"), "w") as f:
                            f.write(batch["captions"][0][0])
                    # 2. save the generated part meshes
                    for n in range(N):
                        if pred_part_meshes[n] is None:
                            # If the generated mesh is None (decoing error), use a dummy mesh
                            pred_part_meshes[n] = trimesh.Trimesh(vertices=[[0, 0, 0]], faces=[[0, 0, 0]])
                        pred_part_meshes[n].export(os.path.join(local_eval_dir, f"{val_step:04d}_{n:02d}.glb"))
                    # 3. render the generated mesh and save the rendered images
                    pred_mesh = get_colored_mesh_composition(pred_part_meshes)
                    rendered_images: List[Image.Image] = render_views_around_mesh(
                        pred_mesh, 
                        num_views=configs['val']['rendering']['num_views'],
                        radius=configs['val']['rendering']['radius'],
                    )
                    rendered_normals: List[Image.Image] = render_normal_views_around_mesh(
                        pred_mesh,
                        num_views=configs['val']['rendering']['num_views'],
                        radius=configs['val']['rendering']['radius'],
                    )
                    export_renderings(
                        rendered_images,
                        os.path.join(local_eval_dir, f"{val_step:04d}.gif"),
                        fps=configs['val']['rendering']['fps']
                    )
                    export_renderings(
                        rendered_normals,
                        os.path.join(local_eval_dir, f"{val_step:04d}_normals.gif"),
                        fps=configs['val']['rendering']['fps']
                    )
                    rendered_images_list.append(rendered_images)
                    rendered_normals_list.append(rendered_normals)
                    
                    if images is not None:
                        medias_dictlist[f"guidance_scale_{guidance_scale:.1f}/gt_image"] += [images[0]] # List[Image.Image] TODO: support batch size > 1
                    medias_dictlist[f"guidance_scale_{guidance_scale:.1f}/pred_rendered_images"] += rendered_images_list # List[List[Image.Image]]
                    medias_dictlist[f"guidance_scale_{guidance_scale:.1f}/pred_rendered_normals"] += rendered_normals_list # List[List[Image.Image]]

                ################################ Compute generation metrics ################################

                parts_chamfer_distances, parts_f_scores = [], []

                for n in range(N):
                    gt_part_surface = part_surfaces[n]
                    pred_part_mesh = pred_part_meshes[n]
                    if pred_part_mesh is None:
                        # If the generated mesh is None (decoing error), use a dummy mesh
                        assert False, "Mesh is None"
                        # pred_part_mesh = trimesh.Trimesh(vertices=[[0, 0, 0]], faces=[[0, 0, 0]])
                    part_cd, part_f = compute_cd_and_f_score_in_training(
                        gt_part_surface, pred_part_mesh,
                        num_samples=configs['val']['metric']['cd_num_samples'],
                        threshold=configs['val']['metric']['f1_score_threshold'],
                        metric=configs['val']['metric']['cd_metric']
                    )
                    # avoid nan
                    part_cd = configs['val']['metric']['default_cd'] if np.isnan(part_cd) else part_cd
                    part_f = configs['val']['metric']['default_f1'] if np.isnan(part_f) else part_f
                    parts_chamfer_distances.append(part_cd)
                    parts_f_scores.append(part_f)

                    # TODO: Fix this
                    # Disable chamfer distance and F1 score for now
                    # parts_chamfer_distances.append(0.0)
                    # parts_f_scores.append(0.0)

                parts_chamfer_distances = torch.tensor(parts_chamfer_distances, device=accelerator.device)
                parts_f_scores = torch.tensor(parts_f_scores, device=accelerator.device)

                metrics_dictlist[f"parts_chamfer_distance_cfg{guidance_scale:.1f}"].append(parts_chamfer_distances.mean())
                metrics_dictlist[f"parts_f_score_cfg{guidance_scale:.1f}"].append(parts_f_scores.mean())
            
        # Only log the last (biggest) cfg metrics in the progress bar
        val_logs = {
            "parts_chamfer_distance": parts_chamfer_distances.mean().item(),
            "parts_f_score": parts_f_scores.mean().item(),
        }
        val_progress_bar.set_postfix(**val_logs)
        logger.info(
            f"Validation [{val_step:02d}/{args.max_val_steps:02d}] " +
            f"parts_chamfer_distance: {val_logs['parts_chamfer_distance']:.4f}, parts_f_score: {val_logs['parts_f_score']:.4f}"
        )
        logger.info(
            f"parts_chamfer_distances: {[f'{x:.4f}' for x in parts_chamfer_distances.tolist()]}"
        )
        logger.info(
            f"parts_f_scores: {[f'{x:.4f}' for x in parts_f_scores.tolist()]}"
        )
        val_step += 1
        val_progress_bar.update(1)

    val_progress_bar.close()

    if accelerator.is_main_process:
        for key, value in medias_dictlist.items():
            if isinstance(value[0], Image.Image): # assuming gt_image
                image_grid = make_grid_for_images_or_videos(
                    value, 
                    nrow=configs['val']['nrow'],
                    return_type='pil', 
                )
                image_grid.save(os.path.join(eval_dir, f"{global_step:06d}", f"{key}.png"))
                if not args.no_wandb:
                    wandb.log({f"validation/{key}": wandb.Image(image_grid)}, step=global_step)
            else: # assuming pred_rendered_images or pred_rendered_normals
                image_grids = make_grid_for_images_or_videos(
                    value, 
                    nrow=configs['val']['nrow'],
                    return_type='ndarray',
                )
                if not args.no_wandb:
                    wandb.log({
                        f"validation/{key}": wandb.Video(
                            image_grids, 
                            fps=configs['val']['rendering']['fps'], 
                            format="gif"
                    )}, step=global_step)
                image_grids = [Image.fromarray(image_grid.transpose(1, 2, 0)) for image_grid in image_grids]
                export_renderings(
                    image_grids, 
                    os.path.join(eval_dir, f"{global_step:06d}", f"{key}.gif"), 
                    fps=configs['val']['rendering']['fps']
                )

        for k, v in metrics_dictlist.items():
            if not args.no_wandb:
                wandb.log({f"validation/{k}": torch.tensor(v).mean().item()}, step=global_step)



if __name__ == "__main__":
    main()