import os
import sys
import time
import math
import logging
from packaging import version
import wandb

import torch
import argparse
import accelerate
from accelerate import Accelerator, DataLoaderConfiguration, DeepSpeedPlugin
from accelerate.logging import get_logger as get_accelerate_logger
from tqdm import tqdm

from transformers import (
    BitImageProcessor,
    Dinov2Model,
    CLIPTextModel,
    CLIPTokenizer,
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.train_utils import get_configs, get_optimizer, get_lr_scheduler
from src.models.condition_processor import ConditionProcessor
from src.datasets import BatchedObjaverseCaptionDataset, MultiEpochsDataLoader


def main():
    parser = argparse.ArgumentParser(
        description="Train a diffusion model for 3D object generation (text-only variant)",
    )

    # Reuse same CLI as train_partcrafter for parity
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
        choices=[1, 2, 3],
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
        default="contrastive_text",
        choices=["none", "direct_text", "contrastive_text", "adaln_text", "contrastive_text_michelangelo", "contrastive_text_pooled"],
        help="Whether to use text conditioning and which type"
    )
    parser.add_argument(
        "--editing",
        type=bool,
        default=False,
        help="Whether to perform editing"
    )
    parser.add_argument(
        "--train_text_encoder",
        action="store_true",
        default=False,
        help="Whether to train the text encoder"
    )
    parser.add_argument("--tag_suffix", type=str, default="")
    args, extras = parser.parse_known_args()

    configs = get_configs(args.config, extras)

    if args.text_conditioning == "none":
        raise ValueError("This script requires text conditioning (contrastive).")

    if args.tag is None:
        args.tag = time.strftime("%Y%m%d_%H_%M_%S")
    if args.tag_suffix:
        args.tag = f"{args.tag}_{args.tag_suffix}"

    exp_dir = os.path.join(args.output_dir, args.tag)
    ckpt_dir = os.path.join(exp_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y/%m/%d %H:%M:%S", level=logging.INFO)
    logger = get_accelerate_logger(__name__, log_level="INFO")

    # Accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        split_batches=False,
        dataloader_config=DataLoaderConfiguration(non_blocking=args.pin_memory),
    )

    if args.seed >= 0:
        accelerate.utils.set_seed(args.seed)

    # register save/load hooks so accelerator.save_state/load_state will persist the
    # condition_processor and clip_text_model subfolders in the same way as train_partcrafter
    def _subfolder_for_model(model):
        name = type(model).__name__.lower()
        if "partcrafterdit" in name or "transformer" in name:
            return "transformer"
        if "conditionprocessor" in name or "condition_processor" in name:
            return "condition_processor"
        if "cliptextmodel" in name:
            return "cliptextmodel"
        return name

    def save_model_hook(models, weights, output_dir):
        if not accelerator.is_main_process:
            return
        # Save each model passed to the hook
        for model in models:
            subfolder = _subfolder_for_model(model)
            try:
                model.save_pretrained(os.path.join(output_dir, subfolder))
            except Exception as e:
                logger.error(f"Failed to save model {type(model).__name__} to {subfolder}: {e}")
            if weights:
                try:
                    weights.pop()
                except Exception:
                    pass

        # ensure condition_processor and text_encoder are also saved even if not in models
        # try:
        #     _ = condition_processor
        #     try:
        #         cp = accelerator.unwrap_model(condition_processor)
        #     except Exception:
        #         cp = condition_processor
        #     cp.save_pretrained(os.path.join(output_dir, "condition_processor"))
        # except Exception:
        #     pass

        # try:
        #     _ = text_encoder
        #     try:
        #         te = accelerator.unwrap_model(text_encoder)
        #     except Exception:
        #         te = text_encoder
        #     te.save_pretrained(os.path.join(output_dir, "clip_text_model"))
        # except Exception:
        #     pass

    def load_model_hook(models, input_dir):
        # load remaining models from their respective subfolders
        for _ in range(len(models)):
            model = models.pop()
            subfolder = _subfolder_for_model(model)
            model_path = os.path.join(input_dir, subfolder)
            if not os.path.isdir(model_path):
                logger.warning(f"No saved weights at {model_path} for model {type(model).__name__}; skipping.")
                continue
            try:
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

                if type(model).__name__.lower().startswith("cliptextmodel") or "cliptextmodel" in subfolder:
                    try:
                        loaded = CLIPTextModel.from_pretrained(input_dir, subfolder=subfolder)
                        if hasattr(model, "register_to_config") and hasattr(loaded, "config"):
                            model.register_to_config(**loaded.config)
                        model.load_state_dict(loaded.state_dict())
                        del loaded
                        continue
                    except Exception:
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

    # Models and processors
    logger.info("Loading models and processors...")
    feature_extractor = BitImageProcessor.from_pretrained(configs["model"]["pretrained_model_name_or_path"], subfolder="feature_extractor_dinov2")
    image_encoder = Dinov2Model.from_pretrained(configs["model"]["pretrained_model_name_or_path"], subfolder="image_encoder_dinov2")

    tokenizer = CLIPTokenizer.from_pretrained(configs["model"]["text_encoder_name"]) if configs["model"].get("text_encoder_name") else None
    text_encoder = CLIPTextModel.from_pretrained(configs["model"]["text_encoder_name"]) if tokenizer is not None else None

    # Condition processor (we'll train it)
    condition_processor = ConditionProcessor.from_config(
        os.path.join(configs["model"]["pretrained_model_name_or_path"], "condition_processor"),
        text_conditioning=args.text_conditioning
    )

    # Dataset: caption dataset
    train_dataset = BatchedObjaverseCaptionDataset(
        configs=configs,
        batch_size=configs["train"]["batch_size_per_gpu"],
        is_main_process=accelerator.is_main_process,
        shuffle=True,
        training=True,
    )

    train_loader = MultiEpochsDataLoader(
        train_dataset,
        batch_size=configs["train"]["batch_size_per_gpu"],
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=args.pin_memory,
        collate_fn=train_dataset.collate_fn,
    )

    # Only train condition_processor and text_encoder
    params = []
    if condition_processor is not None:
        condition_processor.requires_grad_(True)
        params += [p for p in condition_processor.parameters() if p.requires_grad]

    if args.train_text_encoder and text_encoder is not None:
        text_encoder.requires_grad_(True)
        params += [p for p in text_encoder.parameters() if p.requires_grad]
    else:
        if text_encoder is not None:
            text_encoder.requires_grad_(False)
            text_encoder.eval()

    # Optimizer and scheduler
    opt_cfg = configs.get("optimizer", {})
    opt_name = opt_cfg.get("name", "adamw")
    # remove name from kwargs passed to get_optimizer
    opt_kwargs = {k: v for k, v in opt_cfg.items() if k != "name"}
    optimizer = get_optimizer(opt_name, params=params, **opt_kwargs)

    # LR scheduler (optional) - construct with a sensible total_steps
    lr_cfg = configs.get("lr_scheduler", None)
    lr_scheduler = None
    if lr_cfg is not None and "name" in lr_cfg:
        lr_name = lr_cfg.get("name")
        lr_kwargs = {k: v for k, v in lr_cfg.items() if k != "name"}
        # compute a default total_steps if not provided
        total_steps = lr_kwargs.get("total_steps", configs.get("lr_scheduler", {}).get("total_steps", len(train_loader) * configs["train"].get("epochs", 1)))
        lr_kwargs["total_steps"] = total_steps
        try:
            lr_scheduler = get_lr_scheduler(lr_name, optimizer=optimizer, **lr_kwargs)
        except Exception:
            lr_scheduler = None

    # Prepare with accelerator
    to_prepare = [condition_processor, text_encoder, optimizer, train_loader]
    to_prepare = [x for x in to_prepare if x is not None]
    # include lr_scheduler in prepare if present
    if lr_scheduler is not None:
        to_prepare.append(lr_scheduler)

    prepared = accelerator.prepare(*to_prepare)
    pi = 0
    condition_processor = prepared[pi]; pi += 1
    if text_encoder is not None:
        text_encoder = prepared[pi]; pi += 1
    optimizer = prepared[pi]; pi += 1
    train_loader = prepared[pi]; pi += 1
    if lr_scheduler is not None:
        lr_scheduler = prepared[pi]

    # handle resume from checkpoint like train_partcrafter
    global_step = 0
    if args.resume_from_iter is not None:
        if args.resume_from_iter < 0:
            args.resume_from_iter = int(sorted(os.listdir(ckpt_dir))[-1])
        logger.info(f"Load checkpoint from iteration [{args.resume_from_iter}]\n")
        accelerator.load_state(os.path.join(ckpt_dir, f"{args.resume_from_iter:06d}"))
        global_step = int(args.resume_from_iter)

    # Move image encoder and feature extractor to device (we won't train them)
    device = accelerator.device
    # image_encoder.to(device)
    # feature_extractor.to(device)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    image_encoder.to(device, dtype=weight_dtype)

    # Training loop
    logger.info("Starting contrastive training for condition_processor and text_encoder...")
    # compute total steps fallback
    total_steps = configs.get("lr_scheduler", {}).get("total_steps", len(train_loader) * configs["train"].get("epochs", 1))
    if args.max_train_steps is None:
        args.max_train_steps = total_steps

    # show training summary similar to train_partcrafter
    total_batch_size = configs["train"]["batch_size_per_gpu"] * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info(f"Total batch size: [{total_batch_size}]")
    logger.info(f"Learning rate (base): [{opt_cfg.get('lr', opt_kwargs.get('lr', None))}]")
    logger.info(f"Gradient Accumulation steps: [{args.gradient_accumulation_steps}]")

    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_main_process, initial=global_step)
    temperature = configs.get("train", {}).get("contrastive_temperature", 0.07)

    while global_step < args.max_train_steps:
        for batch in train_loader:
            if global_step >= args.max_train_steps:
                break

            texts = batch["captions"]
            images = batch["images"]

            # image preprocessing
            with torch.no_grad():
                images_proc = feature_extractor(images=images, return_tensors="pt").pixel_values
            images_proc = images_proc.to(device=device, dtype=weight_dtype)
            with torch.no_grad():
                dino_out = image_encoder(images_proc)
                image_embeds = dino_out.last_hidden_state
                image_pooled = dino_out.pooler_output

            # text processing
            text_inputs = tokenizer(texts, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
            text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
            clip_out = text_encoder(**text_inputs)
            text_embeds = clip_out.last_hidden_state
            text_pooled = clip_out.pooler_output

            # Use ConditionProcessor's contrastive loss (same as train_partcrafter)
            # batch provides num_parts (sum over items equals batch size)
            num_parts = batch.get("num_parts", None)
            if args.text_conditioning == "adaln_text":
                text_embeds = (text_embeds, text_pooled)
            loss_contrastive, _ = condition_processor(
                image=image_embeds,
                text=text_embeds,
                image_pooled=image_pooled,
                text_pooled=text_pooled,
                num_parts=num_parts,
            )
            loss = loss_contrastive

            accelerator.backward(loss)
            optimizer.step()
            # step lr scheduler if present (per update)
            try:
                if lr_scheduler is not None:
                    lr_scheduler.step()
            except Exception:
                pass
            optimizer.zero_grad()

            global_step += 1
            progress_bar.update(1)
            # include current lr in progress bar
            try:
                cur_lr = optimizer.param_groups[0]["lr"]
            except Exception:
                cur_lr = 0.0
            progress_bar.set_description(f"loss={loss.item():.4f}, lr={cur_lr:.3e}")

            # Save checkpoints every N steps (use accelerator.save_state so optimizer/scheduler are saved too)
            if global_step % configs["train"].get("save_every_steps", 1000) == 0:
                step_dir = os.path.join(ckpt_dir, f"{global_step:06d}")
                os.makedirs(step_dir, exist_ok=True)
                try:
                    # accelerator.save_state will call our save_model_hook to persist model subfolders
                    if accelerator.is_main_process or accelerator.distributed_type == accelerate.utils.DistributedType.DEEPSPEED:
                        accelerator.save_state(step_dir)
                except Exception as e:
                    logger.error(f"Failed to save state at {step_dir}: {e}")

    # final save (use accelerator to save full state)
    final_dir = os.path.join(ckpt_dir, f"{global_step:06d}")
    os.makedirs(final_dir, exist_ok=True)
    try:
        if accelerator.is_main_process or accelerator.distributed_type == accelerate.utils.DistributedType.DEEPSPEED:
            accelerator.save_state(final_dir)
    except Exception as e:
        logger.error(f"Failed to save final state at {final_dir}: {e}")


if __name__ == "__main__":
    main()
