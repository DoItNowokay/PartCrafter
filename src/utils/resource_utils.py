import psutil
import GPUtil
import torch
import logging
from typing import Dict, Any


def get_cpu_stats() -> Dict[str, Any]:
    """Get CPU-related statistics."""
    cpu_percent = psutil.cpu_percent(interval=0.1)
    memory = psutil.virtual_memory()
    return {
        "cpu_percent": cpu_percent,
        "memory_percent": memory.percent,
        "memory_used_gb": memory.used / (1024**3),
        "memory_available_gb": memory.available / (1024**3),
    }


def get_gpu_stats() -> Dict[str, Any]:
    """Get GPU-related statistics using GPUtil and PyTorch."""
    stats = {}
    if torch.cuda.is_available():
        stats["gpu_memory_allocated_gb"] = torch.cuda.memory_allocated() / (1024**3)
        stats["gpu_memory_reserved_gb"] = torch.cuda.memory_reserved() / (1024**3)
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]  # Assuming single GPU, or take first
                stats["gpu_util_percent"] = gpu.load * 100
                stats["gpu_memory_used_mb"] = gpu.memoryUsed
                stats["gpu_memory_total_mb"] = gpu.memoryTotal
                stats["gpu_temperature_c"] = gpu.temperature
        except Exception as e:
            stats["gpu_error"] = str(e)
    else:
        stats["gpu_available"] = False
    return stats


def log_resource_usage(logger: logging.Logger, args, step: int, prefix: str = "") -> Dict[str, Any]:
    """Log CPU and GPU usage to logger and WandB if enabled. Returns the stats dict."""
    cpu_stats = get_cpu_stats()
    gpu_stats = get_gpu_stats()
    
    log_msg = f"{prefix} | Step: {step:04d} | CPU: {cpu_stats['cpu_percent']:.1f}% | Mem: {cpu_stats['memory_used_gb']:.2f}GB ({cpu_stats['memory_percent']:.1f}%)"
    if gpu_stats.get("gpu_available", True):
        log_msg += f" | GPU Mem Alloc: {gpu_stats.get('gpu_memory_allocated_gb', 0):.2f}GB | GPU Util: {gpu_stats.get('gpu_util_percent', 0):.1f}%"
    else:
        log_msg += " | No GPU"
    
    logger.info(log_msg)
    
    if not getattr(args, 'no_wandb', True):
        import wandb
        wandb_stats = {f"resources/{prefix}_{k}": v for k, v in {**cpu_stats, **gpu_stats}.items() if isinstance(v, (int, float))}
        wandb.log(wandb_stats, step=step)
    
    # Return combined stats for structured data
    return {"step": step, "prefix": prefix, **cpu_stats, **gpu_stats}