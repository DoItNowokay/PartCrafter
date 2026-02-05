import psutil
import torch
try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    print("GPUtil not available. GPU metrics will be skipped.")

import time
from typing import Dict, Any


def get_system_metrics() -> Dict[str, Any]:
    """
    Collects CPU, memory, and GPU metrics.
    Returns a dictionary with structured data for logging/visualization.
    """
    metrics = {
        "timestamp": time.time(),
    }

    # CPU Metrics
    metrics["cpu_percent"] = psutil.cpu_percent(interval=None)  # Overall CPU usage %
    metrics["cpu_per_core"] = psutil.cpu_percent(percpu=True)  # List of % per core

    # Memory Metrics
    memory = psutil.virtual_memory()
    metrics["memory_percent"] = memory.percent
    metrics["memory_used_gb"] = memory.used / (1024**3)
    metrics["memory_available_gb"] = memory.available / (1024**3)
    metrics["memory_total_gb"] = memory.total / (1024**3)

    # GPU Metrics (if available)
    if GPU_AVAILABLE and torch.cuda.is_available():
        gpus = GPUtil.getGPUs()
        gpu_metrics = []
        for i, gpu in enumerate(gpus):
            gpu_data = {
                "gpu_id": i,
                "gpu_name": gpu.name,
                "gpu_memory_used_mb": gpu.memoryUsed,
                "gpu_memory_total_mb": gpu.memoryTotal,
                "gpu_memory_percent": (gpu.memoryUsed / gpu.memoryTotal) * 100 if gpu.memoryTotal > 0 else 0,
                "gpu_utilization_percent": gpu.load * 100,
                "gpu_temperature_c": gpu.temperature,
            }
            gpu_metrics.append(gpu_data)
        metrics["gpu"] = gpu_metrics

        # PyTorch CUDA memory
        metrics["cuda_memory_allocated_gb"] = torch.cuda.memory_allocated() / (1024**3)
        metrics["cuda_memory_reserved_gb"] = torch.cuda.memory_reserved() / (1024**3)
    else:
        metrics["gpu"] = None

    return metrics