import torch
import argparse
from src.models.transformers.partcrafter_transformer import PartCrafterDiTModel
from src.utils.point4bit import Point4BitLinear


def replace_linear_with_point4bit(module, fg_ratio, weight_ratio, high_bits, low_bits, calibrated_scales, layer_name_prefix=""):
    """
    Recursively replace nn.Linear layers with Point4BitLinear in the given module.
    Copies weights/biases and applies calibration scales.
    """
    for name, child in module.named_children():
        full_name = f"{layer_name_prefix}.{name}" if layer_name_prefix else name
        if isinstance(child, torch.nn.Linear):
            # Create Point4BitLinear with same dimensions
            point4bit_layer = Point4BitLinear(
                child.in_features,
                child.out_features,
                fg_ratio=fg_ratio,
                weight_ratio=weight_ratio,
                high_bits=high_bits,
                low_bits=low_bits,
                bias=child.bias is not None
            )
            # Copy weights and biases
            point4bit_layer.weight.data = child.weight.data.clone()
            if child.bias is not None:
                point4bit_layer.bias.data = child.bias.data.clone()
            # Apply calibration scales if available
            if full_name in calibrated_scales:
                scale = calibrated_scales[full_name]
                point4bit_layer.weight.data *= scale
            # Replace the layer
            setattr(module, name, point4bit_layer)
            print(f"Replaced {full_name} with Point4BitLinear")
        else:
            # Recurse into child modules
            replace_linear_with_point4bit(child, fg_ratio, weight_ratio, high_bits, low_bits, calibrated_scales, full_name)


def main():
    parser = argparse.ArgumentParser(description="Setup Point4Bit quantization for PartCrafter model.")
    parser.add_argument('--config', type=str, help="Path to config file (not used in this simplified version).")
    parser.add_argument('--m1_fg_ratio', type=float, default=0.2, help="Ratio of foreground tokens.")
    parser.add_argument('--m2_weight_ratio', type=float, default=0.8, help="Ratio of high-bit weight channels.")
    parser.add_argument('--high_bits', type=int, default=4, help="High bit width.")
    parser.add_argument('--low_bits', type=int, default=4, help="Low bit width.")
    parser.add_argument('--calib_path', type=str, default='calibration_output/calibrated_scales.pt', help="Path to calibrated scales.")
    parser.add_argument('--pretrained_dir', type=str, default='pretrained_weights/PartCrafter', help="Directory of pretrained model.")
    parser.add_argument('--output_dir', type=str, default='pretrained_weights/PartCrafter_point4bit', help="Output directory for modified model.")
    args = parser.parse_args()

    # Load pretrained transformer
    print(f"Loading pretrained model from {args.pretrained_dir}")
    transformer = PartCrafterDiTModel.from_pretrained(args.pretrained_dir, subfolder="transformer")

    # Load calibrated scales
    print(f"Loading calibrated scales from {args.calib_path}")
    calibrated_scales = torch.load(args.calib_path, map_location='cpu')

    # Replace Linear layers in joint_blocks with Point4BitLinear
    if hasattr(transformer, 'joint_blocks'):
        print("Replacing nn.Linear layers in joint_blocks with Point4BitLinear")
        for i, block in enumerate(transformer.joint_blocks):
            replace_linear_with_point4bit(
                block,
                args.m1_fg_ratio,
                args.m2_weight_ratio,
                args.high_bits,
                args.low_bits,
                calibrated_scales,
                layer_name_prefix=f"joint_blocks.{i}"
            )
    else:
        print("Warning: No joint_blocks found in transformer")

    # Save the modified transformer
    print(f"Saving modified model to {args.output_dir}")
    transformer.save_pretrained(args.output_dir)

    print("Setup complete. Note: FA-PAQ CDF intervals triggered by V-shape and Curvature spikes during t=20-30 window need to be implemented in Point4BitLinear.forward based on runtime signals.")


if __name__ == "__main__":
    main()