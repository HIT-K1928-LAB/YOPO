#!/usr/bin/env python3
"""
Export YOPO forward(depth, obs) to ONNX.

This keeps the existing depth preprocessing and obs construction outside
the ONNX graph and exports a fixed batch-1 interface:
  depth:    [1, 1, 96, 160]
  obs:      [1, 9, V, H]
  endstate: [1, 9, V, H]
  score:    [1, V, H]
"""

import argparse
import os

import torch

from config.config import cfg
from policy.yopo_network import YopoNetwork


def parse_args() -> argparse.Namespace:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description="Export YOPO checkpoint to ONNX.")
    parser.add_argument("--trial", type=int, default=1, help="Trial number for saved/YOPO_{trial}/epoch{epoch}.pth")
    parser.add_argument("--epoch", type=int, default=50, help="Epoch number for saved/YOPO_{trial}/epoch{epoch}.pth")
    parser.add_argument("--weight", type=str, default="", help="Optional explicit .pth weight path.")
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join(base_dir, "model", "yopo.onnx"),
        help="Output ONNX path.",
    )
    parser.add_argument("--opset", type=int, default=13, help="ONNX opset version.")
    parser.add_argument("--check", action="store_true", help="Run onnx.checker after export.")
    return parser.parse_args()


def resolve_weight_path(args: argparse.Namespace) -> str:
    if args.weight:
        return os.path.abspath(args.weight)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, "saved", f"YOPO_{args.trial}", f"epoch{args.epoch}.pth")


def load_state_dict(weight_path: str, device: torch.device):
    try:
        return torch.load(weight_path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(weight_path, map_location=device)


def print_io_shapes() -> None:
    print(f"depth    : [1, 1, {cfg['image_height']}, {cfg['image_width']}]")
    print(f"obs      : [1, 9, {cfg['vertical_num']}, {cfg['horizon_num']}]")
    print(f"endstate : [1, 9, {cfg['vertical_num']}, {cfg['horizon_num']}]")
    print(f"score    : [1, {cfg['vertical_num']}, {cfg['horizon_num']}]")


def main() -> None:
    args = parse_args()
    weight_path = resolve_weight_path(args)
    output_path = os.path.abspath(args.output)

    if not os.path.isfile(weight_path):
        raise FileNotFoundError(f"Weight file not found: {weight_path}")

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = YopoNetwork().to(device)
    model.load_state_dict(load_state_dict(weight_path, device))
    model.eval()

    depth = torch.zeros((1, 1, cfg["image_height"], cfg["image_width"]), dtype=torch.float32, device=device)
    obs = torch.zeros((1, 9, cfg["vertical_num"], cfg["horizon_num"]), dtype=torch.float32, device=device)

    print("Exporting YOPO ONNX...")
    print(f"Weight: {weight_path}")
    print(f"Output: {output_path}")
    print(f"Device: {device}")
    print(f"Opset : {args.opset}")
    print_io_shapes()

    with torch.inference_mode():
        torch.onnx.export(
            model,
            (depth, obs),
            output_path,
            input_names=["depth", "obs"],
            output_names=["endstate", "score"],
            opset_version=args.opset,
            do_constant_folding=True,
        )

    print(f"Exported YOPO ONNX: {output_path}")

    if args.check:
        import onnx

        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("onnx.checker passed.")


if __name__ == "__main__":
    main()
