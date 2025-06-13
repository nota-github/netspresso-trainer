import argparse
import os
from pathlib import Path

import torch
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir import to_edge_transform_and_lower

import torch._dynamo
torch._dynamo.config.suppress_errors = True



def parse_args():
    parser = argparse.ArgumentParser(description="Parser for XNNPACK lowering")
    parser.add_argument("--model-path", type=str, required=True, help="Model path")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    parser.add_argument("--sample-size", type=int, nargs=2, default=(640, 640), help="Sample size")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    a = torch.export.load(args.model_path)

    example_inputs = (torch.randn(1, 3, *args.sample_size),)

    et_program = to_edge_transform_and_lower(
        a,
        partitioner=[XnnpackPartitioner()]
    ).to_executorch()

    output_path = Path(args.output_dir) / f"{Path(args.model_path).stem}-xnn-lower.pte"

    with open(output_path, "wb") as f:
        f.write(et_program.buffer)

    print(f"XNNPACK lowering completed! Saved to {output_path}")
