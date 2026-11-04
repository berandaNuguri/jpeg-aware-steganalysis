from jpeg_aware_steganalysis._paths import project_root, RESOURCE_ROOT
import os
import sys
import time
import random
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.nn.functional as F
from torchvision.models import efficientnet_b5

_THIS_FILE = Path(__file__).resolve()                           # .../root/utils/statistic/net_analysis.py
_PROJECT_ROOT = project_root()                           # .../root
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
    
from jpeg_aware_steganalysis.models.srnet.srnet_color import SRNet_Color
from jpeg_aware_steganalysis.models.zhunet.zhunet import ZhuNet
from jpeg_aware_steganalysis.models.ucnet.ucnet import UCNet    
from jpeg_aware_steganalysis.models.penet.penet_arbitrary import PENet_Arbitrary
from jpeg_aware_steganalysis.models.tempnet.tempnet_v53 import TempNet_v53
from jpeg_aware_steganalysis.models.sf3net.sf3net import SF3Net

def to_millions(x: int) -> float:
    return x / 1e6

def to_giga(x: int) -> float:
    return x / 1e9

# 소수점 자릿수는 보기 좋게: M은 2자리, G는 3자리
def format_params_m(n_params: int) -> str:
    return f"{to_millions(n_params):.2f} M"

def format_giga(n: int, label: str) -> str:
    return f"{to_giga(n):.3f} G{label}"

def params_to_fp32_mb(n_params: int) -> float:
    return n_params * 4 / 1e6
    
def setup_environment(args):
    if args.gpus != '':
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    print(f'Using GPU: {args.gpus}')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # reproducibility
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    mp.set_start_method('spawn', force=True)

    return device


def count_params(model: nn.Module):
    return sum(p.numel() for p in model.parameters())


def make_inputs(input_shapes, device):
    if isinstance(input_shapes[0], int):
        input_shapes = [tuple(input_shapes)]
    return tuple(torch.randn(*shape, device=device) for shape in input_shapes)


def profile_with_fvcore(model, input_shapes, device):
    m = model.to(device).eval()
    inputs = make_inputs(input_shapes, device)

    total_flops = 0
    orig_conv2d = F.conv2d
    orig_linear = F.linear

    def counted_conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
        nonlocal total_flops
        output = orig_conv2d(input, weight, bias, stride, padding, dilation, groups)
        kernel_mul = weight.shape[1] * weight.shape[2] * weight.shape[3]
        total_flops += int(output.numel() * kernel_mul * 2)
        return output

    def counted_linear(input, weight, bias=None):
        nonlocal total_flops
        output = orig_linear(input, weight, bias)
        total_flops += int(output.numel() * weight.shape[1] * 2)
        return output

    F.conv2d = counted_conv2d
    F.linear = counted_linear
    with torch.no_grad():
        try:
            if len(inputs) == 1:
                _ = m(inputs[0])
            else:
                _ = m(*inputs)
        finally:
            F.conv2d = orig_conv2d
            F.linear = orig_linear
    flops = int(total_flops)

    # MACs ≈ FLOPs / 2
    macs = flops // 2
    return macs, flops


def benchmark_inference(model, device, batch_size=8, runs=1000, warmup=50):
    model = model.to(device).eval()
    x = torch.randn(batch_size, 3, 256, 256, device=device)

    with torch.no_grad():
        # warmup
        for _ in range(warmup):
            _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()

        # timed runs
        t0 = time.perf_counter()
        for _ in range(runs):
            _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

    total = t1 - t0
    avg_ms = (total / runs) * 1000
    ips = (batch_size * runs) / total
    return avg_ms, ips


def build_model_suite():
    return [
        ("SRNet", SRNet_Color(), (1, 3, 256, 256)),
        ("EfficientNet B5", efficientnet_b5(weights=None, num_classes=2), (1, 3, 256, 256)),
        ("ZhuNet", ZhuNet(), (1, 3, 256, 256)),
        ("UCNet", UCNet(), (1, 3, 256, 256)),
        ("PENet-Arbitrary", PENet_Arbitrary(), (1, 3, 256, 256)),
        (
            "SF3Net",
            SF3Net(in_channels=3),
            ((1, 3, 256, 256), (1, 3, 256, 256), (1, 3, 8, 8)),
        ),
        ("RWNet", TempNet_v53(), (1, 3, 256, 256)),
    ]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', default='3', type=str)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--runs', default=1000, type=int)
    parser.add_argument('--warmup', default=50, type=int)
    parser.add_argument('--batch-size', default=8, type=int)
    parser.add_argument('--suite', action='store_true',
                        help='Profile the manuscript model-complexity suite.')
    parser.add_argument('--no-latency', action='store_true',
                        help='Skip latency/throughput benchmark.')
    return parser.parse_args()


def main():
    args = parse_args()
    device = setup_environment(args)

    if args.suite:
        print("\n========== Model Complexity Suite ==========")
        print("Input resolution: 256x256x3, batch=1")
        print("SF3Net: in_channels=3, forward(input_x, input_dct, input_qtable)")
        print("--------------------------------------------")
        print(f"{'Model':<18} {'Params (M)':>12} {'FP32 MB':>10} {'MACs (G)':>12} {'FLOPs (G)':>12}")
        print("--------------------------------------------")
        for name, model, input_shapes in build_model_suite():
            n_params = count_params(model)
            macs, flops = profile_with_fvcore(model, input_shapes, device)
            print(f"{name:<18} {to_millions(n_params):>12.2f} {params_to_fp32_mb(n_params):>10.2f} {to_giga(macs):>12.3f} {to_giga(flops):>12.3f}")
        print("============================================\n")
        return

    # model = UCNet()
    # model = TempNet()
    # model = TempNet_v53()
    # model = PENet_Arbitrary()
    model = efficientnet_b5(weights=None, num_classes=2)
    n_params = count_params(model)

    # MACs / FLOPs per image (batch=1)
    macs, flops = profile_with_fvcore(model, (1, 3, 256, 256), device)

    # Benchmark latency & throughput
    avg_ms, ips = benchmark_inference(model, device,
                                      batch_size=args.batch_size,
                                      runs=args.runs,
                                      warmup=args.warmup)

    print("\n========== Summary ==========")
    print(f"Params       : {format_params_m(n_params)}")
    print(f"MACs / image : {format_giga(macs, 'MACs')}")
    print(f"FLOPs / image: {format_giga(flops, 'FLOPs')}")
    print(f"Latency      : {avg_ms:.3f} ms / iter (batch={args.batch_size})")
    print(f"Throughput   : {ips:.2f} images/s")
    print("=============================\n")

if __name__ == "__main__":
    main()
