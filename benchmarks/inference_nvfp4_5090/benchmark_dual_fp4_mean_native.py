"""Full native forward timing, including fresh activation packing every call."""
from __future__ import annotations
import argparse
import importlib.util
import json
from pathlib import Path

import torch
from transformer_engine.pytorch import cpp_extensions as tex
import Metis.Metis.fused_residual_lowrank_nvfp4 as api
from Metis.Metis.native_nvfp4 import NativeFullNVFP4Linear, set_native_nvfp4_profiling
from benchmark_fused_residual_lowrank_nvfp4 import measure


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=131072)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--mode", choices=("fused", "bf16", "legacy"), default="fused")
    parser.add_argument("--legacy-extension", type=Path)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.mode == "legacy":
        if args.legacy_extension is None:
            parser.error("legacy requires pre-mean-fusion extension .so")
        spec = importlib.util.spec_from_file_location(
            "legacy._fused_residual_lowrank_nvfp4_cuda", args.legacy_extension)
        legacy = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(legacy)

        def old_forward(x, r, z, u, correction, out):
            for p in (x,r,z,u):
                if not p.get_metadata()["with_gemm_swizzled_scales"]:
                    tex.swizzle_scales_for_gemm_(p)
            x,r,z,u = [p.get_metadata() for p in (x,r,z,u)]
            legacy.fused_residual_lowrank_nvfp4(
                x['rowwise_data'],x['rowwise_scale_inv'],r['rowwise_data'],r['rowwise_scale_inv'],
                x['amax_rowwise'],r['amax_rowwise'],z['rowwise_data'],z['rowwise_scale_inv'],
                u['rowwise_data'],u['rowwise_scale_inv'],z['amax_rowwise'],u['amax_rowwise'],out)
            out.add_(correction)
            return out
        api.fused_residual_lowrank_nvfp4 = old_forward
    torch.manual_seed(91)
    layer = NativeFullNVFP4Linear(896,4864,rank=64,device="cuda",bias=False,
        stochastic_rounding=False,activation_columnwise=False,
        lowrank_compute="bf16" if args.mode == "bf16" else "nvfp4").eval()
    x = torch.randn(args.rows,896,device="cuda",dtype=torch.bfloat16) + 0.5
    set_native_nvfp4_profiling(args.profile)

    @torch.inference_mode()
    def forward():
        layer.activation_group.clear()
        return layer(x)

    if args.profile:
        for _ in range(args.warmup):
            forward()
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStart()
        with torch.cuda.nvtx.range("dual_fp4_mean.full_native_forward"):
            forward()
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()
        return
    timing = measure(forward,args.warmup,args.iterations)
    result = {"mode":args.mode,"m":args.rows,"n":4864,"k":896,"rank":64,
        "gpu":torch.cuda.get_device_name(),"scope":"full native forward, fresh activation packing; warm weight cache",
        "activation_columnwise":False,"activation_pack_backend":"te","timing":timing}
    text = json.dumps(result,indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(text+'\n')


if __name__ == '__main__':
    main()
