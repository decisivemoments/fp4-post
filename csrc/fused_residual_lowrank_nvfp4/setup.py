from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parents[1]
CUTLASS_ROOT = PROJECT_ROOT / ".deps" / "cutlass"

if not (CUTLASS_ROOT / "include" / "cutlass" / "cutlass.h").is_file():
    raise RuntimeError(
        f"CUTLASS headers are required at {CUTLASS_ROOT}; clone NVIDIA/cutlass first."
    )

setup(
    name="metis-fused-residual-lowrank-nvfp4",
    packages=["Metis", "Metis.Metis"],
    package_dir={"": str(PROJECT_ROOT / "src")},
    ext_modules=[
        CUDAExtension(
            "Metis.Metis._fused_residual_lowrank_nvfp4_cuda",
            [str(ROOT / "binding.cpp"), str(ROOT / "fused_residual_lowrank_nvfp4.cu")],
            include_dirs=[
                str(CUTLASS_ROOT / "include"),
                str(CUTLASS_ROOT / "tools" / "util" / "include"),
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "--ptxas-options=-v", "-std=c++17", "-gencode=arch=compute_120a,code=sm_120a"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
