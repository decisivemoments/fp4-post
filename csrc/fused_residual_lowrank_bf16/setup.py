from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parents[1]
CUTLASS_ROOT = PROJECT_ROOT / ".deps" / "cutlass"

setup(
    name="metis-fused-residual-lowrank-bf16",
    packages=["Metis", "Metis.Metis"],
    package_dir={"": str(PROJECT_ROOT / "src")},
    ext_modules=[
        CUDAExtension(
            "Metis.Metis._fused_residual_lowrank_bf16_cuda",
            [str(ROOT / "binding.cpp"), str(ROOT / "fused_residual_lowrank_bf16.cu")],
            include_dirs=[str(CUTLASS_ROOT / "include"), str(CUTLASS_ROOT / "tools" / "util" / "include")],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "-std=c++17", "-gencode=arch=compute_120a,code=sm_120a"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
