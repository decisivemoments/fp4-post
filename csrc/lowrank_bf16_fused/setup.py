from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parents[1]

setup(
    name="metis-lowrank-bf16-fused",
    packages=["Metis", "Metis.Metis"],
    package_dir={"": str(PROJECT_ROOT / "src")},
    ext_modules=[
        CUDAExtension(
            "Metis.Metis._lowrank_bf16_fused_cuda",
            [
                str(ROOT / "binding.cpp"),
                str(ROOT / "lowrank_bf16_fused.cu"),
                str(ROOT / "lowrank_bf16_cutlass.cu"),
            ],
            include_dirs=[str(PROJECT_ROOT / ".deps" / "cutlass" / "include")],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "-gencode=arch=compute_120a,code=sm_120a"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
