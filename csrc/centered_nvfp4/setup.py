from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parents[1]

setup(
    name="metis-centered-nvfp4",
    packages=["Metis", "Metis.Metis"],
    package_dir={"": str(PROJECT_ROOT / "src")},
    ext_modules=[
        CUDAExtension(
            "Metis.Metis._centered_nvfp4_cuda",
            [str(ROOT / "binding.cpp"), str(ROOT / "centered_nvfp4.cu")],
            extra_compile_args={
                "cxx": ["-O3"],
                # TE's NVFP4 reference uses IEEE round-to-nearest arithmetic
                # for scaling and conversion boundaries.  Fast-math may use
                # approximate division and is not byte-compatible.
                "nvcc": ["-O3", "-gencode=arch=compute_120a,code=sm_120a"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
