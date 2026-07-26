from .quant import *
import torch.nn as nn
import torch.nn.init as init
# import transformer_engine.pytorch  as te

from functools import partial
import weakref

import math


def _largest_power_of_two_at_most(value: int) -> int:
    return 1 << (int(value).bit_length() - 1)

# -----------------------------------------------------------------------------
# Core autograd.Function implementing low-bit GEMM with optional Spectral Decomposition
# -----------------------------------------------------------------------------
class LinearLowbitFunction(torch.autograd.Function):
    """Custom autograd function that applies (i) optional Spectral Decomposition, and
    (ii) low-bit quantization for weights/activations/gradients.

    The actual quantizer kernels come from .quant via the `quant_func` registry
    (e.g., Cast2Fp4e2m1). Each has methods: get_scalar, quant, rquant.

    Class (static) attributes are configured once by BitLinear.__init__().
    """
    # Quantizers for different tensors (can be configured externally)
    q_forward_input = Cast2Fp4e2m1
    q_forward_weight = Cast2Fp4e2m1

    q_backward_input = Cast2Fp4e2m1
    q_backward_weight = Cast2Fp4e2m1
    q_backward_outputgrad = Cast2Fp4e2m1
    
    enable_nv_recipe = False
    # Cache the small, block-local Hadamard matrices.  These are reused by all
    # projections on one device and are much cheaper than rebuilding them.
    _hadamard_cache = {}

    # Bound temporary FP32 storage used by the block Hadamard preconditioner.
    # The final result still has the same dtype and shape as the input.
    hadamard_workspace_mb = 128
    # NVIDIA's RHT recipe uses fixed 16-wide tiles for the Wgrad operands.
    hadamard_tile_size = 16
    # ``auto`` prefers Dao-AILab's CUDA extension and falls back to chunked
    # cached-GEMM only when that optional dependency is unavailable.
    hadamard_backend = "auto"
    _dao_hadamard_transform = None
    _dao_hadamard_checked = False
    _hadamard_sign_cache = {}

    # Low-rank iteration counts for randomized SVD
    activation_lowrank_niter = 0
    backward_lowrank_niter = 0
    
    # Switches / ranks for SVD-based activation and backward grad paths
    enable_activation_svd = False
    activation_lowrank_svd = -1 # rank; <=0 means disable low-rank route

    enable_backward_svd = False
    backward_lowrank_svd = -1

    # Optional broadcast dim for batched SVD (select first slice along dim)
    activation_broadcast_dim = -1
    backward_broadcast_dim = -1
    
    tp_simulate = False
    tp_parts = 4
    
    compute_dtype = torch.float32
    metis_mode = "mean"  # 可选 "svd" 或 "mean"
    mean_cache = None
    mean_cache_key = "default"

    @staticmethod
    def _get_hadamard(size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Return a cached Sylvester Hadamard matrix of the requested size."""
        cache_key = (int(size), device, dtype)
        cached = LinearLowbitFunction._hadamard_cache.get(cache_key)
        if cached is not None:
            return cached

        if size <= 0 or (size & (size - 1)) != 0:
            raise ValueError(f"Hadamard block size must be a power of two, got {size}")

        hadamard = torch.ones((1, 1), device=device, dtype=dtype)
        while hadamard.shape[0] < size:
            hadamard = torch.cat(
                (
                    torch.cat((hadamard, hadamard), dim=1),
                    torch.cat((hadamard, -hadamard), dim=1),
                ),
                dim=0,
            )
        LinearLowbitFunction._hadamard_cache[cache_key] = hadamard
        return hadamard

    @staticmethod
    def _get_hadamard_sign(size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Return one deterministic, shared RHT sign vector per tile size."""
        cache_key = (int(size), device, dtype)
        cached = LinearLowbitFunction._hadamard_sign_cache.get(cache_key)
        if cached is not None:
            return cached

        generator = torch.Generator(device="cpu")
        generator.manual_seed(1729 + size)
        signs = torch.randint(0, 2, (size,), generator=generator, dtype=torch.int8)
        signs = signs.mul_(2).sub_(1).to(device=device, dtype=dtype)
        LinearLowbitFunction._hadamard_sign_cache[cache_key] = signs
        return signs

    @staticmethod
    def _get_dao_hadamard_transform():
        """Lazily load the optional fused CUDA Hadamard extension once."""
        if not LinearLowbitFunction._dao_hadamard_checked:
            try:
                from fast_hadamard_transform import hadamard_transform
                LinearLowbitFunction._dao_hadamard_transform = hadamard_transform
            except ImportError:
                LinearLowbitFunction._dao_hadamard_transform = None
            LinearLowbitFunction._dao_hadamard_checked = True
        return LinearLowbitFunction._dao_hadamard_transform

    @staticmethod
    def _apply_hadamard_blocks(value: torch.Tensor, inverse: bool = False) -> torch.Tensor:
        """Apply tiled random Hadamard rotations with a CUDA-kernel fast path.

        For dimensions divisible by the configured tile size (16 by default),
        all tiles are flattened into one batch. The optional Dao-AILab CUDA
        extension then performs one fused FWHT launch. The fallback keeps the
        same algebra but uses bounded, cached-GEMM chunks, never ``torch.cat``.
        """
        dim = value.shape[-1]
        if dim <= 1:
            return value
        if value.numel() == 0:
            return value

        tile_size = int(LinearLowbitFunction.hadamard_tile_size)
        if tile_size <= 0 or (tile_size & (tile_size - 1)):
            raise ValueError(f"Hadamard tile size must be a positive power of two, got {tile_size}")
        if dim % tile_size:
            raise ValueError(
                f"RHT tile size {tile_size} must divide hidden dimension {dim}; "
                "set --metis_hadamard_tile_size to a compatible power of two"
            )

        backend = LinearLowbitFunction.hadamard_backend
        if backend not in {"auto", "dao_cuda", "torch_gemm"}:
            raise ValueError(f"Unsupported Hadamard backend: {backend}")

        # The Dao extension accepts BF16 directly and has one output allocation.
        # It is intentionally used only for CUDA tensors; CPU tests and explicit
        # fallback runs retain the portable GEMM implementation below.
        dao_transform = (
            LinearLowbitFunction._get_dao_hadamard_transform()
            if backend in {"auto", "dao_cuda"} and value.is_cuda
            else None
        )
        if backend == "dao_cuda" and dao_transform is None:
            raise RuntimeError(
                "Hadamard backend dao_cuda was requested but fast_hadamard_transform is unavailable. "
                "Install it with: pip install -v git+https://github.com/Dao-AILab/fast-hadamard-transform.git"
            )

        flat_value = value.reshape(-1, tile_size)
        signs = LinearLowbitFunction._get_hadamard_sign(tile_size, value.device, value.dtype)
        if dao_transform is not None:
            if inverse:
                transformed = dao_transform(flat_value, scale=1.0 / tile_size)
                transformed = transformed * signs
            else:
                transformed = dao_transform(flat_value * signs, scale=1.0)
            return transformed.reshape_as(value)

        hadamard_dtype = torch.float32 if value.dtype in (torch.float16, torch.bfloat16) else value.dtype
        flat_value = value.reshape(-1, dim)
        flat_output = torch.empty_like(flat_value)
        max_workspace_bytes = max(1, int(LinearLowbitFunction.hadamard_workspace_mb)) * 1024 * 1024

        offset = 0
        while offset < dim:
            block_size = tile_size
            # One FP32 input workspace and one FP32 GEMM result are live per
            # chunk. The final-dtype output is preallocated separately.
            bytes_per_row = 8 * block_size * torch.empty((), dtype=hadamard_dtype).element_size()
            chunk_rows = max(1, max_workspace_bytes // bytes_per_row)
            hadamard = LinearLowbitFunction._get_hadamard(
                block_size, flat_value.device, hadamard_dtype
            )

            for row_start in range(0, flat_value.shape[0], chunk_rows):
                row_end = min(row_start + chunk_rows, flat_value.shape[0])
                # BF16/FP16 inputs are promoted for the rotation. For FP32,
                # this is a view and ``torch.mm`` remains non-mutating.
                work = flat_value[row_start:row_end, offset : offset + block_size].to(hadamard_dtype)
                signs = LinearLowbitFunction._get_hadamard_sign(
                    block_size, work.device, hadamard_dtype
                )
                if inverse:
                    transformed = torch.mm(work, hadamard)
                    transformed.mul_(1.0 / block_size).mul_(signs)
                else:
                    # x S H is a random Hadamard rotation; the inverse is
                    # H S / d, used above after dequantization.
                    transformed = torch.mm(work * signs, hadamard)
                flat_output[row_start:row_end, offset : offset + block_size].copy_(transformed)
            offset += block_size

        return flat_output.reshape_as(value)
    
    @staticmethod
    def svd_quant(
        input_: torch.Tensor,
        quant_func,
        rank: int = 60,
        niter: int = 0,
        broadcast_dim: int = -1,
        tp_simulate: bool = False,   # 是否开启“竖切 TP 模拟”
        tp_parts: int = 4,           # 把 hidden 维等分为多少份
        metis_mode="mean",
        mean_cache: dict = None, 
        cache_key: str = "default"
    ):
        """Decompose input by low-rank + residual, then recompose.
        Steps:
        1) (Optional) pick a representative slice if broadcast_dim >= 0 to reduce
        SVD cost while sharing factors across the broadcast dimension.
        2) If input is 3D [B, T, D], flatten to 2D [B*T, D] for SVD on the last dim.
        3) Compute randomized/low-rank SVD with (q=rank, niter=power iters).
        4) Form low-rank kernel U S V^T and compute residual R = input - USV^T.
        5) Quantize/dequantize residual with quant_func.
        6) Quantize/dequantize U and V factors as well (S kept in higher precision).
        7) Reconstruct: (quant(U) S quant(V)^T) + residual.
        8) Reshape back if originally 3D.

        If tp_simulate=True, the last (hidden) dimension is split into `tp_parts`
        vertical chunks, and the above procedure is applied independently on each
        chunk, then concatenated along the last dim.
        """

        def _svd_quant_single(chunk: torch.Tensor) -> torch.Tensor:
            """对给定的 chunk(…, D_chunk) 做一次原始的 svd_quant 逻辑。"""
            did_select_broadcast_dim = False

            # --- 选择代表切片（用于 broadcast_dim 优化） ---
            if broadcast_dim >= 0:
                cinput = chunk.select(broadcast_dim, 0)
                did_select_broadcast_dim = True
            else:
                cinput = chunk

            original_shape = cinput.shape

            # --- 如果是 3D，则展平成 [B*T, D_chunk] 做 SVD ---
            if len(original_shape) == 3:
                cinput_flat = cinput.reshape(-1, original_shape[-1])
                input_flat = chunk.reshape(-1, original_shape[-1])
            else:
                cinput_flat = cinput
                input_flat = chunk

            original_dtype = cinput_flat.dtype
            cinput_flat_fp32 = cinput_flat.to(torch.float32)
            if torch.isnan(cinput_flat_fp32).any() or torch.isinf(cinput_flat_fp32).any():
                print(f"⚠️ Warning: NaN or Inf detected in input before SVD")
            # --- 低秩 SVD ---
            with torch.amp.autocast(cinput_flat_fp32.device.type, enabled=False):
                ug, sg, vg = torch.svd_lowrank(
                    cinput_flat_fp32,
                    q=rank,
                    niter=niter,
                )
            ug = ug.to(original_dtype)
            sg = sg.to(original_dtype)
            vg = vg.to(original_dtype)
            
            vg = vg.T
            # ug = ug.T

            # --- 量化 U、V ---
            ug_scalar = quant_func.get_scalar(ug)
            vg_scalar = quant_func.get_scalar(vg)

            ug = quant_func.quant(ug, ug_scalar)
            ug = quant_func.rquant(ug, ug_scalar)

            vg = quant_func.quant(vg, vg_scalar)
            vg = quant_func.rquant(vg, vg_scalar)

            # --- 重建低秩核 ker ---
            ker = (ug @ torch.diag(sg) @ vg)

            # --- 如果之前在某个维度 select 了一个切片，这里再 unsqueeze 回去 ---
            if did_select_broadcast_dim:
                ker = ker.unsqueeze(broadcast_dim)

            # --- 残差量化 ---
            input_res = input_flat - ker
            input_res = quant_func.quantize_dequantize(input_res)

            out = ker + input_res

            # --- 如果原来是 3D，则还原回 [B, T, D_chunk] ---
            if len(original_shape) == 3:
                out = out.view(original_shape[0], original_shape[1], -1)

            return out

        def _mean_quant_single(input_: torch.Tensor) -> torch.Tensor:
            original_shape = input_.shape
            input_flat = input_.reshape(-1, original_shape[-1])
            seq_len = original_shape[1] if len(original_shape) == 3 else original_shape[0]
            if seq_len == 1 and mean_cache is not None and cache_key in mean_cache:
                # Decode：加权滚动更新
                # print(f"Using cached mean for {cache_key} with seq_len={seq_len}")
                cached_mean, cached_count = mean_cache[cache_key]
                new_count = cached_count + input_flat.shape[0]
                this_mean = input_flat.mean(dim=0, keepdim=True)
                me = (cached_mean * cached_count + this_mean * input_flat.shape[0]) / new_count
                mean_cache[cache_key] = (me.detach(), new_count)
            else:
                # Prefill 或首次
                # print(f"Calculating mean for {cache_key} with seq_len={seq_len}")
                me = input_flat.mean(dim=0, keepdim=True)
                if mean_cache is not None:
                    mean_cache[cache_key] = (me.detach(), input_flat.shape[0])

            input_res = input_flat - me
            input_res = quant_func.quantize_dequantize(input_res)
            input_ = me + input_res

            if len(original_shape) == 3:
                input_ = input_.view(original_shape[0], original_shape[1], -1)
            return input_

        # =========================
        # 竖切 TP 模拟逻辑
        # =========================
        if tp_simulate:
            # 按 hidden 维（最后一维）等分为 tp_parts 份，每份各自做一次 _svd_quant_single
            D_full = input_.shape[-1]
            parts = max(1, int(tp_parts))

            outputs = []
            for p in range(parts):
                c0 = (D_full * p) // parts
                c1 = (D_full * (p + 1)) // parts
                if c1 <= c0:
                    continue  # 防止奇怪的整除情况

                chunk = input_[..., c0:c1]          # 竖切子块 (…, D_chunk)
                chunk_out = _svd_quant_single(chunk)
                outputs.append(chunk_out)

            # 在最后一维上拼回来，形状与原始 input_ 完全一致
            return torch.cat(outputs, dim=-1)

        # =========================
        # 原始（非 TP）逻辑
        # =========================
        if metis_mode == "svd":
            return _svd_quant_single(input_)
        elif metis_mode == "mean":
            return _mean_quant_single(input_)
        else:
            raise ValueError(f"Unsupported metis_mode: {metis_mode}")

    
    @staticmethod
    def quantize_input(
        input_: torch.Tensor,
        mean_cache: dict = None,
        cache_key: str = "default",
    ) -> torch.Tensor:
        input_ = input_.to(LinearLowbitFunction.compute_dtype)
        if LinearLowbitFunction.enable_activation_svd:
            return LinearLowbitFunction.svd_quant(
                input_,
                quant_func=LinearLowbitFunction.q_forward_input,
                rank=LinearLowbitFunction.activation_lowrank_svd,
                niter=LinearLowbitFunction.activation_lowrank_niter,
                broadcast_dim=LinearLowbitFunction.activation_broadcast_dim,
                tp_simulate=LinearLowbitFunction.tp_simulate,
                tp_parts=LinearLowbitFunction.tp_parts,
                metis_mode=LinearLowbitFunction.metis_mode,
                mean_cache=mean_cache,
                cache_key=cache_key,
            )

        input_ = LinearLowbitFunction.q_forward_input.quantize_dequantize(input_)
        return input_

    @staticmethod
    def _quantize_wgrad_operand(value: torch.Tensor, quantizer) -> torch.Tensor:
        """QDQ one Wgrad operand, with RHT only for the NV-Hadamard baseline."""
        if LinearLowbitFunction.enable_nv_recipe:
            value = LinearLowbitFunction._apply_hadamard_blocks(value, inverse=False)
        scalar = quantizer.get_scalar(value)
        value = quantizer.quant(value, scalar)
        value = quantizer.rquant(value, scalar)
        if LinearLowbitFunction.enable_nv_recipe:
            value = LinearLowbitFunction._apply_hadamard_blocks(value, inverse=True)
        return value

    @staticmethod
    def forward(
        ctx,
        input_: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        quantized_weight: torch.Tensor,
        quantized_input: torch.Tensor | None,
    ):
        input_original_dtype = input_.dtype
        if quantized_input is None:
            input_ = LinearLowbitFunction.quantize_input(
                input_,
                mean_cache=LinearLowbitFunction.mean_cache,
                cache_key=LinearLowbitFunction.mean_cache_key,
            )
        else:
            input_ = quantized_input
    
        
        ctx.save_for_backward(
            input_, 
            quantized_weight,
            bias
        )
        
        
        
        output = torch.matmul(input_, quantized_weight.T)
        
        if bias is not None:
            output += bias
        
        output.to(input_original_dtype)
        return output
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        input_, weight, bias = ctx.saved_tensors

        # input_ = LinearLowbitFunction.q_backward_input.quant(input_, input_scalar)
        # weight = LinearLowbitFunction.q_backward_weight.quant(weight, weight_scalar)
        # input_ = LinearLowbitFunction.q_backward_input.rquant(input_, input_scalar)
        # weight = LinearLowbitFunction.q_backward_weight.rquant(weight, weight_scalar)
        
        grad_original_dtype = grad_output.dtype
        grad_output = grad_output.to(LinearLowbitFunction.compute_dtype)
        
        grad_bias = grad_output.sum(dim=(0, 1)) if bias is not None else None
        
        grad_output_shape0 = grad_output.shape[0]
        grad_output_shape1 = grad_output.shape[1]
        grad_output_shape2 = grad_output.shape[2]

        
        if LinearLowbitFunction.enable_backward_svd:
            if LinearLowbitFunction.backward_lowrank_svd > 0:
                # print(" ============= ", grad_output.shape)
                grad_output = LinearLowbitFunction.svd_quant(
                    grad_output, 
                    quant_func=LinearLowbitFunction.q_backward_outputgrad,
                    rank=LinearLowbitFunction.backward_lowrank_svd,
                    niter=LinearLowbitFunction.backward_lowrank_niter,
                    broadcast_dim=LinearLowbitFunction.backward_broadcast_dim,
                    tp_simulate=LinearLowbitFunction.tp_simulate,
                    tp_parts=LinearLowbitFunction.tp_parts,
                    metis_mode=LinearLowbitFunction.metis_mode
                )
                grad_output = grad_output.reshape(-1, grad_output.shape[-1]).T

            else:
                ug, sg, vg = torch.linalg.svd(grad_output, full_matrices=False)
                ug_scalar = ug.abs().mean()
                vg_scalar = vg.abs().mean()
                
                grad_output = \
                    LinearLowbitFunction.q_backward_outputgrad(ug / ug_scalar) @ \
                    torch.diag(sg) @ \
                    LinearLowbitFunction.q_backward_outputgrad(vg / vg_scalar)

                grad_output *= ug_scalar * vg_scalar
        else:
            grad_output = LinearLowbitFunction._quantize_wgrad_operand(
                grad_output, LinearLowbitFunction.q_backward_outputgrad
            )
            grad_output = grad_output.reshape(-1, grad_output.shape[-1]).T
            
            
        wgrad_input = input_
        if LinearLowbitFunction.enable_nv_recipe:
            # RHT is intentionally limited to the two Wgrad operands. We do
            # not rotate forward activations or weights for this NV baseline.
            wgrad_input = LinearLowbitFunction._quantize_wgrad_operand(
                input_, LinearLowbitFunction.q_backward_input
            )
        grad_weight = torch.matmul(
            grad_output,
            wgrad_input.reshape(-1, wgrad_input.shape[-1])
        )
    
        grad_output = grad_output.T.reshape(grad_output_shape0, grad_output_shape1, grad_output_shape2)
        grad_input = torch.matmul(grad_output, weight)                    
        
        grad_weight = grad_weight.to(torch.float32)
        grad_bias = grad_bias.to(torch.float32) if grad_bias is not None else None
        
        return grad_input, grad_weight, grad_bias, None, None

class LinearLowbit(torch.nn.Module):
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        bias=True,
        args=None, 
        device=None,
        storage_dtype = torch.float32,
        compute_dtype = torch.float32
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.storage_dtype = storage_dtype
        self.compute_dtype = compute_dtype
        self.cache_quantized_weight = getattr(args, "cache_quantized_weight", False)
        self._quantized_weight_cache = None
        self.weight = torch.nn.Parameter(
            torch.empty((out_features, in_features), dtype=storage_dtype, device=args.device if device is None else device)
        )
        if bias:
            self.bias = torch.nn.Parameter(
                torch.empty((out_features,), dtype=storage_dtype, device=args.device if device is None else device)
            )
        else:
            self.bias = None
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)
    
    def forward(self, input, quantized_input=None):
        weight_compute = self.weight.to(self.compute_dtype)
        bias_compute = self.bias.to(self.compute_dtype) if self.bias is not None else None
        quantized_weight = self._get_quantized_weight(weight_compute)
        return LinearLowbitFunction.apply(
            input,
            weight_compute,
            bias_compute,
            quantized_weight,
            quantized_input,
        )

    @torch.no_grad()
    def _get_quantized_weight(self, weight_compute):
        cache_key = (
            self.weight._version,
            weight_compute.device,
            weight_compute.dtype,
            LinearLowbitFunction.q_forward_weight,
            LinearLowbitFunction.enable_nv_recipe,
        )
        if self.cache_quantized_weight and self._quantized_weight_cache is not None:
            cached_key, cached_weight = self._quantized_weight_cache
            if cached_key == cache_key:
                return cached_weight

        quant_input = weight_compute
        if LinearLowbitFunction.enable_nv_recipe:
            quant_input = LinearLowbitFunction._apply_hadamard_blocks(quant_input, inverse=False)
        quantized_weight = LinearLowbitFunction.q_forward_weight.quantize_dequantize(quant_input)
        if LinearLowbitFunction.enable_nv_recipe:
            quantized_weight = LinearLowbitFunction._apply_hadamard_blocks(
                quantized_weight,
                inverse=True,
            )

        if self.cache_quantized_weight:
            self._quantized_weight_cache = (cache_key, quantized_weight)
        return quantized_weight

    def _apply(self, fn, recurse=True):
        self._quantized_weight_cache = None
        return super()._apply(fn, recurse=recurse)

    pass

class BitLinear(nn.Module):
    rollout_merge_active = False

    def __init__(
        self, 
        in_features, 
        out_features,
        args=None,
        bias=True,
        dtype = torch.float32,
        compute_dtype = None
    ):
        super().__init__()
        self.storage_dtype = dtype  # 权重存储精度（fp32）
        self.compute_dtype = compute_dtype if compute_dtype is not None else dtype  # 计算精度（fp16/bf16）
        
        # 验证 compute_dtype 的合法性
        if self.compute_dtype not in [torch.float32, torch.float16, torch.bfloat16]:
            raise ValueError(f"Unsupported compute_dtype: {self.compute_dtype}")
        
        if args.enable_forward_svd == False and args.enable_lowbit == True:
            if args.enable_te:
                self.warmup_linear = te.Linear(in_features, out_features, device=args.device)
            else:
                self.warmup_linear = LinearLowbit(in_features, out_features, bias=bias, args=args, storage_dtype = self.storage_dtype, compute_dtype = self.compute_dtype)
        else:
            self.warmup_linear = nn.Linear(in_features, out_features, bias=bias, device=args.device, dtype=self.storage_dtype)
            init.kaiming_uniform_(self.warmup_linear.weight, a=math.sqrt(5))
            if bias:
                fan_in, _ = init._calculate_fan_in_and_fan_out(self.warmup_linear.weight)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                init.uniform_(self.warmup_linear.bias, -bound, bound)

        self.ulinear = None
        self.vlinear = None
        self.s = None

        LinearLowbitFunction.q_forward_input = quant_func[args.q_forward_input]
        LinearLowbitFunction.q_forward_weight = quant_func[args.q_forward_weight]
        compile_qdq = getattr(args, "compile_qdq", False)
        LinearLowbitFunction.q_forward_input.compile_quantize_dequantize = compile_qdq
        LinearLowbitFunction.q_forward_weight.compile_quantize_dequantize = compile_qdq
        LinearLowbitFunction.q_backward_input = quant_func[args.q_backward_input]
        LinearLowbitFunction.q_backward_weight = quant_func[args.q_backward_weight]
        LinearLowbitFunction.q_backward_outputgrad = quant_func[args.q_backward_outputgrad]

        LinearLowbitFunction.enable_backward_svd = args.enable_backward_svd
        LinearLowbitFunction.backward_lowrank_svd = args.backward_lowrank_svd
        LinearLowbitFunction.backward_lowrank_niter = args.backward_lowrank_niter
        
        LinearLowbitFunction.enable_activation_svd = args.enable_activation_svd
        LinearLowbitFunction.activation_lowrank_svd = args.activation_lowrank_svd
        LinearLowbitFunction.activation_lowrank_niter = args.activation_lowrank_niter
                
        LinearLowbitFunction.activation_broadcast_dim = args.activation_broadcast_dim
        LinearLowbitFunction.backward_broadcast_dim = args.backward_broadcast_dim
        LinearLowbitFunction.enable_nv_recipe = args.enable_nv_recipe
        LinearLowbitFunction.hadamard_workspace_mb = getattr(args, "hadamard_workspace_mb", 128)
        LinearLowbitFunction.hadamard_tile_size = getattr(args, "hadamard_tile_size", 16)
        LinearLowbitFunction.hadamard_backend = getattr(args, "hadamard_backend", "auto")
        
        LinearLowbitFunction.tp_simulation = args.tp_simulation
        LinearLowbitFunction.tp_parts = args.tp_parts
        LinearLowbitFunction.compute_dtype = self.compute_dtype
        LinearLowbitFunction.metis_mode = args.metis_mode

        self.args = args
        self.is_svd_quant = False
        
        
        if args.forward_svd_warmup_steps <= 0 and args.enable_forward_svd:
            print("split")
            self.split()
        
        self.mean_cache = {}
        self.activation_group = None
        self._merged_rollout_weight_cache = None
        self.layer_name = ""  # 由 convert_to_metis 赋值

    def _get_shared_quantized_input(self, x: torch.Tensor) -> torch.Tensor:
        group = getattr(self, "activation_group", None)
        if group is None:
            return LinearLowbitFunction.quantize_input(
                x,
                mean_cache=self.mean_cache,
                cache_key=f"{self.layer_name}.shared",
            )

        input_ref = group.get("input_ref")
        if input_ref is not None and input_ref() is x:
            return group["quantized_input"]

        quantized_input = LinearLowbitFunction.quantize_input(
            x,
            mean_cache=group["mean_cache"],
            cache_key=f"{group['name']}.shared",
        )

        def clear_cached_input(ref):
            if group.get("input_ref") is ref:
                group["input_ref"] = None
                group["quantized_input"] = None

        group["input_ref"] = weakref.ref(x, clear_cached_input)
        group["quantized_input"] = quantized_input
        return quantized_input

    @torch.no_grad()
    def _get_merged_rollout_weight(self, dtype: torch.dtype) -> torch.Tensor:
        cache_key = (
            self.vlinear.weight._version,
            self.ulinear.weight._version,
            self.s._version,
            self.warmup_linear.weight._version,
            self.vlinear.weight.device,
            dtype,
            LinearLowbitFunction.q_forward_weight,
            LinearLowbitFunction.enable_nv_recipe,
        )
        if self._merged_rollout_weight_cache is not None:
            cached_key, cached_weight = self._merged_rollout_weight_cache
            if cached_key == cache_key:
                return cached_weight

        v_weight = self.vlinear._get_quantized_weight(self.vlinear.weight.to(dtype))
        residual_weight = self.warmup_linear._get_quantized_weight(
            self.warmup_linear.weight.to(dtype)
        )
        u_weight = self.ulinear.weight.to(dtype)
        singular_values = self.s.to(dtype)
        merged_weight = (u_weight * singular_values.unsqueeze(0)) @ v_weight
        merged_weight = merged_weight + residual_weight
        self._merged_rollout_weight_cache = (cache_key, merged_weight)
        return merged_weight

    def _apply(self, fn, recurse=True):
        self._merged_rollout_weight_cache = None
        return super()._apply(fn, recurse=recurse)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        LinearLowbitFunction.mean_cache = self.mean_cache
        if self.is_svd_quant:
            share_activation = (
                isinstance(self.vlinear, LinearLowbit)
                and isinstance(self.warmup_linear, LinearLowbit)
                and self.args.forward_svd_rank > 0
            )
            if share_activation:
                shared_input = self._get_shared_quantized_input(x)
                if BitLinear.rollout_merge_active and not torch.is_grad_enabled():
                    merged_weight = self._get_merged_rollout_weight(shared_input.dtype)
                    y = torch.matmul(shared_input, merged_weight.T)
                    if self.warmup_linear.bias is not None:
                        y = y + self.warmup_linear.bias.to(y.dtype)
                    return y
                y = self.vlinear(x, quantized_input=shared_input)
            else:
                LinearLowbitFunction.mean_cache_key = f"{self.layer_name}.vlinear"
                y = self.vlinear(x)
            y = torch.mul(self.s, y)
            y = self.ulinear(y)
            if self.args.forward_svd_rank > 0:
                if share_activation:
                    y = y + self.warmup_linear(x, quantized_input=shared_input)
                else:
                    LinearLowbitFunction.mean_cache_key = f"{self.layer_name}.warmup"
                    y = y + self.warmup_linear(x)
            
            
        else:
            y = self.warmup_linear(x)
        
        return y
    
    @staticmethod
    def _init_telinear(w, weight):
        torch.nn.init.ones_(weight)
        weight.mul_(w)
    
    @torch.no_grad()
    def split(self):
        if not self.args.enable_forward_svd:
            return
        
        
        
        if not self.vlinear is None:
            u, s, v = torch.linalg.svd(
                self.ulinear.weight @ 
                torch.diag(self.s) @ 
                self.vlinear.weight, full_matrices=False)
            
            bias = self.ulinear.bias
            device = self.ulinear.weight.device
        else:
            device = self.warmup_linear.weight.device
            original_dtype = self.warmup_linear.weight.dtype  # 保存原始数据类型

            # 提升到 float32 进行 SVD 计算
            weight_fp32 = self.warmup_linear.weight.to(torch.float32)
            u, s, v = torch.linalg.svd(weight_fp32, full_matrices=False)

            # 将结果转换回原始数据类型
            u = u.to(original_dtype).cuda(self.warmup_linear.weight.get_device())
            s = s.to(original_dtype).cuda(self.warmup_linear.weight.get_device())
            v = v.to(original_dtype).cuda(self.warmup_linear.weight.get_device())
            
            if not self.warmup_linear.bias is None:
                bias = self.warmup_linear.bias.to(device=device)
            else:
                bias = None
            w = self.warmup_linear.weight.to(device=device)
            # forward svd low rank
            if self.args.forward_svd_rank > 0:
                self.warmup_linear = LinearLowbit(
                    self.warmup_linear.weight.shape[1], 
                    self.warmup_linear.weight.shape[0],
                    bias=True if not bias is None else False, 
                    args=self.args,
                    storage_dtype = original_dtype,
                    compute_dtype=self.compute_dtype
                    # device=device
                )
                if not bias is None:
                    self.warmup_linear.bias.copy_(bias)
                self.warmup_linear.weight.copy_(
                    w - \
                    u[:,:self.args.forward_svd_rank] @ \
                    torch.diag(s[:self.args.forward_svd_rank]) @ \
                    v[:self.args.forward_svd_rank]
                )
            
            
            
        
        if self.args.enable_lowbit: 
            # nv fp8
            # ******************************************************************
            # self.ss = u @ s @ u.transpose()
            # with fp8_model_init(enabled=True):
            #     self.uvlinear = te.Linear(
            #         self.warmup_linear.weight.shape[1], 
            #         self.warmup_linear.weight.shape[0], 
            #         init_method=partial(BitLinear._init_telinear, u @ v), 
            #         bias=False, 
            #         device=self.device
            #     )
            
            if self.args.enable_te:
                self.vlinear = te.Linear(
                    v.shape[1], 
                    v.shape[0], 
                    init_method=partial(BitLinear._init_telinear, v), 
                    bias=False, 
                    device=self.device
                )
                self.ulinear = te.Linear(
                    u.shape[1], 
                    u.shape[0], 
                    init_method=partial(BitLinear._init_telinear, u), 
                    bias=False, 
                    device=self.device
                )
            # ******************************************************************
            
            elif self.args.forward_svd_rank > 0:
                self.vlinear = LinearLowbit(
                    v.shape[1], 
                    self.args.forward_svd_rank, # v.shape[0] // 30, 
                    bias=False, 
                    args=self.args,
                    storage_dtype=original_dtype,
                    compute_dtype=self.compute_dtype
                    # device=device
                )
                self.ulinear = nn.Linear(
                    self.args.forward_svd_rank, # u.shape[1] // 30, 
                    u.shape[0], 
                    bias=False,
                    # device=device
                )
                self.vlinear.weight.copy_(v[: self.args.forward_svd_rank, :])
                self.ulinear.weight.copy_(u[:, : self.args.forward_svd_rank])
            else:
                self.vlinear = LinearLowbit(
                    v.shape[1], 
                    v.shape[0], # v.shape[0] // 30, 
                    bias=False, 
                    args=self.args,                    
                    storage_dtype=original_dtype,
                    compute_dtype=self.compute_dtype
                    # device=device
                )
                self.ulinear = nn.Linear(
                    u.shape[1], # u.shape[1] // 30, 
                    u.shape[0], 
                    bias=False,
                    device=device,
                    # bias=True if not bias is None else False
                )
                self.vlinear.weight.copy_(v)
                self.ulinear.weight.copy_(u)
            
            
            # # forward svd low rank
            # if self.args.forward_svd_rank > 0 and not bias is None:
            #     self.ulinear.bias.copy_(bias)
        else:
            self.vlinear = nn.Linear(v.shape[1], v.shape[0], bias=False)
            self.ulinear = nn.Linear(u.shape[1], u.shape[0])

            
            self.vlinear.weight = nn.Parameter(v)
            self.ulinear.weight = nn.Parameter(u)
            if (not bias is None):
                self.ulinear.bias = nn.Parameter(
                    self.warmup_linear.bias.clone().cuda(self.warmup_linear.weight.get_device())
                )
        
        
        self.is_svd_quant = True
        
        if self.args.forward_svd_rank > 0:
            self.s = torch.nn.Parameter(s[:self.args.forward_svd_rank])
            
        else:
            self.s = torch.nn.Parameter(s)
            self.warmup_linear = None
            
    @torch.no_grad()
    def lowrank_initialization(self, rank: int = 32):
        W = self.warmup_linear.weight
        out_features, in_features = W.shape
        r = max(1, min(int(rank), min(out_features, in_features)))

        W32 = W.detach().to(torch.float32)
        # 精确 SVD
        U, S, Vh = torch.linalg.svd(W32, full_matrices=False)
        U_r, S_r, Vh_r = U[:, :r], S[:r], Vh[:r, :]

        W_lowrank = (U_r * S_r) @ Vh_r
        W.copy_(W_lowrank.to(dtype=W.dtype, device=W.device))

        print(f"[lowrank_init] shape={tuple(W.shape)}  rank={r}")
