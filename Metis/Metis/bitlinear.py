from .quant import *
import torch.nn as nn
import torch.nn.init as init
import transformer_engine.pytorch  as te

from functools import partial

import math
from scipy.linalg import hadamard

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
    
    # # ------------------------------------------------------------------
    # # Spectral Decomposition for a matrix
    # # ------------------------------------------------------------------
    # @staticmethod
    # def svd_quant(
    #     input_:torch.Tensor, 
    #     quant_func, 
    #     rank=60, 
    #     niter=0, 
    #     broadcast_dim=-1,
    #     tp_simulate: bool = False,      # 新增：是否开启“竖切 TP 模拟”
    #     tp_parts: int = 4,              # 新增：把 hidden 维等分为多少份（默认四等分，即取 1/4）
    # ):
    #     """Decompose input by low-rank + residual, then recompose.
    #     Steps:
    #     1) (Optional) pick a representative slice if broadcast_dim >= 0 to reduce
    #     SVD cost while sharing factors across the broadcast dimension.
    #     2) If input is 3D [B, T, D], flatten to 2D [B*T, D] for SVD on the last dim.
    #     3) Compute randomized/low-rank SVD with (q=rank, niter=power iters).        
    #     4) Form low-rank kernel U S V^T and compute residual R = input - USV^T.
    #     6) Quantize/dequantize residual with quant_func.
    #     7) Quantize/dequantize U and V factors as well (S kept in higher precision).
    #     8) Reconstruct: (quant(U) S quant(V)^T) + residual * residual_scale.
    #     9) Reshape back if originally 3D.
    #     """
                
    #     did_select_broadcast_dim = False  # 记录是否走了 B 分支（用于后续 unsqueeze）
    #     if tp_simulate:
    #         cinput = input_
    #         D = cinput.shape[-1]
    #         parts = max(1, int(tp_parts))
    #         part_idx = int(torch.randint(low=0, high=parts, size=(1,)).item())
    #         c0 = (D * part_idx) // parts
    #         c1 = (D * (part_idx + 1)) // parts
    #         if c1 <= c0:  # 兜底
    #             c0, c1 = 0, D
    #         cinput = cinput[..., c0:c1]  # 竖切子块
    #     elif broadcast_dim >= 0:
    #         cinput = input_.select(broadcast_dim, 0)
    #         did_select_broadcast_dim = True
    #     else:
    #         cinput = input_
        
    #     original_shape = cinput.shape
    #     if len(original_shape) == 3:
    #         cinput = cinput.reshape(-1, original_shape[-1])
    #         input_ = input_.reshape(-1, original_shape[-1])
        
        
    #     ug, sg, vg = torch.svd_lowrank(
    #         cinput, 
    #         q=rank, 
    #         niter=niter
    #     )
        
    #     vg = vg.T
    #     ug = ug.T                

    #     # ker = (ug.T @ torch.diag(sg) @ vg)
    #     # if broadcast_dim >= 0:
    #     #     ker = ker.unsqueeze(broadcast_dim)

    #     # input_res = input_ - ker
    #     # input_res_scalar = quant_func.get_scalar(input_res)
    #     # input_res = quant_func.quant(input_res, input_res_scalar)
    #     # input_res = quant_func.rquant(input_res, input_res_scalar)

    #     ug_scalar = quant_func.get_scalar(ug)
    #     vg_scalar = quant_func.get_scalar(vg)
    #     ug = quant_func.quant(ug, ug_scalar)
    #     ug = quant_func.rquant(ug, ug_scalar)
        
    #     vg = quant_func.quant(vg, vg_scalar)
    #     vg = quant_func.rquant(vg, vg_scalar)
        
    #     ker = (ug.T @ torch.diag(sg) @ vg)
        
    #     # --- 广播/还原：三分支衔接 ---
    #     if tp_simulate:
    #         # TP 分支：把 ker(…, D_sub) 沿 hidden 维重复到全宽 D_full
    #         D_full = input_.shape[-1]  # 展平后的 hidden 宽度
    #         D_sub  = ker.shape[-1]
    #         rep = (D_full + D_sub - 1) // D_sub
    #         ker = ker.repeat(1, rep)[..., :D_full]
    #         # 注意：TP 分支不做 broadcast_dim 的 unsqueeze
    #     elif did_select_broadcast_dim:
    #         # 仅当对 batch/seq 维做过 select 时才需要 unsqueeze 回来
    #         ker = ker.unsqueeze(broadcast_dim)

    #     input_res = input_ - ker
    #     input_res_scalar = quant_func.get_scalar(input_res)
    #     input_res = quant_func.quant(input_res, input_res_scalar)
    #     input_res = quant_func.rquant(input_res, input_res_scalar)
        
    #     # input_ = ug.T @ torch.diag(sg) @ vg
    #     # if broadcast_dim >= 0:
    #     #     input_ = input_.unsqueeze(broadcast_dim)

    #     # input_ = input_ + input_res
    #     input_ = ker + input_res
        
    #     if len(original_shape) == 3:
    #         input_ = input_.view(original_shape[0], original_shape[1], -1)
    #     return input_
        

    @staticmethod
    def svd_quant(
        input_: torch.Tensor,
        quant_func,
        rank: int = 60,
        niter: int = 0,
        broadcast_dim: int = -1,
        tp_simulate: bool = False,   # 是否开启“竖切 TP 模拟”
        tp_parts: int = 4,           # 把 hidden 维等分为多少份
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

            # --- 低秩 SVD ---
            ug, sg, vg = torch.svd_lowrank(
                cinput_flat,
                q=rank,
                niter=niter,
            )

            vg = vg.T
            ug = ug.T

            # --- 量化 U、V ---
            ug_scalar = quant_func.get_scalar(ug)
            vg_scalar = quant_func.get_scalar(vg)

            ug = quant_func.quant(ug, ug_scalar)
            ug = quant_func.rquant(ug, ug_scalar)

            vg = quant_func.quant(vg, vg_scalar)
            vg = quant_func.rquant(vg, vg_scalar)

            # --- 重建低秩核 ker ---
            ker = (ug.T @ torch.diag(sg) @ vg)

            # --- 如果之前在某个维度 select 了一个切片，这里再 unsqueeze 回去 ---
            if did_select_broadcast_dim:
                ker = ker.unsqueeze(broadcast_dim)

            # --- 残差量化 ---
            input_res = input_flat - ker
            input_res_scalar = quant_func.get_scalar(input_res)
            input_res = quant_func.quant(input_res, input_res_scalar)
            input_res = quant_func.rquant(input_res, input_res_scalar)

            out = ker + input_res

            # --- 如果原来是 3D，则还原回 [B, T, D_chunk] ---
            if len(original_shape) == 3:
                out = out.view(original_shape[0], original_shape[1], -1)

            return out

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
        return _svd_quant_single(input_)

    
    @staticmethod
    def forward(ctx, input_: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor):
        
        
        wdim = weight.shape[-1]
        idim = input_.shape[-1]
        if (not hasattr(LinearLowbitFunction, "h")) and LinearLowbitFunction.enable_nv_recipe:
            LinearLowbitFunction.hdim = 4096
            H_scipy = hadamard(LinearLowbitFunction.hdim)
            LinearLowbitFunction.h = torch.from_numpy(H_scipy).float().to(weight.device)
        

        if LinearLowbitFunction.enable_activation_svd:
            input_ = LinearLowbitFunction.svd_quant(
                input_, 
                quant_func=LinearLowbitFunction.q_forward_input,
                rank=LinearLowbitFunction.activation_lowrank_svd,
                niter=LinearLowbitFunction.activation_lowrank_niter,
                broadcast_dim=LinearLowbitFunction.activation_broadcast_dim,
                tp_simulate=LinearLowbitFunction.tp_simulate,
                tp_parts=LinearLowbitFunction.tp_parts,
            )
            input_scalar = LinearLowbitFunction.q_forward_input.get_scalar(input_)
        else:
            
            if LinearLowbitFunction.enable_nv_recipe:
                input_ = input_ @ LinearLowbitFunction.h[: idim]
            
            input_scalar = LinearLowbitFunction.q_forward_input.get_scalar(input_)
            input_ = LinearLowbitFunction.q_forward_input.quant(input_, input_scalar)
            input_ = LinearLowbitFunction.q_forward_input.rquant(input_, input_scalar)
            if LinearLowbitFunction.enable_nv_recipe:
                input_ = input_ @ LinearLowbitFunction.h[: idim].mT / LinearLowbitFunction.hdim
    
        
        if LinearLowbitFunction.enable_nv_recipe:
            weight = weight @ LinearLowbitFunction.h[: wdim]
        weight_scalar = LinearLowbitFunction.q_forward_input.get_scalar(weight)
        weight = LinearLowbitFunction.q_forward_weight.quant(weight, weight_scalar)
        weight = LinearLowbitFunction.q_forward_weight.rquant(weight, weight_scalar)
        if LinearLowbitFunction.enable_nv_recipe:
            weight = weight @ LinearLowbitFunction.h[: wdim].mT / LinearLowbitFunction.hdim
            
            
        ctx.save_for_backward(
            input_, 
            weight, 
            input_scalar, 
            weight_scalar, 
            bias
        )
        
        
        
        output = torch.matmul(input_, weight.T)
        
        if bias is not None:
            output += bias
        
        
        return output
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        input_, weight, input_scalar, weight_scalar, bias = ctx.saved_tensors

        # input_ = LinearLowbitFunction.q_backward_input.quant(input_, input_scalar)
        # weight = LinearLowbitFunction.q_backward_weight.quant(weight, weight_scalar)
        # input_ = LinearLowbitFunction.q_backward_input.rquant(input_, input_scalar)
        # weight = LinearLowbitFunction.q_backward_weight.rquant(weight, weight_scalar)
        
        
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
            gdim = grad_output.shape[-1]
            if LinearLowbitFunction.enable_nv_recipe:
                grad_output = grad_output @ LinearLowbitFunction.h[: gdim]
            
            grad_output_scalar = LinearLowbitFunction.q_backward_outputgrad.get_scalar(grad_output)
            
            grad_output = LinearLowbitFunction.q_backward_outputgrad.quant(grad_output, grad_output_scalar)
            grad_output = LinearLowbitFunction.q_backward_outputgrad.rquant(grad_output, grad_output_scalar)
            if LinearLowbitFunction.enable_nv_recipe:
                grad_output = grad_output @ LinearLowbitFunction.h[: gdim].mT / LinearLowbitFunction.hdim
            
            grad_output = grad_output.reshape(-1, grad_output.shape[-1]).T
            
            
        grad_weight = torch.matmul(
            grad_output,
            input_.reshape(-1, input_.shape[-1])
        )
    
        grad_output = grad_output.T.reshape(grad_output_shape0, grad_output_shape1, grad_output_shape2)
        grad_input = torch.matmul(grad_output, weight)                    
        
        return grad_input, grad_weight, grad_bias

class LinearLowbit(torch.nn.Module):
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        bias=True,
        args=None, 
        device=None
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(
            torch.empty((out_features, in_features), dtype=torch.float32, device=args.device if device is None else device)
        )
        if bias:
            self.bias = torch.nn.Parameter(
                torch.empty((out_features,), dtype=torch.float32, device=args.device if device is None else device)
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
    
    def forward(self, input):
        return LinearLowbitFunction.apply(input, self.weight, self.bias)

    pass

class BitLinear(nn.Module):
    def __init__(
        self, 
        in_features, 
        out_features,
        args=None,
        bias=True
    ):
        super().__init__()
        if args.enable_forward_svd == False and args.enable_lowbit == True:
            if args.enable_te:
                self.warmup_linear = te.Linear(in_features, out_features, device=args.device)
            else:
                self.warmup_linear = LinearLowbit(in_features, out_features, bias=bias, args=args)
        else:
            self.warmup_linear = nn.Linear(in_features, out_features, bias=bias, device=args.device)
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
        
        LinearLowbitFunction.tp_simulation = args.tp_simulation
        LinearLowbitFunction.tp_parts = args.tp_parts
        
        
        


        self.args = args
        self.is_svd_quant = False
        
        
        if args.forward_svd_warmup_steps <= 0 and args.enable_forward_svd:
            print("split")
            self.split()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_svd_quant:
            y = self.vlinear(x)
            y = torch.mul(self.s, y)
            y = self.ulinear(y)
            if self.args.forward_svd_rank > 0:
                y += self.warmup_linear(x)
            
            
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
            u, s, v = torch.linalg.svd(self.warmup_linear.weight, full_matrices=False)
            u = u.cuda(self.warmup_linear.weight.get_device())
            s = s.cuda(self.warmup_linear.weight.get_device())
            v = v.cuda(self.warmup_linear.weight.get_device())
            
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
                    # device=device
                )
                self.ulinear = LinearLowbit(
                    self.args.forward_svd_rank, # u.shape[1] // 30, 
                    u.shape[0], 
                    bias=False,
                    args=self.args
                    # device=device
                )
                self.vlinear.weight.copy_(v[: self.args.forward_svd_rank, :])
                self.ulinear.weight.copy_(u[:, : self.args.forward_svd_rank])
            else:
                self.vlinear = nn.Linear(
                    v.shape[1], 
                    v.shape[0], # v.shape[0] // 30, 
                    bias=False, 
                    # args=self.args,
                    device=device
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
        