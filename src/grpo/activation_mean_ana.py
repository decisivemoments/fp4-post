import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

def analyze_activations(pt_path: str, output_root: str = "activation_plots"):
    """
    读取 pt 文件，对每条 record 计算：
      - prefill 部分 (b, prefill_len, h) → (b*prefill_len, h) 每列均值
      - 全序列    (b, s, h)            → (b*s, h)           每列均值
    并绘图保存。
    """
    data = torch.load(pt_path, map_location='cpu')
    if not data:
        print("⚠️  pt 文件为空")
        return

    # 用 pt 文件名（去掉后缀）作为子文件夹名
    layer_name = Path(pt_path).stem
    save_dir = os.path.join(output_root, layer_name)
    os.makedirs(save_dir, exist_ok=True)

    print(f"📂 共 {len(data)} 条记录，图片保存至 {save_dir}")

    for step, record in enumerate(data):
        act         = record['activation'].float()          # (b, s, h)
        prefill_len = record['prefill_len']          # int
        b, s, h     = act.shape

        # ── 计算两段均值 ──────────────────────────────────────────────
        # 全序列均值：(b*s, h) → 每列均值 → (h,)
        mean_full    = act.reshape(-1, h).mean(dim=0).numpy()

        # prefill 段均值：截取前 prefill_len 个 token
        if prefill_len > 0 and prefill_len <= s:
            mean_prefill = act[:, :prefill_len, :].reshape(-1, h).mean(dim=0).numpy()
        else:
            # prefill_len 不合法时退化为全序列
            mean_prefill = mean_full.copy()

        diff = mean_full - mean_prefill                          # (h,)
        channels = np.arange(h)

        # ── 绘图 ──────────────────────────────────────────────────────
        fig, axes = plt.subplots(2, 1, figsize=(max(12, h // 64), 6),
                                 sharex=True, gridspec_kw={'hspace': 0.35})

        # 上图：两条均值曲线
        ax0 = axes[0]
        ax0.plot(channels, mean_full,    lw=0.8, label=f'full  (s={s})',          color='steelblue')
        ax0.plot(channels, mean_prefill, lw=0.8, label=f'prefill (len={prefill_len})', color='tomato', alpha=0.85)
        ax0.set_ylabel("Channel Mean")
        ax0.set_title(f"Step {step:04d}  |  shape=({b},{s},{h})  prefill_len={prefill_len}")
        ax0.legend(fontsize=8, loc='upper right')
        ax0.grid(True, lw=0.3, alpha=0.5)

        # 下图：差值
        ax1 = axes[1]
        ax1.bar(channels, diff, width=1.0, color=np.where(diff >= 0, 'steelblue', 'tomato'), alpha=0.7)
        ax1.axhline(0, color='black', lw=0.6)
        ax1.set_xlabel("Hidden Dim (channel index)")
        ax1.set_ylabel("Diff (full − prefill)")
        ax1.set_title("Channel-wise Difference")
        ax1.grid(True, lw=0.3, alpha=0.5)

        plt.tight_layout()
        out_path = os.path.join(save_dir, f"step_{step:04d}.png")
        fig.savefig(out_path, dpi=120, bbox_inches='tight')
        plt.close(fig)

        print(f"  ✅ step {step:04d}  saved → {out_path}")

    print(f"\n🎉 全部完成，共 {len(data)} 张图片。")


# ── 入口 ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # python activation_mean_ana.py --pt_path ./activation_analysis/layer_model_layers_0_self_attn_q_proj_rank0.pt
    # python activation_mean_ana.py --pt_path ./activation_analysis/layer_model_layers_11_self_attn_q_proj_rank0.pt
    # python activation_mean_ana.py --pt_path ./activation_analysis/layer_model_layers_23_self_attn_q_proj_rank0.pt
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pt_path",     type=str, help="目标 .pt 文件路径")
    parser.add_argument("--output",    type=str, default="activation_plots", help="图片输出根目录")
    args = parser.parse_args()

    analyze_activations(args.pt_path, args.output)
