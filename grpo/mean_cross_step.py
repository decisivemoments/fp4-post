import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.decomposition import PCA
from pathlib import Path
import os
import pickle

def analyze_cross_step(pt_path: str, output_root: str = "cross_step_plots", top_k_channels: int = 20):
    data = torch.load(pt_path, map_location='cpu')
    if not data:
        print("⚠️  pt 文件为空")
        return

    layer_name = Path(pt_path).stem
    save_dir = os.path.join(output_root, layer_name)
    os.makedirs(save_dir, exist_ok=True)

    # ── 构建矩阵 M: (num_steps, h) ───────────────────────────────────
    means = []
    for record in data:
        act = record['activation'].float()
        b, s, h = act.shape
        means.append(act.reshape(-1, h).mean(dim=0).numpy())

    M = np.stack(means, axis=0)          # (T, h)
    T, h = M.shape
    steps = np.arange(T)
    print(f"✅ M.shape = {M.shape}")

    # ── 图1：热力图 ───────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(min(h // 32, 24), max(T // 8, 6)))
    im = ax.imshow(M, aspect='auto', interpolation='nearest', cmap='RdBu_r')
    ax.set_xlabel("Channel (hidden dim)")
    ax.set_ylabel("Step")
    ax.set_title(f"Full Mean per Channel across Steps\n{layer_name}")
    plt.colorbar(im, ax=ax, shrink=0.6)
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "1_heatmap.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  ✅ 1_heatmap.png")

    # ── 图2：步骤间余弦相似度矩阵 ────────────────────────────────────
    norm = np.linalg.norm(M, axis=1, keepdims=True) + 1e-8
    M_normed = M / norm
    cos_sim = M_normed @ M_normed.T          # (T, T)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cos_sim, vmin=-1, vmax=1, cmap='RdBu_r', aspect='auto')
    ax.set_xlabel("Step")
    ax.set_ylabel("Step")
    ax.set_title(f"Cosine Similarity between Steps\n{layer_name}")
    plt.colorbar(im, ax=ax, shrink=0.8)
    # 每隔10步打一个刻度
    tick_gap = max(1, T // 10)
    ticks = np.arange(0, T, tick_gap)
    ax.set_xticks(ticks); ax.set_yticks(ticks)
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "2_cosine_similarity.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  ✅ 2_cosine_similarity.png")

    # ── 图3：PCA 2D 轨迹 ─────────────────────────────────────────────
    pca = PCA(n_components=2)
    M_2d = pca.fit_transform(M)              # (T, 2)
    var_ratio = pca.explained_variance_ratio_

    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(M_2d[:, 0], M_2d[:, 1],
                    c=steps, cmap='viridis', s=30, zorder=3)
    # 连线显示轨迹
    ax.plot(M_2d[:, 0], M_2d[:, 1], lw=0.5, color='gray', alpha=0.5, zorder=2)
    # 标注首尾
    ax.annotate("start", M_2d[0],  fontsize=8, color='green')
    ax.annotate("end",   M_2d[-1], fontsize=8, color='red')
    plt.colorbar(sc, ax=ax, label='Step')
    ax.set_xlabel(f"PC1 ({var_ratio[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({var_ratio[1]*100:.1f}%)")
    ax.set_title(f"PCA Trajectory of Full Mean\n{layer_name}")
    ax.grid(True, lw=0.3, alpha=0.5)
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "3_pca_trajectory.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  ✅ 3_pca_trajectory.png")

    # ── 图4：变化最大的 top-K channel 折线图 ─────────────────────────
    channel_std = M.std(axis=0)              # (h,) 每个 channel 跨步骤的标准差
    top_k_idx = np.argsort(channel_std)[-top_k_channels:][::-1]

    fig, ax = plt.subplots(figsize=(12, 5))
    cmap = plt.get_cmap('tab20')
    for i, ch in enumerate(top_k_idx):
        ax.plot(steps, M[:, ch], lw=1.0,
                label=f"ch{ch}(σ={channel_std[ch]:.3f})",
                color=cmap(i % 20), alpha=0.85)
    ax.set_xlabel("Step")
    ax.set_ylabel("Channel Mean")
    ax.set_title(f"Top-{top_k_channels} Most Varying Channels across Steps\n{layer_name}")
    ax.legend(fontsize=6, ncol=4, loc='upper right')
    ax.grid(True, lw=0.3, alpha=0.5)
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "4_top_channels.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  ✅ 4_top_channels.png")

    # ── 额外输出：channel std 分布直方图 ─────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(channel_std, bins=80, color='steelblue', alpha=0.8, edgecolor='none')
    ax.set_xlabel("Std across Steps")
    ax.set_ylabel("Number of Channels")
    ax.set_title(f"Distribution of Per-Channel Std\n{layer_name}")
    ax.grid(True, lw=0.3, alpha=0.5)
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "5_channel_std_hist.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  ✅ 5_channel_std_hist.png")

    print(f"\n🎉 全部完成，图片保存至 {save_dir}")


def analyze_prediction_error(pt_path: str, output_root: str = "cross_step_plots"):
    data = torch.load(pt_path, map_location='cpu')
    layer_name = Path(pt_path).stem
    save_dir = os.path.join(output_root, layer_name)
    os.makedirs(save_dir, exist_ok=True)

    means = []
    for record in data:
        act = record['activation'].float()
        b, s, h = act.shape
        means.append(act.reshape(-1, h).mean(dim=0).numpy())

    M = np.stack(means, axis=0)   # (T, h)
    T, h = M.shape

    # ── 相邻步骤差分 ─────────────────────────────────────────────────
    diff = np.diff(M, axis=0)                        # (T-1, h)
    pred_error_std  = diff.std(axis=0)               # (h,) 每个channel的预测误差std
    pred_error_mean = np.abs(diff).mean(axis=0)      # (h,) 每个channel的平均绝对误差MAE

    # 激活值本身的均值量级
    act_mean_abs = np.abs(M).mean(axis=0)            # (h,)

    # 相对误差：MAE / |mean|
    relative_error = pred_error_mean / (act_mean_abs + 1e-8)   # (h,)

    # ── 打印汇总 ─────────────────────────────────────────────────────
    print(f"\n{'='*50}")
    print(f"激活值均值绝对值：mean={act_mean_abs.mean():.4f}, max={act_mean_abs.max():.4f}")
    print(f"相邻步预测误差 MAE：mean={pred_error_mean.mean():.4f}, max={pred_error_mean.max():.4f}")
    print(f"相对误差 MAE/|act|：mean={relative_error.mean():.4f}  ← 核心指标")
    print(f"  < 1%  的channel占比: {(relative_error < 0.01).mean()*100:.1f}%")
    print(f"  < 5%  的channel占比: {(relative_error < 0.05).mean()*100:.1f}%")
    print(f"  < 10% 的channel占比: {(relative_error < 0.10).mean()*100:.1f}%")
    print(f"{'='*50}\n")

    # ── 图：三合一 ───────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), gridspec_kw={'hspace': 0.45})

    # 上：激活值量级 vs 预测误差 MAE（按channel叠加）
    ax = axes[0]
    ax.plot(act_mean_abs,    lw=0.8, label='|act mean|',   color='steelblue')
    ax.plot(pred_error_mean, lw=0.8, label='pred MAE',     color='tomato', alpha=0.85)
    ax.set_ylabel("Value")
    ax.set_title("Activation Magnitude vs Prediction Error (MAE) per Channel")
    ax.legend(fontsize=8)
    ax.grid(True, lw=0.3, alpha=0.5)

    # 中：相对误差分布直方图
    ax = axes[1]
    ax.hist(relative_error, bins=80, color='steelblue', alpha=0.8, edgecolor='none')
    ax.axvline(0.05, color='tomato', lw=1.2, linestyle='--', label='5% threshold')
    ax.axvline(0.10, color='orange', lw=1.2, linestyle='--', label='10% threshold')
    ax.set_xlabel("Relative Error  MAE / |act mean|")
    ax.set_ylabel("Number of Channels")
    ax.set_title("Distribution of Per-Channel Relative Prediction Error")
    ax.legend(fontsize=8)
    ax.grid(True, lw=0.3, alpha=0.5)

    # 下：相邻步差分热力图（直观看哪些步骤变化大）
    ax = axes[2]
    im = ax.imshow(np.abs(diff).T, aspect='auto', cmap='hot_r',
                   interpolation='nearest')
    ax.set_xlabel("Step transition (t → t+1)")
    ax.set_ylabel("Channel")
    ax.set_title("|Δ mean| Heatmap across Steps and Channels")
    plt.colorbar(im, ax=ax, shrink=0.6)

    plt.suptitle(f"{layer_name}", fontsize=10, y=1.01)
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "6_prediction_error.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✅ 6_prediction_error.png  →  {save_dir}")

def analyze_pkl_dir(pkl_dir: str, output_root: str = "activation_plots"):
    """
    读取目录下所有 .pkl 文件，对每个 layer：
      1. 每个步骤画 mean_full vs mean_prefill 对比图 + 差值图
      2. 计算跨所有步骤的预测误差统计（prefill mean → full mean）
    """
    pkl_files = sorted(Path(pkl_dir).glob("*.pkl"))
    if not pkl_files:
        print(f"⚠️  {pkl_dir} 下没有找到 .pkl 文件")
        return

    for pkl_path in pkl_files:
        layer_name = pkl_path.stem
        save_dir   = os.path.join(output_root, layer_name)
        os.makedirs(save_dir, exist_ok=True)

        # ── 流式读取 ──────────────────────────────────────────────────
        records = []
        with open(pkl_path, 'rb') as f:
            while True:
                try:
                    records.append(pickle.load(f))
                except EOFError:
                    break
                except Exception:
                    print(f"⚠️  {pkl_path.name} 最后一条损坏，已跳过，有效={len(records)}")
                    break

        if not records:
            print(f"⚠️  {pkl_path.name} 为空，跳过")
            continue

        print(f"\n📂 {layer_name}  共 {len(records)} 条记录 → {save_dir}")

        # 收集所有步骤的均值，用于误差统计
        all_mean_full    = []   # list of (h,) numpy
        all_mean_prefill = []

        # ── 逐步绘图 ──────────────────────────────────────────────────
        for step, record in enumerate(records):
            mean_full    = record['mean_full'].float().numpy()     # (h,)
            mean_prefill = record['mean_prefill'].float().numpy()  # (h,)
            b, s, h      = record['shape']
            prefill_len  = record['prefill_len']

            all_mean_full.append(mean_full)
            all_mean_prefill.append(mean_prefill)

            diff     = mean_full - mean_prefill
            channels = np.arange(h)

            fig, axes = plt.subplots(2, 1, figsize=(max(12, h // 64), 6),
                                     sharex=True, gridspec_kw={'hspace': 0.35})

            # 上图：两条均值曲线
            ax0 = axes[0]
            ax0.plot(channels, mean_full,    lw=0.8, label=f'full  (s={s})',               color='steelblue')
            ax0.plot(channels, mean_prefill, lw=0.8, label=f'prefill (len={prefill_len})', color='tomato', alpha=0.85)
            ax0.set_ylabel("Channel Mean")
            ax0.set_title(f"Step {step:04d}  |  shape=({b},{s},{h})  prefill_len={prefill_len}")
            ax0.legend(fontsize=8, loc='upper right')
            ax0.grid(True, lw=0.3, alpha=0.5)

            # 下图：差值
            ax1 = axes[1]
            ax1.bar(channels, diff, width=1.0,
                    color=np.where(diff >= 0, 'steelblue', 'tomato'), alpha=0.7)
            ax1.axhline(0, color='black', lw=0.6)
            ax1.set_xlabel("Hidden Dim (channel index)")
            ax1.set_ylabel("Diff (full − prefill)")
            ax1.set_title("Channel-wise Difference")
            ax1.grid(True, lw=0.3, alpha=0.5)

            plt.tight_layout()
            out_path = os.path.join(save_dir, f"step_{step:04d}.png")
            fig.savefig(out_path, dpi=120, bbox_inches='tight')
            plt.close(fig)

        print(f"  ✅ {len(records)} 张逐步图已保存")

        # ── 误差统计（prefill mean → full mean）────────────────────────
        M_full    = np.stack(all_mean_full,    axis=0)   # (T, h)
        M_prefill = np.stack(all_mean_prefill, axis=0)   # (T, h)

        diff_mat  = M_full - M_prefill                   # (T, h)
        mae_per_ch       = np.abs(diff_mat).mean(axis=0)         # (h,)
        act_abs_per_ch   = np.abs(M_full).mean(axis=0)           # (h,)
        relative_error   = mae_per_ch / (act_abs_per_ch + 1e-8)  # (h,)

        print(f"  激活值均值绝对值：mean={act_abs_per_ch.mean():.4f}, max={act_abs_per_ch.max():.4f}")
        print(f"  预测误差 MAE    ：mean={mae_per_ch.mean():.4f},     max={mae_per_ch.max():.4f}")
        print(f"  相对误差 MAE/|act|：mean={relative_error.mean():.4f}")
        print(f"    < 1%  的channel占比: {(relative_error < 0.01).mean()*100:.1f}%")
        print(f"    < 5%  的channel占比: {(relative_error < 0.05).mean()*100:.1f}%")
        print(f"    < 10% 的channel占比: {(relative_error < 0.10).mean()*100:.1f}%")

        # ── 误差汇总图（3合1）────────────────────────────────────────
        fig, axes = plt.subplots(3, 1, figsize=(max(12, h // 64), 10),
                                 gridspec_kw={'hspace': 0.45})

        # ① 每 channel 的 MAE
        ax = axes[0]
        ax.plot(mae_per_ch, lw=0.8, color='tomato',    label='MAE per channel')
        ax.plot(act_abs_per_ch, lw=0.8, color='steelblue', alpha=0.7, label='|act mean|')
        ax.set_ylabel("Value")
        ax.set_title("Per-Channel MAE vs Activation Magnitude")
        ax.legend(fontsize=8)
        ax.grid(True, lw=0.3, alpha=0.5)

        # ② 相对误差分布直方图
        ax = axes[1]
        ax.hist(relative_error, bins=80, color='steelblue', alpha=0.8, edgecolor='none')
        ax.axvline(0.05, color='tomato',  lw=1.2, linestyle='--', label='5%')
        ax.axvline(0.10, color='orange',  lw=1.2, linestyle='--', label='10%')
        ax.set_xlabel("Relative Error  MAE / |act mean|")
        ax.set_ylabel("Number of Channels")
        ax.set_title("Distribution of Per-Channel Relative Error")
        ax.legend(fontsize=8)
        ax.grid(True, lw=0.3, alpha=0.5)

        # ③ 差值热力图（步骤 × channel）
        ax = axes[2]
        im = ax.imshow(np.abs(diff_mat).T, aspect='auto', cmap='hot_r',
                       interpolation='nearest')
        ax.set_xlabel("Step")
        ax.set_ylabel("Channel")
        ax.set_title("|full mean − prefill mean| Heatmap")
        plt.colorbar(im, ax=ax, shrink=0.6)

        plt.suptitle(layer_name, fontsize=10, y=1.01)
        plt.tight_layout()
        summary_path = os.path.join(save_dir, "error_summary.png")
        fig.savefig(summary_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  📊 误差汇总图 → {summary_path}")

    print("\n🎉 全部 layer 分析完成。")


if __name__ == "__main__":
    # python mean_cross_step.py ./activation_analysis/layer_model_layers_0_self_attn_q_proj_rank0.pt
    # python mean_cross_step.py ./activation_analysis/layer_model_layers_11_self_attn_q_proj_rank0.pt
    # python mean_cross_step.py ./activation_analysis/layer_model_layers_23_self_attn_q_proj_rank0.pt
    # python mean_cross_step.py ./activation_analysis
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("pt_path",       type=str)
    parser.add_argument("--output",      type=str, default="cross_step_plots")
    parser.add_argument("--top_k",       type=int, default=20)
    args = parser.parse_args()

    # analyze_cross_step(args.pt_path, args.output, args.top_k)
    # analyze_prediction_error(args.pt_path, args.output)
    analyze_pkl_dir(args.pt_path, args.output)
