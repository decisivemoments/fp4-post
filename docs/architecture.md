# Codebase Architecture

## Repository Layout

The repository is organized by responsibility:

| Path | Purpose |
| --- | --- |
| `src/grpo/` | GRPO training, QAT distillation, rollout quality analysis, activation analysis, and Metis diagnostics. |
| `src/dpo/` | DPO training variants and the Metis-enabled DPO wrapper. |
| `src/Metis/` | Metis low-bit quantization implementation, reference GPT/LLaMA models, optimizer code, and CUDA kernel benchmarks. |
| `scripts/grpo/` | Shell entry points for GRPO and QAT runs. |
| `scripts/dpo/` | Shell entry points for DPO runs. |
| `scripts/metis/` | Shell entry points for standalone Metis examples. |
| `configs/grpo/` | Accelerate and DeepSpeed configs used by GRPO/QAT scripts. |
| `configs/dpo/` | Accelerate/FSDP configs used by DPO scripts. |
| `tools/` | One-off checkpoint and tensor analysis utilities. |
| `data/metis/` | Small Metis example dataset. Large datasets should stay outside git. |
| `requirements/` | Dependency files split from source code. |
| `docs/` | Reports, notes, and this design document. |
| `outputs/` | Local checkpoints, logs, plots, TensorBoard runs, caches, and other generated artifacts. This directory is ignored by git. |

## High-Level Design

The project combines three related workflows:

1. **DPO fine-tuning**: `src/dpo/dpo.py` and dtype-specific variants wrap TRL `DPOTrainer`.
2. **GRPO fine-tuning**: `src/grpo/grpo.py` wraps TRL `GRPOTrainer`, custom reward functions, rollout logging, and optional Metis replacement.
3. **Metis low-bit training**: `src/Metis/Metis/bitlinear.py` provides `BitLinear`, while GRPO/DPO wrappers replace selected transformer `nn.Linear` layers with Metis layers.

The training scripts are intentionally thin. They set environment variables, choose accelerate/deepspeed configs, pass dataset/model/output arguments, and write all generated files under `outputs/`.

## GRPO Flow

Main entry: `src/grpo/grpo.py`

The GRPO script:

1. Parses TRL `ScriptArguments`, `GRPOConfig`, `ModelConfig`, and dataset mixture config through `TrlParser`.
2. Builds reward functions from `reward_funcs_registry`.
3. Loads the dataset through either TRL `get_dataset` or Hugging Face `load_dataset`.
4. Optionally replaces transformer projection and MLP linear layers with Metis `BitLinear`.
5. Creates `GRPOTrainer`.
6. Optionally attaches:
   - `ActivationCapture` for selected layer activation means.
   - `RolloutLogitsCapture` and `RolloutRewardWrapper` for rollout analysis.
   - `QualityAnalyzer` outputs under the training output directory.
7. Saves model and metrics to `training_args.output_dir`.

Important GRPO files:

| File | Role |
| --- | --- |
| `src/grpo/grpo.py` | Main GRPO training entry and Metis replacement helpers. |
| `src/grpo/qat_distill.py` | Teacher-student QAT distillation into a Metis low-bit student. |
| `src/grpo/metis_monitor.py` | Diagnostic callback, rollout capture, reward wrapper, and plotting helpers. |
| `src/grpo/rollout_quality.py` | Text quality metrics, per-sample logs, per-step aggregation, and quality plots. |
| `src/grpo/rollout_analysis.py` | Token/logit-level rollout metrics and plots. |
| `src/grpo/mean_cross_step.py` | Cross-step activation/statistics analysis from saved activation records. |
| `src/grpo/activation_mean_ana.py` | Activation mean plotting utility. |

## QAT Distillation Flow

Main entry: `src/grpo/qat_distill.py`

The QAT flow trains a low-bit student against a bf16 teacher:

1. Load teacher and student from the same base model path.
2. Replace student linear layers with Metis `BitLinear` when `--use_metis true`.
3. Generate teacher responses from prompts.
4. Compute KL loss between teacher logits and student logits, optionally using teacher top-k logits.
5. Optimize the student and save periodic checkpoints plus a final model under `outputs/grpo/...`.

This path is useful before running GRPO when direct FP4 reinforcement learning is unstable.

## DPO Flow

Main entries:

| File | Purpose |
| --- | --- |
| `src/dpo/dpo.py` | Main DPO script with additional NaN/Inf hooks. |
| `src/dpo/dpo_bf16.py` | bf16-oriented DPO variant. |
| `src/dpo/dpo_fp16.py` | fp16-oriented DPO variant. |
| `src/dpo/run_dpo_metis.py` | Monkey-patches `AutoModelForCausalLM.from_pretrained` so loaded models use Metis `BitLinear`. |

The standard DPO path loads a model and optional reference model, loads the preference dataset, instantiates `DPOTrainer`, trains, evaluates if configured, and saves to `outputs/dpo/...`.

The Metis DPO wrapper imports `main` and `make_parser` from `src/dpo/dpo.py`, patches model loading, and then runs the same trainer path.

## Metis Design

Main package: `src/Metis/Metis/`

Metis centers on replacing dense linear layers with low-bit and optional low-rank implementations:

| File | Role |
| --- | --- |
| `bitlinear.py` | Current `BitLinear` implementation used by GRPO/DPO integration. |
| `bitlinear_new.py`, `bitlinear_old.py` | Alternative/previous implementations kept for comparison. |
| `quant.py` | FP4/FP6/FP8 and block quantization simulation. |
| `lowrank_eig.py` | Low-rank SVD/eigendecomposition helpers. |
| `quant_view.py`, `quant_reshape.py` | Quantization variants and experiments. |

`BitLinear.split()` decomposes or prepares the original linear weight before optimizer registration. GRPO and DPO integration code copies weights from `nn.Linear` into `BitLinear.warmup_linear`, calls `split()`, and swaps the module into the model.

Standalone examples live in:

| Path | Role |
| --- | --- |
| `src/Metis/dp_main.py` | Data-parallel example training entry. |
| `src/Metis/pp_main.py` | Pipeline/sequential example training entry. |
| `src/Metis/models/` | Reference GPT and LLaMA model definitions using `BitLinear`. |
| `src/Metis/kernels/` | CUDA kernel source and benchmark scripts. |

## Run Scripts

Run scripts should be launched from anywhere; each script resolves the repository root before executing.

Common entries:

```bash
bash scripts/grpo/run_grpo.sh
bash scripts/grpo/run_grpo_long_context.sh
bash scripts/grpo/run_qat.sh

bash scripts/dpo/run_dpo.sh
bash scripts/dpo/run_dpo_metis.sh
bash scripts/dpo/run_dpo_singlegpu_fp16.sh

bash scripts/metis/train-gpt-2.sh
bash scripts/metis/train-llama.sh
```

Most scripts still contain machine-specific absolute dataset/model paths. Before running on a new machine, update:

| Argument | Meaning |
| --- | --- |
| `--dataset_name` / `--dataset-path` | Dataset location. |
| `--model_name_or_path` / `--chkpt-dir` | Base model or checkpoint location. |
| `CUDA_VISIBLE_DEVICES` | GPU selection. |
| `--output_dir` / `--log-dir` | Output location, normally under `outputs/`. |

### NV-Hadamard baseline

`hadamard_fp4` implements a tiled, 16-wide RHT around the two Wgrad operands
only; it does not rotate forward activations or weights. The default
`METIS_HADAMARD_BACKEND=auto` uses the Dao-AILab CUDA extension when installed
and otherwise falls back to a bounded cached-GEMM implementation. Install the
fast backend in the training CUDA environment with:

```bash
pip install -v git+https://github.com/Dao-AILab/fast-hadamard-transform.git
```

Use `METIS_HADAMARD_BACKEND=dao_cuda` to require the extension, or
`METIS_HADAMARD_BACKEND=torch_gemm` for the portable fallback. The model's
input and output projection dimensions must be divisible by
`METIS_HADAMARD_TILE_SIZE` (default: `16`).

## Output Policy

Generated artifacts do not belong in git. `.gitignore` excludes:

- `outputs/`
- Python caches
- logs
- TensorBoard event files
- checkpoints and ad-hoc model directories
- activation and plotting output directories

Scripts create `outputs/grpo`, `outputs/dpo`, or `outputs/metis` before writing. Analysis tools should follow the same convention.

## Development Notes

When adding new training code:

1. Put reusable Python code under `src/<area>/`.
2. Put shell launchers under `scripts/<area>/`.
3. Put accelerate/deepspeed/yaml/json configs under `configs/<area>/`.
4. Write checkpoints, logs, and plots under `outputs/<area>/`.
5. Document new workflows in `docs/`.

When adding Metis integration to a new trainer:

1. Add `src` to `sys.path` based on `__file__` if the entry is executed as a script.
2. Import `BitLinear` from `Metis.Metis`.
3. Replace only known transformer projection/MLP modules unless there is a specific reason to broaden the target list.
4. Copy original linear weights into the Metis warmup layer.
5. Call `split()` before optimizer/trainer construction.

## Known Cleanup Opportunities

- `src/dpo/dpo.py`, `src/dpo/dpo_bf16.py`, and `src/dpo/dpo_fp16.py` duplicate a large amount of DPO trainer setup.
- GRPO and DPO each define similar `MetisArgs` and layer replacement logic.
- Some scripts still include cluster-specific absolute paths and should eventually move to parameterized config files.
- Several analysis tools have hard-coded local example paths; prefer CLI arguments for new analysis utilities.
