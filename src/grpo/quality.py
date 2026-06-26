from rollout_quality import QualityAnalyzer

analyzer = QualityAnalyzer("outputs/grpo/Qwen2_5-0.5B-grpo-metis-after-qat/rollout_quality")
analyzer.plot_all()
