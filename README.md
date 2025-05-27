# Alpaca QLoRA Demo

Minimal end‑to‑end pipeline for fine‑tuning **DistilGPT‑2** with **QLoRA (4‑bit)** on a small Alpaca subset.

```bash
conda create -n qlora_demo python=3.10 -y
conda activate qlora_demo
pip install -r requirements.txt

python scripts/prepare_dataset.py --size 800
python scripts/train_qlora.py
python scripts/infer.py --prompt "### Instruction:\n写一首关于春天的七言律诗"
python scripts/eval.py
```

See comments in each script for details.
