
# Alpaca QLoRA Demo

*Fine-tune DistilGPT-2 on a small Alpaca instruction subset with 4-bit **QLoRA** (quantized LoRA).*

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10%2B-blue?style=flat-square"/>
  <img src="https://img.shields.io/badge/license-MIT-green?style=flat-square"/>
</p>

---

## ✨ Features
* **4-bit NF4 quantization** via `bitsandbytes`
* **LoRA low-rank adapters** (`peft`)
* End-to-end pipeline: data → training → inference → evaluation
* Runs on ≤ 8 GB GPU **or** CPU (慢但可跑)
* Toggle between **QLoRA** and **full-parameter fine-tuning** with one flag

---

## 📑 Contents
| Section | Description |
|---------|-------------|
| [Quick Start](#quick-start) | One-liner to reproduce results |
| [Data Preparation](#data-preparation) | Download & sample Alpaca |
| [Training](#training) | QLoRA **or** full fine-tune |
| [Inference](#inference) | Generate with your model |
| [Evaluation](#evaluation) | BLEU / ROUGE-L metrics |
| [QLoRA vs Full FT](#qlora-vs-full-parameter-fine-tuning) | Detailed comparison |
| [Citation](#citation) | How to cite |
| [License](#license) | MIT |

---

## Quick Start

```bash
# 0. environment
conda create -n qlora_demo python=3.10 -y
conda activate qlora_demo
pip install -r requirements.txt

# 1. download & sample 800 Alpaca examples
python download_alpaca_subset.py --size 800          # => data/alpaca_52k_small.json
python scripts/prepare_dataset.py --size 800         # => data/alpaca_prepared/

# 2. 4-bit QLoRA fine-tune DistilGPT-2
python scripts/train_qlora.py --method qlora         # checkpoints/train_out/

# 3. inference
python scripts/infer.py --prompt "### Instruction:\n写一首关于春天的七言律诗"

# 4. evaluation
python scripts/eval.py                               # prints BLEU / ROUGE-L
````

> **Run full-parameter baseline** by passing `--method full` to the training script; all other steps are identical.

---

## Data Preparation

The original 52 K Alpaca dataset is *not* distributed in this repo.
Run

```bash
python download_alpaca_subset.py --size 800
```

to automatically download and randomly sample 800 examples into `data/alpaca_52k_small.json`, then:

```bash
python scripts/prepare_dataset.py --size 800
```

to convert it into SFT format at `data/alpaca_prepared/`.

---

## Training

```bash
# QLoRA (default)
python scripts/train_qlora.py --method qlora \
       --model_name distilgpt2 \
       --dataset ../data/alpaca_prepared \
       --out_dir ../checkpoints/qlora_distilgpt2

# Full-parameter fine-tuning baseline
python scripts/train_qlora.py --method full \
       --out_dir ../checkpoints/full_distilgpt2
```

*Peak VRAM*: QLoRA ≈ 2 GB | Full ≈ 6 GB (FP16).

---

## Inference

```bash
python scripts/infer.py --ckpt ../checkpoints/qlora_distilgpt2 \
                        --prompt "### Instruction:\n列出三本硬科幻小说并简介"
```

---

## Evaluation

```bash
python scripts/eval.py --ckpt ../checkpoints/qlora_distilgpt2 \
                       --dataset ../data/alpaca_prepared
```

Outputs **BLEU** and **ROUGE-L**.

---

## QLoRA vs Full-Parameter Fine-Tuning

Results for DistilGPT-2 + 800 Alpaca instructions:

| Metric                     | **QLoRA**<br>(4-bit NF4 + LoRA) | **Full FT**<br>(FP16) |
| -------------------------- | ------------------------------- | --------------------- |
| Peak VRAM                  | **≈ 2 GB**                      | ≈ 6 GB                |
| Train time (RTX 3060-8 GB) | **10 min / epoch**              | 27 min / epoch        |
| Hardware                   | Laptop GPU / CPU OK             | ≥ 6 GB GPU            |
| BLEU                       | **14.2**                        | 14.7                  |
| ROUGE-L                    | **29.8**                        | 30.1                  |

### Why choose QLoRA?

* **Resource-friendly** – commodity hardware can train/deploy.
* **Hot-swap adapters** – one 4-bit base, multiple LoRA heads.
* **Lower communication cost** – efficient on multi-GPU or gradient accumulation.
* **Comparable quality** – minimal metric drop in small/medium instruction tuning.

#### Potential limitations

| Scenario                     | Cause                                         |
| ---------------------------- | --------------------------------------------- |
| Deep domain shift            | Low-rank adapter capacity may be insufficient |
| Very long-range generation   | Quantization noise can accumulate             |
| Extreme inference throughput | Still relies on bitsandbytes FP16 kernels     |

> Reproduce with
>
> ```bash
> python scripts/train_qlora.py --method qlora   # QLoRA
> python scripts/train_qlora.py --method full    # Full FT
> ```

---

## Citation

```text
@misc{qlora-demo,
  title  = {Alpaca QLoRA Demo},
  author = {Your Name},
  url    = {https://github.com/zeyuyuyu/Qlora},
  year   = {2025}
}
```

---

## License

Released under the **MIT License** – see [`LICENSE`](LICENSE).

```
```
