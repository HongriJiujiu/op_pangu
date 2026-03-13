# Silent Inconsistency in Data-Parallel Full Fine-Tuning  
### Experimental Fine-Tuned Models (S1-1 / S1-2 / S1-3)

This repository provides **three fully fine-tuned models** corresponding to the experimental settings in the paper:

> **Silent Inconsistency in Data-Parallel Full Fine-Tuning: Diagnosing Worker-Level Optimization Misalignment**

The models were trained to reproduce and analyze the phenomenon of **worker-level optimization misalignment** under synchronous data-parallel (DDP) full-parameter fine-tuning.

---

## Keywords

data-parallel • distributed-training • full-finetuning • inconsistency-diagnosis • alpaca • openpangu

---

## Background

In synchronous data-parallel training with All-Reduce, model parameters are strictly synchronized after each update step. However, synchronization of parameters does **not** guarantee consistency in worker-level optimization dynamics *before gradient aggregation*.

The paper introduces the concept of:

> **Silent Inconsistency** — a hidden divergence in worker-level loss and gradient behavior that remains invisible in the global averaged loss curve.

To diagnose this issue, the paper proposes three lightweight monitoring metrics:

- **Loss Dispersion** — Standard deviation / range of per-worker loss  
- **Gradient-Norm Dispersion** — Standard deviation of per-worker gradient L2 norms  
- **Gradient-Direction Consistency** — Average cosine similarity between worker gradients  

These metrics can be computed online with negligible overhead and without modifying the optimization algorithm.

---

## Base Model

All three models are fully fine-tuned from:

**Ascend Tribe – openPangu-Embedded-1B-V1.1 (1B parameters)**  
https://ai.gitcode.com/ascend-tribe/openPangu-Embedded-1B-V1.1

- Architecture: Causal LM  
- Parameters: ~1B  
- Precision: bf16 mixed precision during training  
- Training mode: Full-parameter fine-tuning (no LoRA / adapters)

---

## Dataset

Fine-tuned on:

**tatsu-lab / alpaca (Instruction tuning dataset)**  
https://huggingface.co/datasets/tatsu-lab/alpaca

- Template format: `Instruction – Input – Response`  
- Maximum sequence length: 1024  
- Loss computed **only on the Response tokens**  
- Prompt and template tokens are masked during loss calculation  

---

## Model Repository

Hugging Face Model Hub:  
https://huggingface.co/jiujiudahaozi/op_pangu

---

## Experimental Settings

### S1-1 — Strict Consistency

- All ranks use the **same random seed**
- Deterministic `DistributedSampler`
- Identical shuffling behavior across workers

**Expected behavior**

- Low loss dispersion  
- Low gradient-norm dispersion  
- High gradient-direction consistency  

---

### S1-2 — Mild Inconsistency

- Rank 0 uses a different random seed  
- Other ranks follow the baseline seed  

**Expected behavior**

- Slight increase in worker-level dispersion  
- Global averaged loss remains smooth  

---

### S1-3 — Significant Inconsistency

- Each rank uses a distinct seed dependent on rank ID  

**Expected behavior**

- Large loss dispersion  
- Larger gradient-norm variation  
- Reduced gradient-direction consistency  
- Global loss may still appear normal  

---