
import os
import json
import math
import warnings
from datetime import datetime

import torch
import torch_npu  # noqa: F401  (keep for NPU env)
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from transformers.trainer_callback import TrainerCallback
from transformers import set_seed
warnings.filterwarnings("ignore")


# -----------------------------
# 1) Pre-allreduce grad-norm tracker + DDP comm hook
# -----------------------------
class PreAllReduceGradNorm:
    def __init__(self):
        self.sum_sq = 0.0
        self.latest_norm = None
        self._seen_any_bucket = False

    def reset(self):
        self.sum_sq = 0.0
        self.latest_norm = None
        self._seen_any_bucket = False


def make_comm_hook(gn: PreAllReduceGradNorm):
    def hook(state, bucket):
        # 兼容不同 PyTorch 版本的 GradBucket API
        if hasattr(bucket, "buffer") and callable(bucket.buffer):
            t = bucket.buffer()
        elif hasattr(bucket, "get_buffer") and callable(bucket.get_buffer):
            t = bucket.get_buffer()
        elif hasattr(bucket, "get_tensor") and callable(bucket.get_tensor):
            t = bucket.get_tensor()
        else:
            raise AttributeError(
                f"Unsupported GradBucket API. Available attrs: {dir(bucket)}"
            )

        gn._seen_any_bucket = True
        gn.sum_sq += float(t.float().pow(2).sum().item())
        gn.latest_norm = math.sqrt(gn.sum_sq)

        world_size = dist.get_world_size() if dist.is_initialized() else 1
        fut = dist.all_reduce(t, op=dist.ReduceOp.SUM, async_op=True).get_future()

        def _post(fut):
            reduced = fut.value()[0]
            reduced /= world_size
            return reduced

        return fut.then(_post)

    return hook


# -----------------------------
# 2) Logger callback: write loss + grad_norm_pre_ar into jsonl
# -----------------------------
class LossGradLogger:
    def __init__(self, log_dir, prefix="loss_grad"):
        self.rank = int(os.getenv("RANK", 0))
        os.makedirs(log_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = os.path.join(log_dir, f"{prefix}_{ts}_rank{self.rank}.jsonl")
        self.f = open(self.path, "a", encoding="utf-8")

    def write(self, rec: dict):
        self.f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self.f.flush()

    def close(self):
        try:
            self.f.close()
        except Exception:
            pass

    def __del__(self):
        self.close()

class Rank0ReducedLogger:
    def __init__(self, log_dir, prefix="loss_grad_reduced"):
        self.rank = int(os.getenv("RANK", 0))
        self.enabled = (self.rank == 0)

        if self.enabled:
            os.makedirs(log_dir, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.path = os.path.join(
                log_dir, f"{prefix}_{ts}_rank0.jsonl"
            )
            self.f = open(self.path, "a", encoding="utf-8")

    def write(self, rec: dict):
        if self.enabled:
            self.f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            self.f.flush()

    def close(self):
        if self.enabled:
            self.f.close()

# -----------------------------
# 3) Trainer subclass: register comm hook once when model is DDP
# -----------------------------
class HookedTrainer(Trainer):
    def __init__(self, *args, grad_norm_obj=None, rank_logger: LossGradLogger = None,reduced_logger=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.grad_norm_obj = grad_norm_obj
        self.rank_logger = rank_logger
        self._hook_registered = False
        self.reduced_logger = reduced_logger

    def training_step(self, model, inputs, *args, **kwargs):
        # ⭐ 每个 backward 前显式 reset
        if self.grad_norm_obj is not None:
            self.grad_norm_obj.reset()
        # 1) register comm hook once
        if (self.grad_norm_obj is not None) and (not self._hook_registered):
            rank = int(os.getenv("RANK", 0))
            if rank == 0:
                print("[debug] training_step model type:", type(model))

            if isinstance(model, DDP):
                model.register_comm_hook(state=None, hook=make_comm_hook(self.grad_norm_obj))
                if rank == 0:
                    print("[debug] DDP comm hook registered ✅")
            else:
                if rank == 0:
                    print("[warn] model is not DDP in training_step; hook not registered")

            self._hook_registered = True

        # 2) compute LOCAL loss (this is what you want)
        model.train()
        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        # keep a copy of local loss BEFORE any reduction
        local_loss_val = float(loss.detach().float().item())

        # match Trainer behavior for grad accumulation
        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        # 3) backward (triggers DDP buckets + our comm hook on sync steps)
        self.accelerator.backward(loss)

        # 4) after backward, grab pre-allreduce grad norm captured by comm hook
        grad_norm_pre_ar = None
        if self.grad_norm_obj is not None and self.grad_norm_obj._seen_any_bucket and self.grad_norm_obj.latest_norm is not None:
            grad_norm_pre_ar = float(self.grad_norm_obj.latest_norm)

        # 5) write per-rank file (NO gather)
        if self.rank_logger is not None:
            rec = {
                "step": int(self.state.global_step),  # optimizer-step counter (更新发生在 optimizer step 后)
                "rank": int(os.getenv("RANK", 0)),
                "loss_local": local_loss_val,
                "grad_norm_pre_ar": grad_norm_pre_ar,  # no_sync 的 micro-step 可能会是 None
            }
            self.rank_logger.write(rec)
        # 6) reduce AFTER sync step (only when grad sync happened)
        if grad_norm_pre_ar is not None and dist.is_initialized():
            # ---- reduce loss ----
            loss_tensor = torch.tensor(
                local_loss_val, device=loss.device, dtype=torch.float32
            )
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            loss_reduced = loss_tensor.item() / dist.get_world_size()

            # ---- reduce grad norm ----
            # 正确方式：reduce sum(g^2)，再 sqrt
            gn_sq = torch.tensor(
                grad_norm_pre_ar ** 2,
                device=loss.device,
                dtype=torch.float32
            )
            dist.all_reduce(gn_sq, op=dist.ReduceOp.SUM)
            grad_norm_reduced = math.sqrt(gn_sq.item())

            # ---- rank0 write ----
            if self.reduced_logger is not None:
                self.reduced_logger.write({
                    "step": int(self.state.global_step),
                    "loss_reduced": loss_reduced,
                    "grad_norm_reduced": grad_norm_reduced,
                })

        return loss.detach()


# -----------------------------
# 4) Main
# -----------------------------
def main():
    seed = 42
    set_seed(seed)
    torch.manual_seed(seed)


    model_path = "/home/dndx/lihong/op项目/model/openPangu-Embedded-1B-V1.1"
    output_path = "/data/home/lihong/S2-1"
    data_path = "/home/dndx/lihong/op项目/data/data"
    log_dir = os.path.join(output_path, "logs")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)

    ds = load_dataset("parquet", data_files={
        "train": os.path.join(data_path, "train-00000-of-00001-a09b74b3ef9c3b56.parquet")
    })
    print("Total training samples:", len(ds["train"]))


    def preprocess_function(examples):
        texts = []
        labels = []
        for inst, inp, out in zip(examples["instruction"], examples["input"], examples["output"]):
            prompt = (
                "Below is an instruction that describes a task, paired with an input that provides further context. "
                "Write a response that appropriately completes the request.\n\n"
                "### Instruction:\n"
                f"{inst}\n\n"
                "### Input:\n"
                f"{inp}\n\n"
                "### Response:\n"
            )
            full = prompt + out

            tokenized = tokenizer(
                full,
                truncation=True,
                max_length=1024,
                padding="max_length"
            )
            input_ids = tokenized["input_ids"]

            prompt_tokenized = tokenizer(
                prompt,
                truncation=True,
                max_length=1024,
                padding="max_length"
            )
            prompt_len = sum(1 for t in prompt_tokenized["input_ids"] if t != tokenizer.pad_token_id)

            label_ids = [-100] * len(input_ids)
            for i in range(prompt_len, len(input_ids)):
                if input_ids[i] != tokenizer.pad_token_id:
                    label_ids[i] = input_ids[i]

            texts.append(input_ids)
            labels.append(label_ids)

        return {
            "input_ids": texts,
            "attention_mask": [[1 if t != tokenizer.pad_token_id else 0 for t in ids] for ids in texts],
            "labels": labels,
        }

    tokenized_datasets = ds.map(preprocess_function, batched=True, remove_columns=ds["train"].column_names)

    # ---- Training args: make sure logging is frequent enough ----
    training_args = TrainingArguments(
        output_dir=output_path,
        overwrite_output_dir=True,
        num_train_epochs=1,
        learning_rate=5e-5,
        weight_decay=0.01,
        bf16=False,
        fp16=False,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,

        seed=42,
        data_seed=42,

        logging_strategy="steps",
        logging_steps=1,           # ✅ 每个 optimizer step 都 log
        logging_first_step=True,
        logging_dir=None,
        report_to=["none"],

        dataloader_drop_last=True,
        dataloader_num_workers=8,

        local_rank=int(os.getenv("LOCAL_RANK", -1)),
        torch_compile=False,
        remove_unused_columns=True,
        label_names=["labels"],
    )

    print("初始化 Trainer...")

    pre_gn = PreAllReduceGradNorm()
    rank_logger = LossGradLogger(log_dir=log_dir, prefix="loss_grad")
    rank0_logger = Rank0ReducedLogger(log_dir=log_dir)

    trainer = HookedTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        grad_norm_obj=pre_gn,
        rank_logger=rank_logger,
        reduced_logger=rank0_logger,
    )

    if trainer.args.local_rank in [-1, 0]:
        print("开始训练...")
    trainer.train()
    rank_logger.close()
    rank0_logger.close()

    if trainer.args.local_rank in [-1, 0]:
        print("保存模型...")
        model.save_pretrained(os.path.join(output_path, "final_model"))
        tokenizer.save_pretrained(os.path.join(output_path, "final_model"))

    print("训练完成！")


if __name__ == "__main__":
    main()
