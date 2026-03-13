# V1
import os
import torch
import torch_npu
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset
import matplotlib.pyplot as plt
from transformers.trainer_callback import TrainerCallback
import warnings

warnings.filterwarnings("ignore")
# 设置多NPU环境
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12355'

base_path = "/opt/huawei/edu-apaas/src/init/luoyuping/pangu1B_training"
model_path = os.path.join(base_path, "openPangu-Embedded-1B-V1.1")
output_path = os.path.join(base_path, "outputs/crash_test/bf16_alldata")
data_path = os.path.join(base_path, "data")
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)

ds = load_dataset("parquet", data_files={
    "train": os.path.join(data_path, "train-00000-of-00001-a09b74b3ef9c3b56.parquet")
})
ds["train"] = ds["train"] 
'''
train_size = 500
ds["train"] = ds["train"].select(range(min(train_size, len(ds["train"]))))
'''
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

        # 构建 labels，仅对 response 部分算 loss
        # 找到 response 开始的位置（简单方法：重新 tokenize prompt）
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



tokenized_datasets = ds.map(preprocess_function, batched=True)

training_args = TrainingArguments(
    output_dir=output_path,
    overwrite_output_dir=True,
    num_train_epochs=1,
    learning_rate=5e-5,
    warmup_steps=200,
    weight_decay=0.01,
    bf16=True,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    save_steps=100,
    save_total_limit=2,
    logging_dir=None,
    logging_strategy="steps",
    report_to=["none"],
    logging_steps=10,
    # evaluation_strategy="steps",
    #eval_steps=50,
    dataloader_drop_last=True,
    dataloader_num_workers=8,
    gradient_accumulation_steps=4,
    local_rank=int(os.getenv('LOCAL_RANK', -1)),
    torch_compile=False,
    remove_unused_columns=True,
    label_names=["labels"],
)

# 定义 Trainer
print("初始化 Trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],  # 训练集
    #eval_dataset=tokenized_datasets["train"],  # 评估集（这里简单用训练集代替）
)

# 开始训练
print("开始训练...")
trainer.train()

# 保存最终模型
print("保存模型...")
if trainer.args.local_rank in [-1, 0]:
    model.save_pretrained(os.path.join(output_path, "final_model"))
    tokenizer.save_pretrained(os.path.join(output_path, "final_model"))

print("训练完成！")
