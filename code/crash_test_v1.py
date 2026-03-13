# V1
import os
import torch
import torch_npu
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset, DatasetDict
import matplotlib.pyplot as plt
from transformers.trainer_callback import TrainerCallback
import warnings
import math
from random import sample
warnings.filterwarnings("ignore")
# 设置多NPU环境
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12355'
base_path = "/opt/huawei/edu-apaas/src/init/luoyuping/pangu1B_training"
model_path = os.path.join(base_path, "openPangu-Embedded-1B-V1.1")
output_path = os.path.join(base_path, "outputs/crash_test")
data_path = os.path.join(base_path, "data")
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
device = torch.device("npu" if torch.npu.is_available() else "cpu")
model = model.to(device)
model.eval()
#print(f"当前设备: {device}, 模型: {model}")
ds = load_dataset("parquet", data_files={
    "test": os.path.join(data_path, "test-00000-of-00001.parquet.parquet")
})
shuffle_data = False  # 设置为 True 表示打乱数据

# 随机选择 6000 条数据
indices = sample(range(len(ds["train"])), 600)
data_600 = ds["train"].select(indices)
#print(data_6000[:5])
# 如果需要打乱数据，使用 shuffle
if shuffle_data:
    data_600 = data_600.shuffle(seed=42)  # 设置随机种子以保证结果可复现
    #print(data_6000[:5])

# 划分训练集和测试集
train_ds = data_600.select(range(500))  # 前 5000 条为训练集
test_ds = data_600.select(range(500, 600))  # 后 1000 条为测试集

# 构建数据集字典

# 构建数据集字典
ds = {
    "train": train_ds,
    "test": test_ds
}

'''
test_size = 5000
total_size = len(ds['train'])
train_size = total_size - test_size
ds = ds['train'].train_test_split(test_size = test_size,shuffle = False)
'''

def preprocess_function(examples):
    # 正确的方式：对每个样本分别处理
    inputs = []
    targets = []

    for i in range(len(examples["instruction"])):
        # 处理每个样本
        instruction = examples["instruction"][i] if i < len(examples["instruction"]) else ""
        input_text = examples["input"][i] if i < len(examples["input"]) else ""
        output_text = examples["output"][i] if i < len(examples["output"]) else ""

        # 构建输入和目标
        input_str = f"{instruction}\n{input_text}".strip()
        target_str = output_text

        inputs.append(input_str)
        targets.append(target_str)

    # 对批量数据进行tokenize
    model_inputs = tokenizer(inputs, truncation=True, padding="max_length", max_length=1024)
    labels = tokenizer(targets, truncation=True, padding="max_length", max_length=1024)
    # 将填充 token 的位置设置为 -100
    labels["input_ids"] = [
        [(token if token != tokenizer.pad_token_id else -100) for token in label]
        for label in labels["input_ids"]
    ]
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

ds = DatasetDict(ds)
tokenized_datasets = ds.map(
    preprocess_function,
    batched=True,
    remove_columns=ds["train"].column_names)


training_args = TrainingArguments(
    output_dir=output_path,
    overwrite_output_dir=True,
    num_train_epochs=4,
    learning_rate=5e-5,
    warmup_steps=20,
    weight_decay=0.01,
    bf16=True,
    #fp16=True,
    #关闭bf16 and fp16就是fp32
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    save_steps=100,
    save_total_limit=2,
    logging_dir=None,
    logging_strategy="steps",
    report_to=["none"],
    logging_steps=10,
    #evaluation_strategy="steps",
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
    eval_dataset=tokenized_datasets["test"],  # 评估集（这里简单用训练集代替）
)

# 开始训练
print("开始训练...")
trainer.train()

print("训练结束，开始在验证集上评估...")
eval_results = trainer.evaluate()
print(eval_results)
print('bf16 不打乱顺序')
# 保存最终模型
print("保存模型...")
if trainer.args.local_rank in [-1, 0]:
    model.save_pretrained(os.path.join(output_path, "final_model"))
    tokenizer.save_pretrained(os.path.join(output_path, "final_model"))

print("训练完成！")
