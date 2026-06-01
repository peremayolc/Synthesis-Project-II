from imports import *
from config import TOKEN, REPO, TOKEN_LEGAL, REPO_LEGAL
import sys
import torch
from dataclasses import dataclass
from transformers import PreTrainedTokenizerBase
from medical import medical
from legal import legal
from automotive import automotive


@dataclass
class _CompletionCollator:
    """Pads a batch of {input_ids, attention_mask, labels} dicts.
    Labels are padded with -100 so padding positions are ignored in the loss."""
    tokenizer: PreTrainedTokenizerBase

    def __call__(self, features):
        max_len = max(len(f["input_ids"]) for f in features)
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for f in features:
            ids = f["input_ids"]
            pad = max_len - len(ids)
            batch["input_ids"].append(ids + [self.tokenizer.pad_token_id] * pad)
            attn = f.get("attention_mask", [1] * len(ids))
            batch["attention_mask"].append(attn + [0] * pad)
            batch["labels"].append(f["labels"] + [-100] * pad)
        return {k: torch.tensor(v) for k, v in batch.items()}

REPOS = {
    "medical": f"{REPO.rsplit('/', 1)[0]}/llama-medical",
    "legal": f"{REPO.rsplit('/', 1)[0]}/llama-legal",
    "automotive": f"{REPO.rsplit('/', 1)[0]}/llama-automotive",
}


def _run_training(model, tokenizer, train_dataset, eval_dataset, args, repo_id, field_name):
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=_CompletionCollator(tokenizer),
    )

    print(f"Starting Llama {field_name.title()} Fine-Tuning with LORA...")
    trainer.train()

    model.push_to_hub(repo_id, token=TOKEN)
    tokenizer.push_to_hub(repo_id, token=TOKEN)

    print(f"LORA {field_name} model successfully pushed to {repo_id}")


# MEDICAL --------------->
def finetune_medical(output_dir: str) -> None:
    train_dataset, eval_dataset, model, tokenizer = medical()

    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_ratio=0.05,
        num_train_epochs=1,
        learning_rate=5e-5,
        lr_scheduler_type="cosine",

        fp16=False,
        bf16=True,

        logging_steps=10,
        weight_decay=0.01,
        seed=3407,
        push_to_hub=False,
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
    )

    _run_training(model, tokenizer, train_dataset, eval_dataset, args, REPOS["medical"], "medical")


# LEGAL --------------->
def finetune_legal(output_dir: str) -> None:
    train_dataset, eval_dataset, model, tokenizer = legal()

    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_ratio=0.05,
        num_train_epochs=1,
        learning_rate=5e-5,
        lr_scheduler_type="cosine",

        fp16=False,
        bf16=True,

        logging_steps=10,
        weight_decay=0.01,
        seed=3407,
        push_to_hub=False,
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
    )

    _run_training(model, tokenizer, train_dataset, eval_dataset, args, REPOS["legal"], "legal")


# AUTOMOTIVE --------------->
def finetune_automotive(output_dir: str) -> None:
    train_dataset, eval_dataset, model, tokenizer = automotive()

    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_ratio=0.05,
        num_train_epochs=1,
        learning_rate=5e-5,
        lr_scheduler_type="cosine",

        fp16=False,
        bf16=True,

        logging_steps=10,
        weight_decay=0.01,
        seed=3407,
        push_to_hub=False,
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
    )

    _run_training(model, tokenizer, train_dataset, eval_dataset, args, REPOS["automotive"], "automotive")


FINETUNERS = {
    "medical": finetune_medical,
    "legal": finetune_legal,
    "automotive": finetune_automotive,
}


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 finetune.py <domain> <output_dir>")
        print("  Example: python3 finetune.py medical /hhome/ps2g02/model_output_medical_v2")
        sys.exit(1)

    dominio = sys.argv[1]
    output_dir = sys.argv[2]

    FINETUNERS[dominio](output_dir)
