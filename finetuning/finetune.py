from imports import *
from config import TOKEN, REPO
from medical import medical
from legal import legal
from automotive import automotive


# general finetuning loop
def finetuning(field_name : str = "medical", output_dir: str = None) -> None:
    
    field_functions = {
        "medical": medical,
        "legal": legal,
        "automotive": automotive
    }
    
    field_function = field_functions[field_name]
    train_dataset, model, tokenizer = field_function()

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8
    )

    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=100,
        num_train_epochs=1,
        learning_rate=2e-4,
        fp16=torch.cuda.is_available(),
        logging_steps=10,
        weight_decay=0.01,
        seed=3407,
        push_to_hub=False,
        remove_unused_columns=False,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        max_seq_length=2048,
    )

    print(f"Starting Llama {field_name.title()} Fine-Tuning with LORA...")
    trainer.train()
    
    model.push_to_hub(REPO, token=TOKEN)
    tokenizer.push_to_hub(REPO, token=TOKEN)

    print(f"LORA {field_name} model successfully pushed to {REPO}")



if __name__ == "__main__":
    finetuning()