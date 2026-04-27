from imports import *
from config import TOKEN, REPO
from medical import medical


# general finetuning loop
def finetuning(field : str = "medical", output_dir: str = "llama-medical-lora-es") -> None:
    
    # this loads the appropiate dataset, model and tokenizer depending on the needed field.
    train_dataset, model, tokenizer = field()

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

    print("Starting Llama Medical Fine-Tuning with LORA...")
    trainer.train()

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    model.push_to_hub(REPO, token=TOKEN)
    tokenizer.push_to_hub(REPO, token=TOKEN)

    print(f"LORA model successfully pushed to {REPO}")



if __name__ == "__main__":
    finetuning()