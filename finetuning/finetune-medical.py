from finetuning.imports import *
from config import TOKEN, REPO


# MEDICAL --------------->

# loading the base model and tokenizer for the llama 8B model.
def load_ini_model(model_name: str = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit", max_lenght: int = 2048, load_bit : bool = True):
    max_seq_length = 2048

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name,
        max_seq_length = max_seq_length,
        load_in_4bit = load_bit,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r = 16, 
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 16,
        lora_dropout = 0, 
        bias = "none",    
        use_gradient_checkpointing = "unsloth", 
        random_state = 3407,
    )

    return model, tokenizer

def dataset_medical(path: str, tokenizer):
    dataset = load_dataset("SINAI/ALIA-parallel-translation", split="train", streaming=True)

    medical_dataset = dataset.filter(lambda x: x['id'].startswith("00"))

    translation_prompt = """### Instruction:
    Translate the following medical text from English to Spanish, maintaining clinical accuracy.

    ### Input:
    {}

    ### Response:
    {}"""

    def formatting_prompts_func(examples):
        inputs  = examples["text_en"]
        outputs = examples["text_es"]
        texts = []
        for input, output in zip(inputs, outputs):
            text = translation_prompt.format(input, output) + tokenizer.eos_token
            texts.append(text)
        return { "text" : texts, }
        
    train_data = [next(iter(medical_dataset)) for _ in range(len(medical_dataset))]

    return train_data

# general medical finetuning loop
def finetuning_medical() -> None:

    model, tokenizer = load_ini_model()
    train_dataset = dataset_medical(tokenizer=tokenizer)

    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = train_dataset,
        dataset_text_field = "text",
        max_seq_length = 2048,
        dataset_num_proc = 2,
        packing = False, 
        args = TrainingArguments(
            per_device_train_batch_size = 2,
            gradient_accumulation_steps = 4,
            warmup_steps = 5,
            num_epochs = 2, 
            learning_rate = 2e-4,
            fp16 = not torch.cuda.is_bf16_supported(),
            bf16 = torch.cuda.is_bf16_supported(),
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            output_dir = "outputs",
        ),
    )

    trainer.train()

    model.push_to_hub_merged(
        REPO, 
        tokenizer, 
        save_method = "merged_16bit", 
        token = TOKEN
    )

    print(f"model successfully pushed to {REPO}")
    
    return None