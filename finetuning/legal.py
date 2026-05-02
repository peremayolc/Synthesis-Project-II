from imports import *


# LEGAL --------------->
# loading the base model and tokenizer
def load_legal_model(model_name: str = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit", max_lenght: int = 2048, load_bit : bool = True):
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
    
    tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.chat_template is None:
        tokenizer.chat_template = "{{ bos_token }}{% for message in messages %}{{ message['content'] }}{{ eos_token }}{% endfor %}"

    return model, tokenizer

def preprocess_legal_function(examples, tokenizer, max_length: int = 512):
    formatted_texts = []
    
    for en_text, es_text in zip(examples["en"], examples["es"]):
        instruction = f"Translate the following English legal text to Spanish using formal EU legal terminology:\n\nEnglish: {en_text}\nSpanish: {es_text}"
        formatted_texts.append(instruction)
    
    model_inputs = tokenizer(
        formatted_texts,
        max_length=max_length,
        truncation=True,
        padding=False,  
        return_tensors=None
    )
    
    model_inputs["labels"] = model_inputs["input_ids"].copy()
    
    return model_inputs

def legal():
    model, tokenizer = load_legal_model()

    dataset = load_dataset("opus100", "en-es", split="train")
    
    legal_dataset = dataset.shuffle(seed=3407).select(range(50000))

    tokenized_dataset = legal_dataset.map(
        lambda examples: preprocess_legal_function(
            {
                "en": [ex["en"] for ex in examples["translation"]],
                "es": [ex["es"] for ex in examples["translation"]]
            }, 
            tokenizer
        ),
        batched=True,
        remove_columns=legal_dataset.column_names
    )

    split_dataset = tokenized_dataset.train_test_split(test_size=0.1, seed=3407)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]

    return train_dataset, eval_dataset, model, tokenizer
# ---------------
