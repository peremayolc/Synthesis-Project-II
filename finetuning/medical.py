from imports import *


# MEDICAL --------------->
# loading the base model and tokenizer
def load_nllb_model(model_name: str = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit", max_lenght: int = 2048, load_bit : bool = True):
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

def preprocess_function(examples, tokenizer, max_length: int = 512):
    formatted_texts = []
    
    for en_text, es_text in zip(examples["text_en"], examples["text_es"]):
        instruction = f"Translate the following English medical text to Spanish:\n\nEnglish: {en_text}\nSpanish: {es_text}"
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

def medical():

    model, tokenizer = load_nllb_model()

    dataset = load_dataset("SINAI/ALIA-parallel-translation", split="train")

    medical_dataset = dataset.filter(lambda x: str(x['id']).startswith("00"))

    tokenized_dataset = medical_dataset.map(
        lambda examples: preprocess_function(examples, tokenizer),
        batched=True,
        remove_columns=medical_dataset.column_names
    )

    return tokenized_dataset, model, tokenizer
# ---------------