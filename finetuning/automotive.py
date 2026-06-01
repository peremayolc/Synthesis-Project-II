from imports import *
import torch
from datasets import load_dataset
from unsloth import FastLanguageModel

# AUTOMOTIVE --------------->
# loading the base model and tokenizer
def load_automotive_model(model_name: str = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit", max_lenght: int = 2048, load_bit : bool = True):
    max_seq_length = 2048

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name,
        max_seq_length = max_seq_length,
        load_in_4bit = load_bit,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r = 8,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"],
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


def _load_clean_automotive_dataset():
    dataset = load_dataset("un_pc", "en-es", split="train")
    # Skip noisy first rows; sample a different slice than legal (600k–1200k)
    # to avoid overlap and get a varied technical/regulatory section of the corpus.
    dataset = dataset.select(range(600000, 1200000))

    def is_clean(example):
        en = example["translation"]["en"]
        es = example["translation"]["es"]
        en_words = len(en.split())
        es_words = len(es.split())
        if not (10 <= en_words <= 120):
            return False
        ratio = es_words / max(en_words, 1)
        if not (0.5 <= ratio <= 2.0):
            return False
        digit_ratio = sum(c.isdigit() for c in en) / max(len(en), 1)
        if digit_ratio > 0.15:
            return False
        return True

    return dataset.filter(is_clean, num_proc=4)


def preprocess_automotive_function(examples, tokenizer, max_length: int = 512):
    input_ids_list, attention_mask_list, labels_list = [], [], []
    for en_text, es_text in zip(examples["en"], examples["es"]):
        prompt = f"Translate the following English technical text to Spanish using precise industrial engineering, automation, and automotive terminology:\n\nEnglish: {en_text}\nSpanish: "
        full_text = prompt + es_text + tokenizer.eos_token

        prompt_len = len(tokenizer(prompt, add_special_tokens=True)["input_ids"])
        enc = tokenizer(full_text, truncation=True, max_length=max_length, add_special_tokens=True)

        input_ids = enc["input_ids"]
        labels = [-100] * prompt_len + input_ids[prompt_len:]
        labels = labels[:len(input_ids)]

        input_ids_list.append(input_ids)
        attention_mask_list.append(enc["attention_mask"])
        labels_list.append(labels)

    return {"input_ids": input_ids_list, "attention_mask": attention_mask_list, "labels": labels_list}

def automotive():
    model, tokenizer = load_automotive_model()

    dataset = _load_clean_automotive_dataset()
    automotive_dataset = dataset.shuffle(seed=3407).select(range(50000))

    tokenized_dataset = automotive_dataset.map(
        lambda examples: preprocess_automotive_function(
            {
                "en": [ex["en"] for ex in examples["translation"]],
                "es": [ex["es"] for ex in examples["translation"]]
            },
            tokenizer
        ),
        batched=True,
        remove_columns=automotive_dataset.column_names,
        load_from_cache_file=False,
    )

    split_dataset = tokenized_dataset.train_test_split(test_size=0.1, seed=3407)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]

    return train_dataset, eval_dataset, model, tokenizer


def evaluate_automotive(checkpoint_path: str, num_samples: int = 200):
    from sacrebleu.metrics import BLEU, CHRF

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=checkpoint_path,
        max_seq_length=2048,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    device = next(model.parameters()).device

    dataset = _load_clean_automotive_dataset()
    shuffled = dataset.shuffle(seed=3407)

    start = 50000
    end = start + num_samples
    if end > len(shuffled):
        raise ValueError(
            f"Not enough unseen samples: dataset has {len(shuffled)} total, "
            f"training used the first 50000, only {len(shuffled) - start} remain."
        )
    test_data = shuffled.select(range(start, end))

    references, hypotheses = [], []
    for example in test_data:
        en_text = example["translation"]["en"]
        es_text = example["translation"]["es"]
        prompt = (
            f"Translate the following English technical text to Spanish using precise industrial engineering, automation, and automotive terminology:\n\n"
            f"English: {en_text}\nSpanish: "
        )
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                stop_strings=["\nEnglish:", "\n\n"],
                tokenizer=tokenizer,
            )
        generated = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        for stop in ["\nEnglish:", "\n\nEnglish:", "\n\n"]:
            if stop in generated:
                generated = generated.split(stop)[0]
                break
        hypotheses.append(generated.strip())
        references.append(es_text)

    import sys, traceback
    try:
        bleu_score = BLEU().corpus_score(hypotheses, [references])
        chrf_score = CHRF().corpus_score(hypotheses, [references])
    except Exception:
        traceback.print_exc()
        sys.exit(1)

    print(f"\n=== Automotive Evaluation — {num_samples} unseen samples ===", flush=True)
    print(f"Checkpoint: {checkpoint_path}", flush=True)
    for i in range(min(3, len(hypotheses))):
        en_text = test_data[i]["translation"]["en"]
        print(f"\n[Example {i + 1}]", flush=True)
        print(f"  EN : {en_text[:120]}", flush=True)
        print(f"  REF: {references[i][:120]}", flush=True)
        print(f"  GEN: {hypotheses[i][:120]}", flush=True)
    print(f"\nBLEU : {bleu_score.score:.2f}", flush=True)
    print(f"chrF : {chrf_score.score:.2f}", flush=True)

    return {"bleu": bleu_score.score, "chrf": chrf_score.score}
# ---------------
