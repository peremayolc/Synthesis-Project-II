from imports import *
from config import TOKEN, REPO
import json
from datetime import datetime
from huggingface_hub import HfApi


def verify_model_on_hub(repo_id: str = REPO) -> bool:
    """Check the finetuned model exists and has the expected files on HuggingFace Hub."""
    api = HfApi(token=TOKEN)
    try:
        info = api.repo_info(repo_id=repo_id, repo_type="model")
        files = [f.rfilename for f in api.list_repo_files(repo_id=repo_id, token=TOKEN)]
        print(f"Model found on Hub: {repo_id}")
        print(f"Last modified: {info.lastModified}")
        required = ["config.json", "tokenizer_config.json"]
        missing = [f for f in required if not any(f in file for file in files)]
        if missing:
            print(f"WARNING: missing expected files: {missing}")
            return False
        print(f"Required files present: {required}")
        return True
    except Exception as e:
        print(f"Model NOT found on Hub ({repo_id}): {e}")
        return False


def load_finetuned_model(repo_id: str = REPO):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=repo_id,
        max_seq_length=2048,
        load_in_4bit=True,
        token=TOKEN
    )
    model = FastLanguageModel.for_inference(model)
    return model, tokenizer


def get_test_dataset(num_samples: int = 500, start_idx: int = 50000):
    # Starts at 50000 to avoid overlap with the 50k training samples
    dataset = load_dataset("SINAI/ALIA-parallel-translation", split="train")
    medical_dataset = dataset.filter(lambda x: str(x['id']).startswith("00"))
    test_samples = medical_dataset.select(range(start_idx, min(start_idx + num_samples, len(medical_dataset))))
    return test_samples


def calculate_token_accuracy(reference: str, prediction: str, tokenizer) -> float:
    ref_tokens = set(tokenizer.tokenize(reference.lower()))
    pred_tokens = set(tokenizer.tokenize(prediction.lower()))
    if not ref_tokens:
        return 0.0
    correct = len(ref_tokens.intersection(pred_tokens))
    return (correct / len(ref_tokens)) * 100


def calculate_bleu(references: list, predictions: list) -> float:
    try:
        from sacrebleu.metrics import BLEU
        bleu = BLEU(effective_order=True)
        return bleu.corpus_score(predictions, [references]).score
    except ImportError:
        print("sacrebleu not installed — skipping BLEU. Run: pip install sacrebleu")
        return -1.0


def generate_translation(model, tokenizer, english_text: str) -> str:
    prompt = f"Translate the following English medical text to Spanish:\n\nEnglish: {english_text}\nSpanish:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        top_p=0.95,
    )

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return result.split("Spanish:")[-1].strip() if "Spanish:" in result else result


def test_medical(repo_id: str = REPO, num_samples: int = 100):
    """Test finetuned medical model on fresh test data."""
    print(f"\n{'='*70}")
    print(f"Medical Model Evaluation")
    print(f"Model: {repo_id}")
    print(f"{'='*70}\n")

    print("Verifying model on HuggingFace Hub...")
    if not verify_model_on_hub(repo_id):
        print("Aborting: model not properly saved to Hub.")
        return []
    print()

    model, tokenizer = load_finetuned_model(repo_id)
    test_data = get_test_dataset(num_samples=num_samples)

    predictions = []
    token_accuracies = []
    references_list = []
    predictions_list = []
    print(f"Testing on {len(test_data)} samples...\n")

    for idx, sample in enumerate(test_data):
        en_text = sample["text_en"]
        es_ref = sample["text_es"]
        es_pred = generate_translation(model, tokenizer, en_text)
        token_acc = calculate_token_accuracy(es_ref, es_pred, tokenizer)

        predictions.append({
            "english": en_text,
            "reference": es_ref,
            "prediction": es_pred,
            "token_accuracy": round(token_acc, 2),
        })
        token_accuracies.append(token_acc)
        references_list.append(es_ref)
        predictions_list.append(es_pred)

        if (idx + 1) % 50 == 0:
            print(f"Processed {idx + 1}/{len(test_data)} samples")

    avg_token_accuracy = sum(token_accuracies) / len(token_accuracies) if token_accuracies else 0
    bleu_score = calculate_bleu(references_list, predictions_list)

    metrics = {
        "avg_token_accuracy": round(avg_token_accuracy, 2),
        "bleu_score": round(bleu_score, 2) if bleu_score >= 0 else "N/A",
    }

    output_file = f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({"metrics": metrics, "predictions": predictions}, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*70}")
    print(f"Results saved to:       {output_file}")
    print(f"Samples tested:         {len(predictions)}")
    print(f"Average Token Accuracy: {avg_token_accuracy:.2f}%")
    if bleu_score >= 0:
        print(f"BLEU Score:             {bleu_score:.2f}")
    print(f"{'='*70}\n")

    for i, pred in enumerate(predictions[:3], 1):
        print(f"\n[Sample {i}] Token Accuracy: {pred['token_accuracy']}%")
        print(f"English:     {pred['english'][:80]}...")
        print(f"Reference:   {pred['reference'][:80]}...")
        print(f"Predicted:   {pred['prediction'][:80]}...")

    return predictions


if __name__ == "__main__":
    test_medical(repo_id=REPO, num_samples=10000)
