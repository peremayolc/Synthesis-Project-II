from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from retriever import Retriever


class LocalLLMCorrector:
    def __init__(self):
        print("Loading local model...")

        self.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
        self.model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")

        self.retriever = Retriever()

    def build_prompt(self, source_en, mt_es):
        retrieved = self.retriever.retrieve(source_en + " " + mt_es, k=3)
        context = "\n".join([r["text"] for r in retrieved])

        prompt = f"""
Fix this Spanish translation.

EN: {source_en}
BAD ES: {mt_es}

Rules:
{context}

Correct Spanish:
"""
        return prompt

    def correct(self, source_en, mt_es):
        prompt = self.build_prompt(source_en, mt_es)

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=100
        )

        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return {
            "corrected_es": text.strip(),
            "explanation": "Local seq2seq model"
        }