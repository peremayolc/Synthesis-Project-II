import json
import os
from dotenv import load_dotenv
from openai import OpenAI

from retriever import Retriever

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class LLMCorrector:
    def __init__(self):
        self.retriever = Retriever()

    def build_prompt(self, source_en, mt_es):
        retrieved = self.retriever.retrieve(source_en + " " + mt_es, k=3)

        context = "\n\n".join([r["text"] for r in retrieved])

        prompt = f"""
You are an expert English-to-Spanish translation corrector.

Your job:
- Fix the Spanish translation.
- Use the rules provided.
- Preserve meaning exactly.
- Do NOT add information.
- Fix terminology, grammar, and style.

---

ENGLISH:
{source_en}

SPANISH (incorrect):
{mt_es}

---

RULES:
{context}

---

Return ONLY JSON:

{{
  "corrected_es": "...",
  "explanation": "what you fixed"
}}
"""
        return prompt

    def correct(self, source_en, mt_es):
        prompt = self.build_prompt(source_en, mt_es)

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
        )

        text = response.choices[0].message.content.strip()

        try:
            return json.loads(text)
        except:
            return {
                "corrected_es": mt_es,
                "explanation": "Failed to parse response"
            }