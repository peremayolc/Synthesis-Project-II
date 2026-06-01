from imports import *
from config import TOKEN

# MEDICAL --------------->
# loading the base model and tokenizer
def load_medical_model(model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct", max_lenght: int = 2048, load_bit : bool = True):
    max_seq_length = 2048

    model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = max_seq_length,
    load_in_4bit = load_bit,
    token=TOKEN
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

def preprocess_function(examples, tokenizer, max_length: int = 512):
    input_ids_list, attention_mask_list, labels_list = [], [], []
    for en_text, es_text in zip(examples["text_en"], examples["text_es"]):
        prompt = f"Translate the following English medical text to Spanish:\n\nEnglish: {en_text}\nSpanish: "
        full_text = prompt + es_text + tokenizer.eos_token

        prompt_len = len(tokenizer(prompt, add_special_tokens=True)["input_ids"])
        enc = tokenizer(full_text, truncation=True, max_length=max_length, add_special_tokens=True)

        input_ids = enc["input_ids"]
        labels = [-100] * prompt_len + input_ids[prompt_len:]
        labels = labels[:len(input_ids)]  # guard if full_text was truncated

        input_ids_list.append(input_ids)
        attention_mask_list.append(enc["attention_mask"])
        labels_list.append(labels)

    return {"input_ids": input_ids_list, "attention_mask": attention_mask_list, "labels": labels_list}

def medical():

    model, tokenizer = load_medical_model()

    dataset = load_dataset("SINAI/ALIA-parallel-translation", split="train")

    medical_dataset = dataset.filter(lambda x: str(x['id']).startswith("00"))

    medical_dataset = medical_dataset.shuffle(seed=42).select(range(50000))

    split_dataset = medical_dataset.train_test_split(test_size=0.05)
    
    tokenized_train = split_dataset["train"].map(
        lambda examples: preprocess_function(examples, tokenizer),
        batched=True,
        remove_columns=medical_dataset.column_names,
        num_proc=4,
        load_from_cache_file=False,
    )
    tokenized_val = split_dataset["test"].map(
        lambda examples: preprocess_function(examples, tokenizer),
        batched=True,
        remove_columns=medical_dataset.column_names,
        num_proc=4,
        load_from_cache_file=False,
    )

    return tokenized_train, tokenized_val, model, tokenizer


def evaluate_medical(checkpoint_path: str, num_samples: int = 200):
    from sacrebleu.metrics import BLEU, CHRF

    # Load the saved adapter on top of the base model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=checkpoint_path,
        max_seq_length=2048,
        load_in_4bit=True,
        token=TOKEN,
    )
    FastLanguageModel.for_inference(model)
    device = next(model.parameters()).device

    # Recreate the exact same shuffle used in training so we can take samples beyond
    # the first 50k, which are guaranteed not to have been seen during training
    dataset = load_dataset("SINAI/ALIA-parallel-translation", split="train")
    medical_dataset = dataset.filter(lambda x: str(x["id"]).startswith("00"))
    shuffled = medical_dataset.shuffle(seed=42)

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
        prompt = (
            f"Translate the following English medical text to Spanish:\n\n"
            f"English: {example['text_en']}\nSpanish: "
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
        references.append(example["text_es"])

    import sys, traceback
    try:
        bleu_score = BLEU().corpus_score(hypotheses, [references])
        chrf_score = CHRF().corpus_score(hypotheses, [references])
    except Exception:
        traceback.print_exc()
        sys.exit(1)

    print(f"\n=== Medical Evaluation — {num_samples} unseen samples ===", flush=True)
    print(f"Checkpoint: {checkpoint_path}", flush=True)
    for i in range(min(3, len(hypotheses))):
        print(f"\n[Example {i + 1}]", flush=True)
        print(f"  EN : {test_data[i]['text_en'][:120]}", flush=True)
        print(f"  REF: {references[i][:120]}", flush=True)
        print(f"  GEN: {hypotheses[i][:120]}", flush=True)
    print(f"\nBLEU : {bleu_score.score:.2f}", flush=True)
    print(f"chrF : {chrf_score.score:.2f}", flush=True)

    return {"bleu": bleu_score.score, "chrf": chrf_score.score}


# Out-of-domain check: same model evaluated on general (non-medical) parallel
# text with a neutral prompt. Used to compare base vs medical-finetuned model
# and detect catastrophic forgetting of general translation ability.
def evaluate_general(checkpoint_path: str, num_samples: int = 200):
    from sacrebleu.metrics import BLEU, CHRF

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=checkpoint_path,
        max_seq_length=2048,
        load_in_4bit=True,
        token=TOKEN,
    )
    FastLanguageModel.for_inference(model)
    device = next(model.parameters()).device

    dataset = load_dataset("opus100", "en-es", split="train")
    shuffled = dataset.shuffle(seed=99)
    test_data = shuffled.select(range(num_samples))

    references, hypotheses = [], []
    for example in test_data:
        en_text = example["translation"]["en"]
        es_text = example["translation"]["es"]
        prompt = (
            f"Translate the following English text to Spanish:\n\n"
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

    print(f"\n=== General (opus100) Evaluation — {num_samples} samples ===", flush=True)
    print(f"Checkpoint: {checkpoint_path}", flush=True)
    for i in range(min(3, len(hypotheses))):
        print(f"\n[Example {i + 1}]", flush=True)
        print(f"  EN : {test_data[i]['translation']['en'][:120]}", flush=True)
        print(f"  REF: {references[i][:120]}", flush=True)
        print(f"  GEN: {hypotheses[i][:120]}", flush=True)
    print(f"\nBLEU : {bleu_score.score:.2f}", flush=True)
    print(f"chrF : {chrf_score.score:.2f}", flush=True)

    return {"bleu": bleu_score.score, "chrf": chrf_score.score}
# ---------------


def test_documents(document, model) -> dict:
    """
    Evaluate the finetuned EN→ES translation model on hand-aligned sentence pairs
    extracted from the Unitron Remote Plus PDF user guide (ENG and ESP editions).

    The 26 pairs below were verified by reading both PDFs and identifying segments
    where the English and Spanish carry the same meaning, despite the documents
    being slightly different version numbers (EN: v2021, ES: v5.4).

    Args:
        document: unused — kept for interface consistency.  Pass anything (e.g. None).
        model:    checkpoint path for the finetuned FastLanguageModel adapter.

    Returns:
        dict with BLEU, chrF, number of pairs, and per-pair details.
    """
    from sacrebleu.metrics import BLEU, CHRF

    # --- 1. Hand-curated EN↔ES pairs from the Unitron Remote Plus PDFs --------
    _PAIRS = [
        (
            "Unitron Bluetooth wireless hearing aids are required to use the Unitron Remote Plus app. "
            "The Unitron Remote Plus app can be used on devices with Bluetooth® Low-Energy (BT-LE) "
            "capability and is compatible with iOS Version 12 or newer.",
            "Para poder utilizar la aplicación Unitron Remote Plus se necesitan audífonos Unitron "
            "con conexión inalámbrica Bluetooth®. La aplicación Unitron Remote Plus se puede "
            "utilizar en dispositivos con Bluetooth® de baja energía (Low-Energy, BT-LE) y es "
            "compatible con iOS versión 16 o posterior.",
        ),
        (
            "The Unitron Remote Plus app can be used on Google Mobile Services (GMS) certified Android "
            "devices supporting Bluetooth 4.2 and Android OS 7 or newer.",
            "Unitron Remote Plus app se puede utilizar en dispositivos Android certificados por Google "
            "Mobile Services (GMS) que admitan Bluetooth® 4.2 y Android OS 10.0 o posterior.",
        ),
        (
            "Some phones have touch sounds or keypad tones, which could be streamed to the hearing aid(s). "
            "To avoid this, go to your phone settings, select sounds and make sure that all touch sounds "
            "and keypad tones are deactivated.",
            "Algunos teléfonos tienen sonidos de pulsación o tonos de teclado que podrían "
            "transmitirse a los audífonos. Para evitarlo, vaya a los ajustes de su teléfono y, en "
            "la sección de sonidos, asegúrese de desactivar todos los sonidos de pulsación y "
            "los tonos del teclado.",
        ),
        (
            "The features available in the Unitron Remote Plus app vary depending on the hearing aids "
            "connected. Not all features are available for all hearing aids.",
            "Las funciones disponibles en la aplicación Unitron Remote Plus varían en función "
            "de los audífonos conectados. No todas las funciones están disponibles para todos "
            "los audífonos.",
        ),
        (
            "You can always select the \"demo\" mode to try the app without connecting a Unitron hearing "
            "aid and get a first impression of the functionalities. In this mode no remote control "
            "functionality is available for your hearing aids.",
            "Seleccione \"Modo de demostración\" para probar la aplicación sin conectar los "
            "audífonos. Tenga en cuenta que en este modo no está disponible la función "
            "de control remoto.",
        ),
        (
            "Move the slider up or down to increase or decrease the hearing aid volume on both sides.",
            "Mueva el ajuste de volumen subir o bajar el volumen del audífono en ambos oídos.",
        ),
        (
            "Press the \"mute\" button below the slider to mute or unmute the hearing aids.",
            "Pulse el botón silenciar para silenciar los audífonos o activar el sonido.",
        ),
        (
            "Press the \"split volume\" button to control the volume on each hearing aid separately. "
            "Use the volume slider to change the volume. Press the \"join volume\" button to merge "
            "the volume sliders.",
            "Pulse el botón separar volumen para controlar el volumen de cada audífono por separado. "
            "Pulse el botón unir volumen para combinar los ajustes de volumen.",
        ),
        (
            "For the Automatic Program, \"Clarity\" is available to enhance speech, whereas \"Comfort\" "
            "is used to reduce noise to improve overall listening comfort. Clarity and Comfort are "
            "mutually exclusive, and cannot both be in the 'On' state at the same time.",
            "En el programa Automático, puede seleccionar entre Normal, Comodidad o Claridad. "
            "Claridad sirve para realzar el habla, mientras que Confort se utiliza para reducir el ruido "
            "y mejorar el confort general de escucha. Claridad y Comodidad son mutuamente excluyentes y "
            "no pueden estar activados al mismo tiempo.",
        ),
        (
            "If you opted in for the Insights feature, you will see a happy face icon on the right side "
            "of the main screen. Tap on it to send feedback to your clinician.",
            "Al activar Percepciones y habilitar Calificaciones, puede tocar el icono de Calificaciones "
            "para compartir información sobre su experiencia auditiva con su profesional de salud "
            "auditiva.",
        ),
        (
            "Further adjustments are available depending on the program currently selected, your hearing "
            "aid configuration, and connected audio sources (e.g. TV Connector). Tap the advanced "
            "features button at the bottom-right corner to access these options:",
            "En función del programa seleccionado, de la configuración de los audífonos y de "
            "las fuentes de audio conectadas, es posible que se puedan realizar otros ajustes. Acceda a "
            "estos ajustes en la pantalla de Inicio, tocando el botón funciones avanzadas ubicado en "
            "la esquina superior derecha del panel del programa para los ajustes más avanzados.",
        ),
        (
            "If you use an external streaming device, (e.g. TV Connector, music) you can adjust the "
            "focus to hear more of the streamed signal or alternatively more of the surrounding environment.",
            "Si utiliza un dispositivo de transmisión externo (p. ej., TV Connector, música), "
            "puede ajustar el balance para escuchar más la señal transmitida o el ambiente "
            "circundante.",
        ),
        (
            "If you have tinnitus, and have been instructed by your hearing care professional on how to "
            "use the Tinnitus Masker, you can adjust the volume of the masking noise.",
            "Si su profesional de salud auditiva ha habilitado el enmascarador de tinnitus, dispondrá "
            "de una opción para ajustar el volumen del ruido de enmascaramiento.",
        ),
        (
            "The \"Reduce Noise\" control allows you to increase or reduce the level of noise to the "
            "desired comfort level.",
            "El control Reducir el ruido permite aumentar o reducir el nivel de ruido hasta el nivel de "
            "comodidad deseado.",
        ),
        (
            "The \"Enhance Speech\" control allows you to increase or reduce the focus on speech to the "
            "desired comfort level.",
            "El control Realzar habla permite realzar o reducir la voz hasta el nivel o de confort deseado.",
        ),
        (
            "You can adjust the \"Focus Mic\" control to focus more on sounds from the front or listen "
            "all around you.",
            "Puede ajustar el control Enfocar micrófono para centrarse más en los sonidos del "
            "frente o escuchar todo a su alrededor.",
        ),
        (
            "The app is available in different languages. It will automatically match the language of the "
            "phone's operating system. If the phone's language is not supported, the default language "
            "is English.",
            "La aplicación está disponible en varios idiomas. Se adaptará automáticamente "
            "al idioma del sistema operativo del teléfono, si dicho idioma es compatible. Si el idioma "
            "del teléfono no es compatible, el idioma predeterminado es el inglés.",
        ),
        (
            "Hereby Sonova AG declares that this Unitron product is in compliance with the essential "
            "requirements of the Medical Devices Directive 93/42/EEC.",
            "Por la presente, Sonova AG declara que este producto cumple con los requisitos de la "
            "Regulación sobre productos sanitarios (UE) 2017/745.",
        ),
        (
            "If the hearing aids do not respond to the device because of an unusual field disturbance, "
            "move away from the disturbing field.",
            "Si los audífonos no responden al dispositivo debido a una perturbación inusual del "
            "campo, aléjese del campo perturbador.",
        ),
        (
            "Instructions are available at: unitron.com/appguide in Adobe® Acrobat® PDF format. "
            "To view them, you must have Adobe Acrobat Reader installed. Visit Adobe.com to download.",
            "Puede encontrar las instrucciones en: unitron.com/appguide, en formato Adobe® Acrobat® "
            "PDF. Para verlas, debe tener instalado Adobe Acrobat Reader. Visite Adobe.com para descargarlo.",
        ),
        (
            "To obtain a free paper copy of the instructions, please contact your local Unitron "
            "representative. A copy will be sent within 7 days.",
            "Para obtener una copia impresa gratuita de las instrucciones, comúníquese con su "
            "representante local de Unitron. Se le enviará una copia en 7 días.",
        ),
        (
            "With the CE symbol, Sonova AG confirms that this Unitron product – including accessories "
            "– meets the requirements of the Medical Devices Directive 93/42/EEC.",
            "Con el símbolo CE, Sonova AG declara que este producto cumple con los requisitos de la "
            "Regulación sobre productos sanitarios (UE) 2017/745.",
        ),
        (
            "This symbol indicates that it is important for the user to read and take into account the "
            "relevant information in this user guide.",
            "Este símbolo indica que es importante que el usuario lea y tenga en cuenta la "
            "información relevante de estas guías del usuario.",
        ),
        (
            "This symbol indicates that it is important for the user to pay attention to the relevant "
            "warning notices in this user guide.",
            "Este símbolo indica que es importante que el usuario preste atención a los avisos de "
            "advertencia relevantes de esta guía del usuario.",
        ),
        (
            "Indicates the Authorized representative in the European Community. The EC REP is also the "
            "importer to the European Union.",
            "Indica el representante autorizado de la Unión Europea. El EU REP también es el "
            "importador a la Unión Europea.",
        ),
        (
            "The Bluetooth® word mark and logos are registered trademarks owned by Bluetooth SIG, Inc. "
            "and any use of such marks by Unitron is under license. Other trademarks and trade names are "
            "those of their respective owners.",
            "La marca Bluetooth® y sus logotipos son marcas comerciales registradas por Bluetooth® "
            "SIG, Inc., y cualquier uso que Sonova AG realice de dichas marcas será bajo licencia. "
            "Las demás marcas y nombres comerciales pertenecen a sus respectivos propietarios.",
        ),
    ]

    # --- 2. Load the finetuned model -----------------------------------------
    ft_model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model,
        max_seq_length=2048,
        load_in_4bit=True,
        token=TOKEN,
    )
    FastLanguageModel.for_inference(ft_model)
    device = next(ft_model.parameters()).device

    # --- 3. Translate each EN fragment and collect hypotheses ----------------
    references: list = []
    hypotheses: list = []
    per_pair: list = []

    for en_text, es_ref in _PAIRS:
        prompt = (
            "Translate the following English medical text to Spanish:\n\n"
            f"English: {en_text}\nSpanish: "
        )
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            out_ids = ft_model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                stop_strings=["\nEnglish:", "\n\n"],
                tokenizer=tokenizer,
            )

        generated = tokenizer.decode(
            out_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        for stop in ["\nEnglish:", "\n\nEnglish:", "\n\n"]:
            if stop in generated:
                generated = generated.split(stop)[0]
                break
        hypothesis = generated.strip()

        hypotheses.append(hypothesis)
        references.append(es_ref)
        per_pair.append({"en": en_text, "reference": es_ref, "hypothesis": hypothesis})

    # --- 4. Compute corpus-level metrics -------------------------------------
    bleu = BLEU().corpus_score(hypotheses, [references])
    chrf = CHRF().corpus_score(hypotheses, [references])

    print(f"\n=== Document Evaluation ({len(_PAIRS)} hand-aligned pairs) ===", flush=True)
    print(f"Checkpoint : {model}", flush=True)
    for i, pair in enumerate(per_pair[:3]):
        print(f"\n[Example {i + 1}]", flush=True)
        print(f"  EN  : {pair['en'][:120]}", flush=True)
        print(f"  REF : {pair['reference'][:120]}", flush=True)
        print(f"  GEN : {pair['hypothesis'][:120]}", flush=True)
    print(f"\nBLEU : {bleu.score:.2f}", flush=True)
    print(f"chrF : {chrf.score:.2f}", flush=True)

    info = {
        "bleu": bleu.score,
        "chrf": chrf.score,
        "num_pairs": len(_PAIRS),
        "pairs": per_pair,
    }
    return info