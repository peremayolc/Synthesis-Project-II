#!/usr/bin/env python3
"""
compare_models.py <checkpoint_path>

Side-by-side comparison of a fine-tuned model vs its base model on:
  1. 26 hand-aligned Unitron Remote Plus (ENG/ESP PDF) pairs
  2. Auto-aligned pairs from Unitron TrueFit 5.6 PDFs
     en-US-guide-user.pdf  /  es-ES-guide-user.pdf

Usage:
    python3 compare_models.py /hhome/ps2g02/model_output/checkpoint-5500
"""

import sys, re, json, math, gc, torch
from pathlib import Path
from sacrebleu.metrics import BLEU, CHRF
from unsloth import FastLanguageModel
from config import TOKEN

# ── paths ───────────────────────────────────────────────────────────────────
PDF_PAIRS = {
    "Unitron TrueFit 5.6": (
        "/hhome/ps2g02/en-US-guide-user.pdf",
        "/hhome/ps2g02/es-ES-guide-user.pdf",
    ),
    "AQ JAM XC Pro R": (
        "/hhome/ps2g02/aq_jam_xc_pro_r_user_guide_en.PDF",
        "/hhome/ps2g02/aq_jam_xc_pro_r_user_guide_es.pdf",
    ),
    "Unitron Moxi V-312": (
        "/hhome/ps2g02/UH_UserGuide_Moxi-V312_92x125_EN.pdf",
        "/hhome/ps2g02/UH_UserGuide_Moxi-V312_92x125_ES.pdf",
    ),
}

PROMPTS = {
    "medical":     "Translate the following English medical text to Spanish:\n\nEnglish: {en}\nSpanish: ",
    "legal":       "Translate the following English legal text to Spanish using formal legal terminology:\n\nEnglish: {en}\nSpanish: ",
    "automotive":  "Translate the following English technical text to Spanish using precise industrial engineering, automation, and automotive terminology:\n\nEnglish: {en}\nSpanish: ",
}
PROMPT = PROMPTS["medical"]  # default; override with --domain flag or by editing directly

# ── hand-aligned pairs from Unitron Remote Plus PDFs ────────────────────────
REMOTE_PLUS_PAIRS = [
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


# ── PDF extraction ───────────────────────────────────────────────────────────
def extract_sentences(path: str, min_words: int = 8) -> list:
    import pdfplumber
    with pdfplumber.open(path) as pdf:
        full = "\n".join(page.extract_text() or "" for page in pdf.pages)
    sents = []
    for s in re.split(r"(?<=[.!?])\s+", full):
        s = " ".join(s.split())
        if len(s.split()) < min_words:
            continue
        # skip TOC lines (dense dots)
        if s.count(".") / max(len(s), 1) > 0.08:
            continue
        # skip lines starting with a digit (page numbers / TOC entries)
        if re.match(r"^\d", s):
            continue
        # skip UI label dumps where avg word length is very short
        words = s.split()
        if sum(len(w) for w in words) / max(len(words), 1) < 4.5:
            continue
        sents.append(s)
    return sents


# ── Gale-Church sentence aligner ────────────────────────────────────────────
_BEAD_PRIORS = {
    (1, 1): 0.89,
    (1, 2): 0.089,
    (2, 1): 0.089,
    (1, 0): 0.011,
    (0, 1): 0.011,
    (2, 2): 0.011,
}


def _match_cost(sl: int, tl: int, mean: float = 1.0, var: float = 6.8) -> float:
    if sl + tl == 0:
        return 0.0
    delta = tl - sl * mean
    return delta ** 2 / (var * (sl + tl))


def gale_church_1to1(src: list, tgt: list,
                     ratio_lo: float = 0.4, ratio_hi: float = 2.5) -> list:
    """
    Gale-Church DP alignment; returns only the 1:1 beads that pass the
    character-length ratio filter — good clean pairs for MT evaluation.
    """
    m, n = len(src), len(tgt)
    INF = float("inf")
    dp = [[INF] * (n + 1) for _ in range(m + 1)]
    bp = [[None] * (n + 1) for _ in range(m + 1)]
    dp[0][0] = 0.0

    for i in range(m + 1):
        for j in range(n + 1):
            if dp[i][j] == INF:
                continue
            for (bi, bj), prior in _BEAD_PRIORS.items():
                ni, nj = i + bi, j + bj
                if ni > m or nj > n:
                    continue
                sl = sum(len(src[i + k]) for k in range(bi))
                tl = sum(len(tgt[j + k]) for k in range(bj))
                cost = dp[i][j] - math.log(prior) + _match_cost(sl, tl)
                if cost < dp[ni][nj]:
                    dp[ni][nj] = cost
                    bp[ni][nj] = (i, j, bi, bj)

    alignment = []
    i, j = m, n
    while i > 0 or j > 0:
        if bp[i][j] is None:
            break
        pi, pj, bi, bj = bp[i][j]
        alignment.append((pi, pj, bi, bj))
        i, j = pi, pj
    alignment.reverse()

    pairs = []
    for si, tj, bi, bj in alignment:
        if bi == 1 and bj == 1:
            en, es = src[si], tgt[tj]
            if ratio_lo <= len(es) / max(len(en), 1) <= ratio_hi:
                pairs.append((en, es))
    return pairs


# ── model helpers ────────────────────────────────────────────────────────────
def _load_model(model_name: str, token: str = None):
    kw = dict(model_name=model_name, max_seq_length=2048, load_in_4bit=True)
    if token:
        kw["token"] = token
    model, tok = FastLanguageModel.from_pretrained(**kw)
    FastLanguageModel.for_inference(model)
    return model, tok


def _translate(model, tok, en: str, prompt_template: str = None) -> str:
    device = next(model.parameters()).device
    prompt = (prompt_template or PROMPT).format(en=en)
    inp = tok(prompt, return_tensors="pt", truncation=True, max_length=512)
    inp = {k: v.to(device) for k, v in inp.items()}
    with torch.no_grad():
        out = model.generate(
            **inp,
            max_new_tokens=256,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
            stop_strings=["\nEnglish:", "\n\n"],
            tokenizer=tok,
        )
    gen = tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)
    for stop in ["\nEnglish:", "\n\nEnglish:", "\n\n"]:
        if stop in gen:
            gen = gen.split(stop)[0]
            break
    return gen.strip()


def _translate_all(model, tok, pairs: list, prompt_template: str = None) -> list:
    hyps = []
    total = len(pairs)
    for idx, (en, _) in enumerate(pairs, 1):
        print(f"  [{idx}/{total}] translating...", end="\r", flush=True)
        hyps.append(_translate(model, tok, en, prompt_template))
    print(flush=True)
    return hyps


# ── evaluation & display ─────────────────────────────────────────────────────
def evaluate(pairs: list, base_hyps: list, ft_hyps: list,
             title: str, num_examples: int = 3) -> dict:
    refs = [es for _, es in pairs]
    base_bleu = BLEU().corpus_score(base_hyps, [refs])
    base_chrf = CHRF().corpus_score(base_hyps, [refs])
    ft_bleu   = BLEU().corpus_score(ft_hyps,   [refs])
    ft_chrf   = CHRF().corpus_score(ft_hyps,   [refs])

    print(f"\n{'=' * 68}", flush=True)
    print(f"  {title}  ({len(pairs)} pairs)", flush=True)
    print(f"{'=' * 68}", flush=True)

    for i in range(min(num_examples, len(pairs))):
        en, es_ref = pairs[i]
        print(f"\n[Example {i + 1}]", flush=True)
        print(f"  EN   : {en[:120]}", flush=True)
        print(f"  REF  : {es_ref[:120]}", flush=True)
        print(f"  BASE : {base_hyps[i][:120]}", flush=True)
        print(f"  FT   : {ft_hyps[i][:120]}", flush=True)

    print(f"\n  {'Model':<14}  BLEU    chrF", flush=True)
    print(f"  {'Base':<14}  {base_bleu.score:5.2f}   {base_chrf.score:5.2f}", flush=True)
    print(f"  {'Finetuned':<14}  {ft_bleu.score:5.2f}   {ft_chrf.score:5.2f}", flush=True)

    delta_bleu = ft_bleu.score - base_bleu.score
    delta_chrf = ft_chrf.score - base_chrf.score
    sign_b = "+" if delta_bleu >= 0 else ""
    sign_c = "+" if delta_chrf >= 0 else ""
    print(f"  {'Δ (FT - Base)':<14}  {sign_b}{delta_bleu:.2f}   {sign_c}{delta_chrf:.2f}", flush=True)

    return {
        "base": {"bleu": base_bleu.score, "chrf": base_chrf.score},
        "ft":   {"bleu": ft_bleu.score,   "chrf": ft_chrf.score},
    }


# ── per-document txt writer ──────────────────────────────────────────────────
def write_doc_file(out_path: Path, title: str, checkpoint: str,
                   domain: str, base_model_name: str,
                   pairs: list, base_hyps: list, ft_hyps: list,
                   scores: dict) -> None:
    """Write a self-contained translation file for one document."""
    sep = "=" * 68
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"{sep}\n")
        f.write(f"  Document  : {title}\n")
        f.write(f"  Checkpoint: {checkpoint}\n")
        f.write(f"  Domain    : {domain}\n")
        f.write(f"  Base model: {base_model_name}\n")
        f.write(f"  Pairs     : {len(pairs)}\n")
        f.write(f"{sep}\n\n")

        for i, ((en, ref), base_h, ft_h) in enumerate(
            zip(pairs, base_hyps, ft_hyps), 1
        ):
            f.write(f"[{i}]\n")
            f.write(f"  EN   : {en}\n")
            f.write(f"  REF  : {ref}\n")
            f.write(f"  BASE : {base_h}\n")
            f.write(f"  FT   : {ft_h}\n\n")

        f.write(f"{sep}\n")
        f.write("  METRICS\n")
        f.write(f"  Base BLEU : {scores['base']['bleu']:6.2f}    "
                f"Base chrF : {scores['base']['chrf']:6.2f}\n")
        f.write(f"  FT   BLEU : {scores['ft']['bleu']:6.2f}    "
                f"FT   chrF : {scores['ft']['chrf']:6.2f}\n")
        b = scores['ft']['bleu']  - scores['base']['bleu']
        c = scores['ft']['chrf']  - scores['base']['chrf']
        f.write(f"  Δ    BLEU : {b:+6.2f}    Δ    chrF : {c:+6.2f}\n")
        f.write(f"{sep}\n")


# ── main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        print("Usage: python3 compare_models.py <checkpoint_path> [domain]")
        print("  domain: medical (default) | legal | automotive")
        sys.exit(1)

    checkpoint = sys.argv[1]
    domain = sys.argv[2] if len(sys.argv) == 3 else "medical"
    if domain not in PROMPTS:
        print(f"Unknown domain '{domain}'. Choose from: {list(PROMPTS)}")
        sys.exit(1)
    PROMPT = PROMPTS[domain]

    adapter_cfg = Path(checkpoint) / "adapter_config.json"
    with open(adapter_cfg) as f:
        base_model_name = json.load(f)["base_model_name_or_path"]

    print(f"Base model : {base_model_name}", flush=True)
    print(f"Checkpoint : {checkpoint}", flush=True)

    # ── build datasets ────────────────────────────────────────────────────
    datasets = [("Unitron Remote Plus", REMOTE_PLUS_PAIRS)]

    for title, (en_pdf, es_pdf) in PDF_PAIRS.items():
        print(f"\nExtracting pairs from {title}...", flush=True)
        en_sents = extract_sentences(en_pdf)
        es_sents = extract_sentences(es_pdf)
        pairs = gale_church_1to1(en_sents, es_sents)
        print(f"  {len(pairs)} aligned pairs "
              f"(from {len(en_sents)} EN / {len(es_sents)} ES sentences)",
              flush=True)
        datasets.append((title, pairs))

    all_pairs = [p for _, ps in datasets for p in ps]

    # ── pass 1: base model ────────────────────────────────────────────────
    print("\nLoading base model...", flush=True)
    base_model, base_tok = _load_model(base_model_name)
    print(f"Translating {len(all_pairs)} pairs with base model...", flush=True)
    all_base_hyps = _translate_all(base_model, base_tok, all_pairs, PROMPT)
    del base_model, base_tok
    gc.collect()
    torch.cuda.empty_cache()

    # ── pass 2: finetuned model ───────────────────────────────────────────
    print("\nLoading finetuned model...", flush=True)
    ft_model, ft_tok = _load_model(checkpoint, token=TOKEN)
    print(f"Translating {len(all_pairs)} pairs with finetuned model...", flush=True)
    all_ft_hyps = _translate_all(ft_model, ft_tok, all_pairs, PROMPT)
    del ft_model, ft_tok
    gc.collect()
    torch.cuda.empty_cache()

    # ── evaluate, print, and write per-doc files ──────────────────────────
    offset = 0
    results = {}
    written = []
    for title, pairs in datasets:
        n = len(pairs)
        base_hyps = all_base_hyps[offset : offset + n]
        ft_hyps   = all_ft_hyps[offset : offset + n]

        scores = evaluate(pairs, base_hyps, ft_hyps, title)
        results[title] = scores

        slug = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")
        out_path = Path(checkpoint) / f"translations_{slug}.txt"
        write_doc_file(out_path, title, checkpoint, domain,
                       base_model_name, pairs, base_hyps, ft_hyps, scores)
        written.append(str(out_path))

        offset += n

    # ── summary table ─────────────────────────────────────────────────────
    print(f"\n{'=' * 68}", flush=True)
    print("  SUMMARY", flush=True)
    print(f"  {'Dataset':<25}  {'Base BLEU':>9}  {'FT BLEU':>7}  "
          f"{'Base chrF':>9}  {'FT chrF':>7}", flush=True)
    for title, r in results.items():
        print(
            f"  {title:<25}  {r['base']['bleu']:>9.2f}  {r['ft']['bleu']:>7.2f}"
            f"  {r['base']['chrf']:>9.2f}  {r['ft']['chrf']:>7.2f}",
            flush=True,
        )

    print(f"\nFiles written:", flush=True)
    for p in written:
        print(f"  {p}", flush=True)
