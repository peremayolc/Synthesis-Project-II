from pipeline import TranslationPipeline

pipeline = TranslationPipeline()

examples = [
    (
        "Press the mute button below the slider to mute or unmute the hearing aids.",
        "Presionar el botón mute debajo del slider para mudo o inmuto los instrumentos auditivos.",
        "Pulse el botón silenciar situado debajo del control deslizante para silenciar o activar el sonido de los audífonos.",
    ),
    (
        "The Unitron Remote Plus app is compatible with iOS Version 12 or newer.",
        "La aplicación Unitron Remote Plus es compatible con iOS versión 16 o posterior.",
        "La aplicación Unitron Remote Plus es compatible con iOS versión 12 o posterior.",
    ),
    (
        "These hearing aids should never be exposed to chlorinated water, soap, salt water or other liquids with a chemical content.",
        "Estos audífonos pueden exponerse a agua clorada.",
        "Estos audífonos no deben exponerse nunca a agua clorada, jabón, agua salada ni a otros líquidos con contenido químico.",
    ),
]

for en, es, ref in examples:
    result = pipeline.process(en, es, reference_es=ref, document="demo", segment_id="x", variant="MT")
    print("====")
    print("EN:", en)
    print("MT:", es)
    print("Corrected:", result["corrected_es"])
    print("Decision:", result["decision"])
    print("Issues:", result["issues"])
