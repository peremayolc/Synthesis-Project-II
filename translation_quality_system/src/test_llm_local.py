from llm_corrector_local import LocalLLMCorrector

corrector = LocalLLMCorrector()

source = "The power cap is 200 W"
mt = "La tapa de potencia es 200 W"

result = corrector.correct(source, mt)

print(result)