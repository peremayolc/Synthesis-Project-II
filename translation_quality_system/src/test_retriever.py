from retriever import Retriever

r = Retriever()

query = "The power cap is wrong in the Spanish translation"

results = r.retrieve(query, k=3)

for res in results:
    print("----")
    print("Score:", res["score"])
    print("Type:", res["type"])
    print("Text:", res["text"])