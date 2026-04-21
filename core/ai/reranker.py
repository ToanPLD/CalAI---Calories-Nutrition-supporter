from sentence_transformers import CrossEncoder

model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def rerank(query, results):

    pairs = [(query, str(r.payload)) for r in results]

    scores = model.predict(pairs)

    ranked = list(zip(scores, results))

    ranked.sort(key=lambda x: x[0], reverse=True)

    return [x[1] for x in ranked[:5]]