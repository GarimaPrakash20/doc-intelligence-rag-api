from transformers import pipeline

generator = None

def get_generator():
    global generator
    if generator is None:
        generator = pipeline(
            task="text-generation",
            model="google/flan-t5-base",
            max_length=200
        )
    return generator


def generate_answer(question, docs):

    context = "\n".join(docs)

    prompt = f"""
    Answer the question using the context below.

    Context:
    {context}

    Question:
    {question}
    """

    gen = get_generator()

    result = gen(prompt)

    return result[0]["generated_text"]
