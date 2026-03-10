"""
Language Model service for generating answers based on retrieved context.

Uses Google's FLAN-T5-base, a text-to-text transformer model fine-tuned
for instruction following and question answering tasks.
"""

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Global variables to cache model and tokenizer
# This prevents reloading on every request, improving performance
model = None
tokenizer = None


def get_model_and_tokenizer():
    """
    Lazy-load the FLAN-T5 model and tokenizer.
    Loads only once and caches for subsequent requests.

    Returns:
        Tuple of (model, tokenizer)
    """
    global model, tokenizer

    if model is None or tokenizer is None:
        # Load FLAN-T5 base model (250M parameters)
        # For better answers, consider upgrading to flan-t5-large or flan-t5-xl
        tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
        model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

    return model, tokenizer


def generate_answer(question, docs):
    """
    Generate an answer to the question using the provided context documents.

    Uses a text-to-text generation model (FLAN-T5) to synthesize information
    from the retrieved documents into a coherent answer.

    Args:
        question: The user's question string
        docs: List of relevant document chunks to use as context

    Returns:
        Generated answer string, or a "not found" message if no docs provided
    """
    if not docs:
        return "No relevant information found."

    # Combine all document chunks into a single context
    context = "\n".join(docs)

    # Create instruction-following prompt for FLAN-T5
    prompt = f"""Use the context below to answer the question.

Context:
{context}

Question: {question}

Answer:"""

    # Get the cached model and tokenizer
    model, tokenizer = get_model_and_tokenizer()

    # Tokenize input with truncation to fit model's context window (512 tokens)
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)

    # Generate answer with controlled parameters
    # - max_length: Maximum tokens in the generated answer
    # - min_length: Prevent very short answers
    # - do_sample: False for deterministic/greedy decoding
    # - early_stopping: Stop when EOS token is generated
    outputs = model.generate(
        **inputs,
        max_length=150,
        min_length=10,
        do_sample=False,
        early_stopping=True
    )

    # Decode generated tokens back to text, removing special tokens
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    return answer
