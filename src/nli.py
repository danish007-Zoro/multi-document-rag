from sentence_transformers import CrossEncoder


# ==================================================
# NLI MODEL LOADER
# ==================================================

def load_nli_model(
    model_name="cross-encoder/nli-deberta-v3-base"
):
    """
    Load the Natural Language Inference model.

    This must return a SentenceTransformers
    CrossEncoder, NOT a SentenceTransformer.

    The model predicts:

        contradiction
        entailment
        neutral
    """

    print(
        f"Loading NLI model: {model_name}"
    )

    model = CrossEncoder(
        model_name
    )

    return model