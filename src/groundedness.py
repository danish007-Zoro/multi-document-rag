import re
import numpy as np

from nli import load_nli_model


# ==================================================
# CONFIGURATION
# ==================================================

DEFAULT_NLI_MODEL = (
    "cross-encoder/nli-deberta-v3-base"
)


# ==================================================
# GLOBAL NLI MODEL
# ==================================================

_NLI_MODEL = None


# ==================================================
# CLAIM / SENTENCE SPLITTING
# ==================================================

def split_into_claims(text):
    """
    Split generated text into individual claims.

    A lightweight sentence splitter is used so that
    each claim can be evaluated independently.
    """

    if text is None:
        return []

    text = str(text).strip()

    if not text:
        return []

    # Normalize whitespace
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    # Split on sentence-ending punctuation
    claims = re.split(
        r"(?<=[.!?])\s+",
        text
    )

    claims = [
        claim.strip()
        for claim in claims
        if claim.strip()
    ]

    return claims


# ==================================================
# NLI MODEL
# ==================================================

def get_nli_model():
    """
    Load the NLI CrossEncoder once and reuse it.

    The RAG embedding model is a SentenceTransformer.
    The groundedness model is a CrossEncoder.
    """

    global _NLI_MODEL

    if _NLI_MODEL is None:

        _NLI_MODEL = load_nli_model(
            DEFAULT_NLI_MODEL
        )

    return _NLI_MODEL


# ==================================================
# ENSURE NLI MODEL
# ==================================================

def ensure_nli_model(model=None):
    """
    Ensure that the supplied model is actually an
    NLI model capable of .predict().

    If the RAG embedding model is accidentally passed,
    the correct NLI CrossEncoder is loaded instead.
    """

    if model is None:

        return get_nli_model()

    # CrossEncoder exposes predict()
    if hasattr(
        model,
        "predict"
    ):

        return model

    # SentenceTransformer or incompatible model
    return get_nli_model()


# ==================================================
# NLI LABEL ORDER
# ==================================================

def get_nli_labels(model):
    """
    Determine the label ordering used by the NLI model.

    For cross-encoder/nli-deberta-v3-base this is normally:

        contradiction
        entailment
        neutral
    """

    # ------------------------------------------------
    # Try model.model.config.id2label
    # ------------------------------------------------

    try:

        id2label = (
            model.model.config.id2label
        )

        if id2label:

            labels = []

            for index in range(
                len(id2label)
            ):

                labels.append(
                    str(
                        id2label[index]
                    ).lower()
                )

            return labels

    except Exception:

        pass


    # ------------------------------------------------
    # Try model.config.id2label
    # ------------------------------------------------

    try:

        id2label = (
            model.config.id2label
        )

        if id2label:

            labels = []

            for index in range(
                len(id2label)
            ):

                labels.append(
                    str(
                        id2label[index]
                    ).lower()
                )

            return labels

    except Exception:

        pass


    # ------------------------------------------------
    # Known ordering for this model
    # ------------------------------------------------

    return [
        "contradiction",
        "entailment",
        "neutral"
    ]


# ==================================================
# NLI PREDICTION
# ==================================================

def predict_nli(
    premise,
    hypothesis,
    nli_model=None
):
    """
    Perform NLI classification.

    Premise:
        Retrieved document evidence.

    Hypothesis:
        Generated claim.

    Returns:

        {
            "label": ...,
            "confidence": ...,
            "scores": {...}
        }

    Confidence is the probability of the
    predicted NLI label.
    """

    # ------------------------------------------------
    # Ensure correct model
    # ------------------------------------------------

    nli_model = ensure_nli_model(
        nli_model
    )


    # ------------------------------------------------
    # Clean inputs
    # ------------------------------------------------

    premise = (
        ""
        if premise is None
        else str(premise)
    )

    hypothesis = (
        ""
        if hypothesis is None
        else str(hypothesis)
    )


    # ------------------------------------------------
    # Empty input
    # ------------------------------------------------

    if not premise.strip():

        return {
            "label": "neutral",
            "confidence": 0.0,
            "scores": {
                "contradiction": 0.0,
                "entailment": 0.0,
                "neutral": 0.0
            }
        }


    if not hypothesis.strip():

        return {
            "label": "neutral",
            "confidence": 0.0,
            "scores": {
                "contradiction": 0.0,
                "entailment": 0.0,
                "neutral": 0.0
            }
        }


    # ------------------------------------------------
    # Run NLI
    # ------------------------------------------------

    prediction = nli_model.predict(
        [
            (
                premise,
                hypothesis
            )
        ],
        apply_softmax=True
    )


    # ------------------------------------------------
    # Convert prediction to NumPy
    # ------------------------------------------------

    scores = np.asarray(
        prediction[0],
        dtype=float
    )


    # ------------------------------------------------
    # Get labels
    # ------------------------------------------------

    labels = get_nli_labels(
        nli_model
    )


    # ------------------------------------------------
    # Safety fallback
    # ------------------------------------------------

    if len(labels) != len(scores):

        labels = [
            "contradiction",
            "entailment",
            "neutral"
        ]


    # ------------------------------------------------
    # Predicted class
    # ------------------------------------------------

    predicted_index = int(
        np.argmax(scores)
    )

    predicted_label = (
        labels[predicted_index]
    )

    predicted_confidence = float(
        scores[predicted_index]
    )


    # ------------------------------------------------
    # Build score dictionary
    # ------------------------------------------------

    score_dict = {}

    for index, label in enumerate(
        labels
    ):

        score_dict[
            label
        ] = float(
            scores[index]
        )


    # ------------------------------------------------
    # Ensure standard NLI keys
    # ------------------------------------------------

    for label in [
        "contradiction",
        "entailment",
        "neutral"
    ]:

        if label not in score_dict:

            score_dict[label] = 0.0


    return {
        "label":
            predicted_label,

        "confidence":
            predicted_confidence,

        "scores":
            score_dict
    }


# ==================================================
# FIND BEST EVIDENCE
# ==================================================

def find_best_evidence(
    claim,
    retrieved_chunks,
    nli_model=None
):
    """
    Compare a claim against every retrieved chunk.

    Evidence selection priority:

        1. Entailment
        2. Highest entailment confidence

    If no evidence entails the claim,
    the strongest non-entailment result is returned.
    """

    if not retrieved_chunks:

        return None


    nli_model = ensure_nli_model(
        nli_model
    )


    best_result = None


    for chunk in retrieved_chunks:

        # --------------------------------------------
        # Extract evidence text
        # --------------------------------------------

        evidence_text = chunk.get(
            "text",
            ""
        )


        if not evidence_text:

            continue


        # --------------------------------------------
        # NLI prediction
        # --------------------------------------------

        nli_result = predict_nli(
            evidence_text,
            claim,
            nli_model
        )


        # --------------------------------------------
        # Evidence metadata
        # --------------------------------------------

        evidence = {

            "document":
                chunk.get(
                    "document",
                    "Unknown"
                ),

            "page":
                chunk.get(
                    "page",
                    "Unknown"
                ),

            "chunk_id":
                chunk.get(
                    "chunk_id",
                    "Unknown"
                ),

            "distance":
                chunk.get(
                    "distance",
                    None
                ),

            "text":
                evidence_text
        }


        # --------------------------------------------
        # Result
        # --------------------------------------------

        result = {

            "grounded":
                nli_result[
                    "label"
                ] == "entailment",

            "nli_label":
                nli_result[
                    "label"
                ],

            "nli_confidence":
                nli_result[
                    "confidence"
                ],

            "nli_scores":
                nli_result[
                    "scores"
                ],

            "evidence":
                evidence
        }


        # --------------------------------------------
        # First result
        # --------------------------------------------

        if best_result is None:

            best_result = result

            continue


        current_entails = (
            result[
                "nli_label"
            ] == "entailment"
        )

        best_entails = (
            best_result[
                "nli_label"
            ] == "entailment"
        )


        # --------------------------------------------
        # Entailment beats non-entailment
        # --------------------------------------------

        if (
            current_entails
            and not best_entails
        ):

            best_result = result

            continue


        # --------------------------------------------
        # Both entailment
        # --------------------------------------------

        if (
            current_entails
            and best_entails
        ):

            if (
                result[
                    "nli_confidence"
                ]
                >
                best_result[
                    "nli_confidence"
                ]
            ):

                best_result = result

            continue


        # --------------------------------------------
        # Neither entails
        # --------------------------------------------

        if (
            not current_entails
            and not best_entails
        ):

            if (
                result[
                    "nli_confidence"
                ]
                >
                best_result[
                    "nli_confidence"
                ]
            ):

                best_result = result


    return best_result


# ==================================================
# SINGLE CLAIM EVALUATION
# ==================================================

def evaluate_claim(
    claim,
    retrieved_chunks,
    nli_model=None,
    groundedness_threshold=0.55
):
    """
    Evaluate one generated claim.

    Groundedness rule:

        entailment     -> GROUNDED
        contradiction  -> UNGROUNDED
        neutral        -> UNGROUNDED

    groundedness_threshold is retained for compatibility
    with older scripts. NLI entailment is the actual
    classification criterion.
    """

    nli_model = ensure_nli_model(
        nli_model
    )


    # ------------------------------------------------
    # Find evidence
    # ------------------------------------------------

    best_result = find_best_evidence(
        claim,
        retrieved_chunks,
        nli_model
    )


    # ------------------------------------------------
    # No evidence
    # ------------------------------------------------

    if best_result is None:

        return {

            # Current terminology
            "claim":
                claim,

            # Backward compatibility
            "sentence":
                claim,

            "grounded":
                False,

            "nli_label":
                "no_evidence",

            "nli_confidence":
                0.0,

            "nli_scores":
                {},

            "evidence":
                None
        }


    # ------------------------------------------------
    # Return evaluated claim
    # ------------------------------------------------

    return {

        # Current terminology
        "claim":
            claim,

        # Backward compatibility for
        # test_groundedness.py
        "sentence":
            claim,

        "grounded":
            best_result[
                "grounded"
            ],

        "nli_label":
            best_result[
                "nli_label"
            ],

        "nli_confidence":
            best_result[
                "nli_confidence"
            ],

        "nli_scores":
            best_result[
                "nli_scores"
            ],

        "evidence":
            best_result[
                "evidence"
            ]
    }


# ==================================================
# FULL GROUNDEDNESS EVALUATION
# ==================================================

def evaluate_groundedness(
    answer,
    retrieved_chunks,
    nli_model=None,
    groundedness_threshold=0.55
):
    """
    Evaluate every claim in a generated answer.

    The function is intentionally compatible with both
    the newer claim-based evaluation code and the older
    sentence-based test scripts.
    """

    # ------------------------------------------------
    # Ensure correct NLI model
    # ------------------------------------------------

    nli_model = ensure_nli_model(
        nli_model
    )


    # ------------------------------------------------
    # Split answer into claims
    # ------------------------------------------------

    claims = split_into_claims(
        answer
    )


    # ------------------------------------------------
    # Evaluate claims
    # ------------------------------------------------

    results = []


    for claim in claims:

        result = evaluate_claim(
            claim,
            retrieved_chunks,
            nli_model=nli_model,
            groundedness_threshold=
                groundedness_threshold
        )

        results.append(
            result
        )


    # ------------------------------------------------
    # Counts
    # ------------------------------------------------

    total_claims = len(
        results
    )


    grounded_claims = sum(
        1
        for result in results
        if result[
            "grounded"
        ]
    )


    ungrounded_claims = (
        total_claims
        - grounded_claims
    )


    # ------------------------------------------------
    # Groundedness score
    # ------------------------------------------------

    if total_claims > 0:

        groundedness_score = (
            grounded_claims
            /
            total_claims
        )

    else:

        groundedness_score = 0.0


    # ------------------------------------------------
    # Return
    #
    # Both claim-based and sentence-based keys are
    # deliberately provided.
    # ------------------------------------------------

    return {

        # --------------------------------------------
        # Current claim terminology
        # --------------------------------------------

        "claims":
            claims,

        "claim_results":
            results,

        "total_claims":
            total_claims,

        "grounded_claims":
            grounded_claims,

        "ungrounded_claims":
            ungrounded_claims,


        # --------------------------------------------
        # Backward-compatible sentence terminology
        # --------------------------------------------

        "sentence_results":
            results,

        "total_sentences":
            total_claims,

        "grounded_sentences":
            grounded_claims,

        "ungrounded_sentences":
            ungrounded_claims,


        # --------------------------------------------
        # Overall score
        # --------------------------------------------

        "groundedness_score":
            groundedness_score
    }


# ==================================================
# FORMAT GROUNDEDNESS RESULT
# ==================================================

def format_groundedness_result(
    result
):
    """
    Format groundedness results for terminal output.
    """

    lines = []


    # ------------------------------------------------
    # Summary
    # ------------------------------------------------

    lines.append(
        "Groundedness:"
    )

    lines.append(
        f"Total Claims        : "
        f"{result['total_claims']}"
    )

    lines.append(
        f"Grounded Claims     : "
        f"{result['grounded_claims']}"
    )

    lines.append(
        f"Ungrounded Claims   : "
        f"{result['ungrounded_claims']}"
    )

    lines.append(
        f"Groundedness Score  : "
        f"{result['groundedness_score'] * 100:.2f}%"
    )


    lines.append("")

    lines.append(
        "Claim Analysis:"
    )


    # ------------------------------------------------
    # Individual claims
    # ------------------------------------------------

    for item in result[
        "sentence_results"
    ]:

        status = (
            "GROUNDED"
            if item[
                "grounded"
            ]
            else
            "UNGROUNDED"
        )


        lines.append("")

        lines.append(
            f"[{status}] "
            f"NLI Label: "
            f"{item['nli_label']}"
        )

        lines.append(
            f"Confidence: "
            f"{item['nli_confidence']:.4f}"
        )

        lines.append(
            "Claim:"
        )

        lines.append(
            f"  {item['claim']}"
        )


        # --------------------------------------------
        # Evidence
        # --------------------------------------------

        evidence = item.get(
            "evidence"
        )


        if evidence:

            lines.append("")

            lines.append(
                "Evidence:"
            )

            lines.append(
                f"  Document : "
                f"{evidence['document']}"
            )

            lines.append(
                f"  Page     : "
                f"{evidence['page']}"
            )

            lines.append(
                f"  Chunk    : "
                f"{evidence['chunk_id']}"
            )


    return "\n".join(
        lines
    )