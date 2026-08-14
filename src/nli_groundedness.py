import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)


# ==================================================
# CONFIGURATION
# ==================================================

MODEL_NAME = (
    "cross-encoder/nli-deberta-v3-base"
)


# ==================================================
# NLI GROUNDEDNESS MODEL
# ==================================================

class NLIGroundednessEvaluator:

    def __init__(
        self,
        model_name=MODEL_NAME
    ):

        print(
            f"Loading NLI model: "
            f"{model_name}"
        )

        self.tokenizer = (
            AutoTokenizer.from_pretrained(
                model_name
            )
        )

        self.model = (
            AutoModelForSequenceClassification
            .from_pretrained(
                model_name
            )
        )

        self.model.eval()


    # ==================================================
    # PREDICT
    # ==================================================

    def predict(
        self,
        premise,
        hypothesis
    ):
        """
        Determine whether the premise supports
        the hypothesis.

        Premise:
            Retrieved evidence.

        Hypothesis:
            Generated claim.
        """

        inputs = self.tokenizer(
            premise,
            hypothesis,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )


        with torch.no_grad():

            outputs = self.model(
                **inputs
            )


        probabilities = torch.softmax(
            outputs.logits,
            dim=-1
        )[0]


        predicted_id = int(
            torch.argmax(
                probabilities
            )
        )


        label = (
            self.model.config.id2label[
                predicted_id
            ]
        )


        scores = {}

        for index, probability in enumerate(
            probabilities
        ):

            model_label = (
                self.model.config.id2label[
                    index
                ]
            )

            scores[
                model_label
            ] = float(
                probability
            )


        return {
            "label": label,
            "scores": scores
        }


    # ==================================================
    # GROUNDEDNESS
    # ==================================================

    def evaluate_claim(
        self,
        claim,
        contexts
    ):
        """
        Evaluate a claim against multiple
        retrieved context chunks.

        A claim is considered grounded if
        at least one context chunk entails it.
        """

        if not contexts:

            return {
                "grounded": False,
                "label": "NO_CONTEXT",
                "confidence": 0.0,
                "evidence": None
            }


        best_result = None


        for context in contexts:

            result = self.predict(
                context,
                claim
            )


            # ------------------------------------------
            # Normalize label
            # ------------------------------------------

            label = result[
                "label"
            ].lower()


            # ------------------------------------------
            # Extract entailment confidence
            # ------------------------------------------

            entailment_score = 0.0


            for key, value in result[
                "scores"
            ].items():

                if "entail" in key.lower():

                    entailment_score = value

                    break


            candidate = {
                "grounded":
                    "entail" in label,

                "label":
                    result["label"],

                "confidence":
                    entailment_score,

                "evidence":
                    context,

                "scores":
                    result["scores"]
            }


            # ------------------------------------------
            # Keep strongest entailment
            # ------------------------------------------

            if (
                best_result is None
                or candidate["confidence"]
                >
                best_result["confidence"]
            ):

                best_result = candidate


        return best_result