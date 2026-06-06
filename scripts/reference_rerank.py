# /// script
# dependencies = [
#   "einops==0.8.0",
#   "torch==2.9.1",
#   "transformers==4.57.3",
# ]
# ///

import argparse
import json
import sys

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    args = parser.parse_args()

    pairs = json.load(sys.stdin)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=torch.float32,
    )
    model.eval()

    with torch.no_grad():
        inputs = tokenizer(
            pairs,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=model.config.max_position_embeddings - 2,
        )
        scores = model(**inputs).logits.squeeze(-1).tolist()

    if isinstance(scores, float):
        scores = [scores]

    json.dump(scores, sys.stdout)


if __name__ == "__main__":
    main()
