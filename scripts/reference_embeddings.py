# /// script
# dependencies = [
#   "sentence-transformers==5.1.2",
# ]
# ///

import argparse
import json
import sys

from sentence_transformers import SentenceTransformer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--kind",
        choices=("document", "query"),
        default="document",
    )
    parser.add_argument("--prompt", default=None)
    args = parser.parse_args()

    texts = json.load(sys.stdin)
    model = SentenceTransformer(args.model)

    if args.prompt is None:
        embeddings = model.encode(texts, normalize_embeddings=True)
    elif args.kind == "query" and hasattr(model, "encode_query"):
        embeddings = model.encode_query(
            texts,
            prompt=args.prompt,
            normalize_embeddings=True,
        )
    elif args.kind == "document" and hasattr(model, "encode_document"):
        embeddings = model.encode_document(
            texts,
            prompt=args.prompt,
            normalize_embeddings=True,
        )
    else:
        embeddings = model.encode(
            texts,
            prompt=args.prompt,
            normalize_embeddings=True,
        )

    json.dump(embeddings.tolist(), sys.stdout)


if __name__ == "__main__":
    main()
