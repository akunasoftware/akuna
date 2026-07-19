from pathlib import Path

from huggingface_hub import hf_hub_download


HF_REPO_TEST_CORPUS = "akunasoftware/test-corpus"
HF_REPO_TEST_CORPUS_REVISION = "628188f924994038784e21e828518441634948ba"


def fixture_path(name: str) -> Path:
    """Return a fixture from the pinned test corpus revision."""
    return Path(
        hf_hub_download(
            HF_REPO_TEST_CORPUS,
            name,
            repo_type="dataset",
            revision=HF_REPO_TEST_CORPUS_REVISION,
        )
    )
