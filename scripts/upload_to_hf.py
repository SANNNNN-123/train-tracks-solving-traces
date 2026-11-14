"""Publish the training JSONL and dataset card, without other local files.

Run from the repository root: python scripts/upload_to_hf.py
"""

import os
from pathlib import Path
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parent.parent
FILES = {
    "README.md": "data/README.md",
    "data/train.jsonl": "data/train.jsonl",
}


def config():
    values = dict(os.environ)
    env_file = ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            values.setdefault(key.strip(), value.strip().strip('"\''))
    return values


def main():
    from huggingface_hub import CommitOperationAdd, HfApi

    values = config()
    destination = values.get("HF_DATASET_PATH", "").strip()
    if destination.startswith("https://"):
        parsed = urlparse(destination)
        if parsed.netloc != "huggingface.co" or not parsed.path.startswith("/datasets/"):
            raise ValueError("HF_DATASET_PATH must be a Hugging Face dataset URL or owner/name")
        repo_id = parsed.path.removeprefix("/datasets/").strip("/")
    else:
        repo_id = destination
    token = values.get("HF_TOKEN")
    if len(repo_id.split("/")) != 2 or not token:
        raise ValueError("Set HF_DATASET_PATH and HF_TOKEN in .env")

    missing = [source for source in FILES.values() if not (ROOT / source).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing upload files: {missing}")
    operations = [CommitOperationAdd(path_in_repo=target, path_or_fileobj=ROOT / source)
                  for target, source in FILES.items()]
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
    result = api.create_commit(repo_id=repo_id, repo_type="dataset",
                               operations=operations,
                               commit_message="Publish Train Tracks solving traces")
    print(f"Uploaded {', '.join(FILES)}: {result.commit_url}")


if __name__ == "__main__":
    main()
