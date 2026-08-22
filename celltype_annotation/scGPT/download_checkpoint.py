"""Download the official scGPT whole-human checkpoint from Google Drive."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

OFFICIAL_FOLDER_URL = (
    "https://drive.google.com/drive/folders/"
    "1oWh_-ZRdhtoGQ2Fw24HP41FgLoomVo-y?usp=sharing"
)
REQUIRED_FILES = ("best_model.pt", "vocab.json", "args.json")


def checkpoint_complete(output_dir):
    output_dir = Path(output_dir)
    return all((output_dir / filename).is_file() for filename in REQUIRED_FILES)


def download_checkpoint(output_dir):
    """Download and verify the official whole-human checkpoint directory."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if checkpoint_complete(output_dir):
        print(f"scGPT checkpoint already exists: {output_dir}")
        return output_dir

    try:
        import gdown
    except ImportError as exc:
        raise RuntimeError(
            "gdown is required. Activate the scGPT environment and run: "
            "pip install gdown"
        ) from exc

    print("Downloading the official scGPT whole-human checkpoint...")
    try:
        gdown.download_folder(
            OFFICIAL_FOLDER_URL,
            output=str(output_dir),
            quiet=False,
            use_cookies=False,
            remaining_ok=True,
        )
    except TypeError:
        # Compatibility with older gdown releases without remaining_ok.
        gdown.download_folder(
            OFFICIAL_FOLDER_URL,
            output=str(output_dir),
            quiet=False,
            use_cookies=False,
        )

    # Some gdown versions reproduce the Drive folder as one nested directory.
    for filename in REQUIRED_FILES:
        destination = output_dir / filename
        if destination.exists():
            continue
        matches = list(output_dir.rglob(filename))
        if len(matches) == 1:
            shutil.copy2(matches[0], destination)

    if not checkpoint_complete(output_dir):
        missing = [
            filename for filename in REQUIRED_FILES
            if not (output_dir / filename).is_file()
        ]
        raise RuntimeError(
            "Checkpoint download is incomplete; missing: " + ", ".join(missing)
        )

    print(f"scGPT checkpoint ready: {output_dir}")
    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path(__file__).parent / "models" / "scGPT_human",
    )
    arguments = parser.parse_args()
    download_checkpoint(arguments.output_dir)
