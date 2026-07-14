#!/usr/bin/env python3
"""Convert local ScienceQA and ChartQA folders to a shared JSONL format.

The output records do not copy images.  Each record stores an absolute image path,
which keeps the manifest small and makes it convenient to run on an offline server.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import yaml
from tqdm import tqdm


LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def write_jsonl(records: Iterable[dict[str, Any]], output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def scienceqa_split_file(root: Path) -> Path:
    for name in ("pid_splits.json", "pid_split.json"):
        candidate = root / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Cannot find pid_splits.json or pid_split.json in ScienceQA.")


def build_scienceqa_records(root: Path, split: str) -> list[dict[str, Any]]:
    with (root / "problems.json").open("r", encoding="utf-8") as handle:
        problems = json.load(handle)
    with scienceqa_split_file(root).open("r", encoding="utf-8") as handle:
        split_ids = json.load(handle)[split]

    records: list[dict[str, Any]] = []
    missing_images = 0
    for pid in tqdm(split_ids, desc=f"ScienceQA {split}"):
        item = problems[str(pid)]
        # ScienceQA's image field can be null.  Only image-conditioned examples
        # belong in this visual-token project.
        if not item.get("image"):
            continue
        image_path = root / split / str(pid) / "image.png"
        if not image_path.exists():
            missing_images += 1
            continue
        choices = item.get("choices", [])
        formatted_choices = "\n".join(
            f"({LETTERS[index]}) {choice}" for index, choice in enumerate(choices)
        )
        prompt = (
            f"Question: {item['question']}\n"
            f"Choices:\n{formatted_choices}\n"
            "Answer with the option letter only."
        )
        answer_index = item.get("answer")
        answer = LETTERS[answer_index] if isinstance(answer_index, int) else str(answer_index)
        records.append(
            {
                "id": f"scienceqa_{pid}",
                "dataset": "scienceqa",
                "split": split,
                "image": str(image_path.resolve()),
                "question": item["question"],
                "prompt": prompt,
                "choices": choices,
                "answer": answer,
                "metadata": {
                    "pid": str(pid),
                    "subject": item.get("subject"),
                    "topic": item.get("topic"),
                    "category": item.get("category"),
                },
            }
        )
    print(f"ScienceQA {split}: {len(records)} image examples; {missing_images} missing images.")
    return records


def first_present(item: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in item and item[key] is not None:
            return item[key]
    return None


def build_chartqa_records(root: Path, split: str, source: str) -> list[dict[str, Any]]:
    annotation_path = root / split / f"{split}_{source}.json"
    if not annotation_path.exists():
        raise FileNotFoundError(f"Cannot find {annotation_path}")
    with annotation_path.open("r", encoding="utf-8") as handle:
        annotations = json.load(handle)

    records: list[dict[str, Any]] = []
    missing_images = 0
    for index, item in enumerate(tqdm(annotations, desc=f"ChartQA {split}-{source}")):
        image_name = first_present(item, ("imgname", "image", "image_name"))
        question = first_present(item, ("query", "question"))
        answer = first_present(item, ("label", "answer", "answers"))
        if image_name is None or question is None or answer is None:
            raise ValueError(f"Unexpected ChartQA record at {annotation_path}:{index}: {item}")
        image_path = root / split / "png" / str(image_name)
        if not image_path.exists():
            missing_images += 1
            continue
        if isinstance(answer, list):
            answer = answer[0]
        records.append(
            {
                "id": f"chartqa_{split}_{source}_{index}",
                "dataset": "chartqa",
                "split": split,
                "source": source,
                "image": str(image_path.resolve()),
                "question": str(question),
                "prompt": f"Question: {question}\nAnswer concisely.",
                "answer": str(answer),
                "metadata": {"image_name": str(image_name), "annotation_index": index},
            }
        )
    print(f"ChartQA {split}-{source}: {len(records)} examples; {missing_images} missing images.")
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-per-split", type=int, default=0,
                        help="For smoke tests only. 0 keeps every example.")
    args = parser.parse_args()

    cfg = read_yaml(args.config)
    output_dir = args.output_dir or Path(cfg["output_root"]) / "manifests"
    science_root = Path(cfg["scienceqa_root"])
    chart_root = Path(cfg["chartqa_root"])

    for split in ("train", "val", "test"):
        records = build_scienceqa_records(science_root, split)
        if args.max_per_split:
            records = records[: args.max_per_split]
        count = write_jsonl(records, output_dir / f"scienceqa_{split}.jsonl")
        print(f"Wrote {count} records to {output_dir / f'scienceqa_{split}.jsonl'}")

    for split in ("train", "val", "test"):
        for source in ("human", "augmented"):
            annotation = chart_root / split / f"{split}_{source}.json"
            if not annotation.exists():
                print(f"Skip missing {annotation}")
                continue
            records = build_chartqa_records(chart_root, split, source)
            if args.max_per_split:
                records = records[: args.max_per_split]
            output = output_dir / f"chartqa_{split}_{source}.jsonl"
            count = write_jsonl(records, output)
            print(f"Wrote {count} records to {output}")


if __name__ == "__main__":
    main()
