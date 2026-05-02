"""
OOD evaluation of the Bi-LSTM injection detector.

Uses the deepset/prompt-injections TEST split as the held-out evaluation set.
The train split of this dataset is used during model training, so only the
test split is genuinely out-of-distribution for the final model.

Usage:
    python scripts/eval_bilstm_ood.py \
        --checkpoint bilstm_checkpoint.pt \
        --output docs/raporlar/bilstm_ood.json \
        [--threshold 0.5]
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def load_ood_data() -> tuple[list[str], list[int], dict]:
    """Load deepset/prompt-injections TEST split (never seen during training)."""
    from datasets import load_dataset

    print("  deepset/prompt-injections test split yükleniyor...")
    ds = load_dataset("deepset/prompt-injections", split="test")
    texts, labels = [], []
    for row in ds:
        t = str(row.get("text", "") or "").strip()
        if t:
            texts.append(t)
            labels.append(int(row.get("label", 0)))
    print(f"    {len(texts)} örnek  (injection={sum(labels)}, clean={len(labels)-sum(labels)})")
    return texts, labels, {"deepset/prompt-injections (test)": len(texts)}


def evaluate(
    checkpoint_path: str,
    texts: list[str],
    labels: list[int],
    threshold: float = 0.5,
) -> dict:
    import torch
    from src.bilstm.model import InjectionDetector
    from src.bilstm.tokenizer import WordTokenizer

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    tokenizer = WordTokenizer(ckpt["vocab"])
    detector = InjectionDetector(**ckpt["model_kwargs"])
    detector.load_state_dict(ckpt["model_state"])
    detector.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    detector = detector.to(device)

    print(f"\nModel yüklendi — device: {device}  threshold: {threshold}")
    print(f"Toplam: {len(texts)}  injection={sum(labels)}  clean={len(labels)-sum(labels)}")

    preds: list[int] = []
    scores: list[float] = []
    batch_size = 128

    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        encoded = [tokenizer.encode(t) for t in batch_texts]
        lengths = torch.tensor([len(ids) for ids in encoded], dtype=torch.long)
        max_len = int(lengths.max().item())
        padded = torch.zeros(len(encoded), max_len, dtype=torch.long)
        for i, ids in enumerate(encoded):
            padded[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        padded = padded.to(device)
        lengths = lengths.to(device)
        with torch.no_grad():
            logits = detector(padded, lengths)
            probs = torch.sigmoid(logits).cpu().tolist()
        scores.extend(probs)
        preds.extend([1 if p > threshold else 0 for p in probs])

    tp = sum(p == 1 and l == 1 for p, l in zip(preds, labels))
    fp = sum(p == 1 and l == 0 for p, l in zip(preds, labels))
    tn = sum(p == 0 and l == 0 for p, l in zip(preds, labels))
    fn = sum(p == 0 and l == 1 for p, l in zip(preds, labels))

    accuracy  = (tp + tn) / len(labels) if labels else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    fpr       = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return {
        "checkpoint": checkpoint_path,
        "threshold": threshold,
        "n_total": len(labels),
        "n_injection": sum(labels),
        "n_clean": len(labels) - sum(labels),
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "metrics": {
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "false_positive_rate": round(fpr, 4),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="OOD eval of Bi-LSTM detector")
    parser.add_argument("--checkpoint", default=str(REPO_ROOT / "bilstm_checkpoint.pt"))
    parser.add_argument("--output", default=str(REPO_ROOT / "docs" / "raporlar" / "bilstm_ood.json"))
    parser.add_argument("--threshold", type=float, default=0.3)
    args = parser.parse_args()

    print("OOD dataset yükleniyor...")
    texts, labels, sources = load_ood_data()

    results = evaluate(args.checkpoint, texts, labels, threshold=args.threshold)
    results["datasets"] = sources

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("\n=== OOD Evaluation Results ===")
    m = results["metrics"]
    print(f"  Accuracy           : {m['accuracy']:.4f}")
    print(f"  Precision          : {m['precision']:.4f}")
    print(f"  Recall             : {m['recall']:.4f}")
    print(f"  F1                 : {m['f1']:.4f}")
    print(f"  False Positive Rate: {m['false_positive_rate']:.4f}  (utility cost)")
    print(f"\nRapor: {out_path}")


if __name__ == "__main__":
    main()
