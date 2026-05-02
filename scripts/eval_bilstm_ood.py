"""
OOD evaluation of the Bi-LSTM injection detector.

Uses two publicly available datasets (no auth required):
  1. deepset/prompt-injections  — labeled injection vs. clean prompts
  2. JasperLS/prompt-injection  — second source for diversity

Both are out-of-distribution: the model was trained only on Alpaca-derived
synthetic attacks and has never seen these examples.

Usage:
    python scripts/eval_bilstm_ood.py \
        --checkpoint bilstm_checkpoint.pt \
        --output docs/raporlar/bilstm_ood.json \
        [--threshold 0.5] \
        [--max-samples 1000]
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_deepset() -> tuple[list[str], list[int]]:
    """deepset/prompt-injections: 'text' + 'label' (0=clean, 1=injection)."""
    from datasets import load_dataset
    ds = load_dataset("deepset/prompt-injections", split="train")
    texts, labels = [], []
    for row in ds:
        t = str(row.get("text", "") or "").strip()
        if t:
            texts.append(t)
            labels.append(int(row.get("label", 0)))
    return texts, labels


def _load_jasper() -> tuple[list[str], list[int]]:
    """JasperLS/prompt-injection: 'text' + 'label' (0=clean, 1=injection)."""
    from datasets import load_dataset
    ds = load_dataset("JasperLS/prompt-injection", split="train")
    texts, labels = [], []
    for row in ds:
        t = str(row.get("text", "") or "").strip()
        if t:
            texts.append(t)
            labels.append(int(row.get("label", 0)))
    return texts, labels


def load_ood_data(max_samples: int | None = None) -> tuple[list[str], list[int], dict]:
    """Load and merge OOD datasets. Returns (texts, labels, source_info)."""
    sources: dict[str, int] = {}
    all_texts: list[str] = []
    all_labels: list[int] = []

    for name, loader in [("deepset/prompt-injections", _load_deepset),
                          ("JasperLS/prompt-injection", _load_jasper)]:
        try:
            print(f"  {name} yükleniyor...")
            t, l = loader()
            sources[name] = len(t)
            all_texts.extend(t)
            all_labels.extend(l)
            print(f"    {len(t)} örnek  (injection={sum(l)}, clean={len(l)-sum(l)})")
        except Exception as exc:
            print(f"    UYARI: {name} yüklenemedi — {exc}")

    if not all_texts:
        raise RuntimeError("Hiçbir OOD dataset yüklenemedi.")

    if max_samples and len(all_texts) > max_samples:
        all_texts = all_texts[:max_samples]
        all_labels = all_labels[:max_samples]

    return all_texts, all_labels, sources


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
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    print("OOD dataset'ler yükleniyor...")
    texts, labels, sources = load_ood_data(max_samples=args.max_samples)

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
