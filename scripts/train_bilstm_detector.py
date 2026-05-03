import argparse
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.bilstm.model import InjectionDetector
from src.bilstm.tokenizer import WordTokenizer
from src.official_stacks.meta_secalign.qwen_alpaca import ATTACK_BUILDERS
from src.official_stacks.meta_secalign.utils import jload

DEFAULT_DATA = REPO_ROOT / "src" / "official_stacks" / "meta_secalign" / "data" / "davinci_003_outputs.json"
DEFAULT_OUTPUT = REPO_ROOT / "bilstm_checkpoint.pt"


def generate_training_data(data_path: str) -> tuple[list[str], list[int]]:
    rows = jload(data_path)
    texts: list[str] = []
    labels: list[int] = []

    # Negative: clean input fields
    for row in rows:
        texts.append(str(row.get("input", "")).strip())
        labels.append(0)

    # Positive: attacked input fields (only rows that have non-empty input)
    attack_rows = [row for row in rows if str(row.get("input", "")).strip()]
    for attack_name in ("ignore", "completion", "completion_ignore"):
        for attacked in [ATTACK_BUILDERS[attack_name](row) for row in attack_rows]:
            texts.append(str(attacked.get("input", "")).strip())
            labels.append(1)

    return texts, labels


def load_deepset_train() -> tuple[list[str], list[int]]:
    """Load deepset/prompt-injections train split for training enrichment."""
    from datasets import load_dataset
    ds = load_dataset("deepset/prompt-injections", split="train")
    texts, labels = [], []
    for row in ds:
        t = str(row.get("text", "") or "").strip()
        if t:
            texts.append(t)
            labels.append(int(row.get("label", 0)))
    return texts, labels


def load_rogue_security(local_path: str | None = None, hf_token: str | None = None) -> tuple[list[str], list[int]]:
    """Load rogue-security/prompt-injections-benchmark from local CSV or HF Hub."""
    import csv
    texts, labels = [], []

    if local_path:
        with open(local_path, encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                t = str(row.get("text", "") or "").strip()
                if not t:
                    continue
                raw_label = str(row.get("label", "benign")).strip().lower()
                texts.append(t)
                labels.append(0 if raw_label == "benign" else 1)
        return texts, labels

    from datasets import load_dataset
    ds = load_dataset("rogue-security/prompt-injections-benchmark", token=hf_token, split="train")
    for row in ds:
        t = str(row.get("text", "") or "").strip()
        if not t:
            continue
        raw_label = str(row.get("label", "benign")).strip().lower()
        texts.append(t)
        labels.append(0 if raw_label == "benign" else 1)
    return texts, labels


def load_hackaprompt(local_path: str | None = None, max_samples: int = 10000) -> tuple[list[str], list[int]]:
    """Load HackAPrompt dataset from local parquet or HuggingFace Hub.

    All user_inputs are injection attempts (label=1).
    Successful ones (correct=True) are prioritized; remaining slots filled from failed attempts.
    max_samples caps total to avoid overwhelming class balance.
    """
    seen: set[str] = set()
    successful, failed = [], []

    if local_path:
        import pyarrow.parquet as pq
        table = pq.read_table(local_path, columns=["user_input", "correct"])
        rows_iter = zip(table.to_pydict()["user_input"], table.to_pydict()["correct"])
    else:
        from datasets import load_dataset
        ds = load_dataset("hackaprompt/hackaprompt-dataset", split="train")
        rows_iter = ((row["user_input"], row["correct"]) for row in ds)

    for text, correct in rows_iter:
        t = str(text or "").strip()
        if not t or t in seen:
            continue
        seen.add(t)
        (successful if correct else failed).append(t)

    selected = successful[:max_samples]
    remaining = max_samples - len(selected)
    if remaining > 0:
        selected += failed[:remaining]

    print(f"  HackAPrompt: {len(selected)} benzersiz örnek "
          f"(başarılı={min(len(successful), max_samples)}, başarısız={max(0, len(selected)-min(len(successful), max_samples))})")
    return selected, [1] * len(selected)


class _InjectionDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int], tokenizer: WordTokenizer, max_len: int = 512):
        self.samples = [(tokenizer.encode(t, max_len), l) for t, l in zip(texts, labels)]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self.samples[idx]


def _collate(batch):
    ids_list, labels = zip(*batch)
    lengths = torch.tensor([len(ids) for ids in ids_list], dtype=torch.long)
    max_len = int(lengths.max().item())
    padded = torch.zeros(len(ids_list), max_len, dtype=torch.long)
    for i, ids in enumerate(ids_list):
        padded[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
    return padded, lengths, torch.tensor(labels, dtype=torch.float)


def train(
    data_path: str,
    output_path: str,
    epochs: int = 20,
    batch_size: int = 64,
    lr: float = 1e-3,
    seed: int = 42,
    include_deepset: bool = True,
    hf_token: str | None = None,
    rogue_security_path: str | None = None,
    hackaprompt_path: str | None = None,
    include_hackaprompt: bool = False,
) -> None:
    random.seed(seed)
    torch.manual_seed(seed)

    print("Generating training data...")
    texts, labels = generate_training_data(data_path)

    if include_deepset:
        print("  deepset/prompt-injections train split ekleniyor...")
        try:
            ds_texts, ds_labels = load_deepset_train()
            texts.extend(ds_texts)
            labels.extend(ds_labels)
            print(f"  +{len(ds_texts)} örnek (injection={sum(ds_labels)}, clean={len(ds_labels)-sum(ds_labels)})")
        except Exception as exc:
            print(f"  UYARI: deepset yüklenemedi — {exc}")

    if rogue_security_path or hf_token:
        print("  rogue-security/prompt-injections-benchmark ekleniyor...")
        try:
            rs_texts, rs_labels = load_rogue_security(local_path=rogue_security_path, hf_token=hf_token)
            texts.extend(rs_texts)
            labels.extend(rs_labels)
            print(f"  +{len(rs_texts)} örnek (injection={sum(rs_labels)}, clean={len(rs_labels)-sum(rs_labels)})")
        except Exception as exc:
            print(f"  UYARI: rogue-security yüklenemedi — {exc}")

    if hackaprompt_path or include_hackaprompt:
        print("  HackAPrompt dataset ekleniyor...")
        try:
            hp_texts, hp_labels = load_hackaprompt(local_path=hackaprompt_path)
            texts.extend(hp_texts)
            labels.extend(hp_labels)
            print(f"  +{len(hp_texts)} örnek (hepsi injection=1)")
        except Exception as exc:
            print(f"  UYARI: HackAPrompt yüklenemedi — {exc}")

    combined = list(zip(texts, labels))
    random.shuffle(combined)
    split = int(len(combined) * 0.8)
    train_pairs, val_pairs = combined[:split], combined[split:]

    n_pos = sum(l for _, l in combined)
    print(f"  Total: {len(combined)}  Positive(attack): {n_pos}  Negative(clean): {len(combined) - n_pos}")
    print(f"  Train: {len(train_pairs)}  Val: {len(val_pairs)}")

    train_texts, train_labels = zip(*train_pairs)
    val_texts, val_labels = zip(*val_pairs)

    tokenizer = WordTokenizer.build_vocab(list(train_texts))
    print(f"  Vocab size: {len(tokenizer)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    train_loader = DataLoader(
        _InjectionDataset(list(train_texts), list(train_labels), tokenizer),
        batch_size=batch_size, shuffle=True, collate_fn=_collate,
    )
    val_loader = DataLoader(
        _InjectionDataset(list(val_texts), list(val_labels), tokenizer),
        batch_size=batch_size, shuffle=False, collate_fn=_collate,
    )

    model_kwargs = {"vocab_size": len(tokenizer), "embed_dim": 128, "hidden_dim": 256, "num_layers": 2}
    model = InjectionDetector(**model_kwargs).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")
    best_checkpoint: dict = {}
    patience_counter = 0
    patience = 5

    print("\nTraining...")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for ids, lengths, lbls in train_loader:
            ids, lengths, lbls = ids.to(device), lengths.to(device), lbls.to(device)
            optimizer.zero_grad()
            loss = criterion(model(ids, lengths), lbls)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        correct = total = 0
        with torch.no_grad():
            for ids, lengths, lbls in val_loader:
                ids, lengths, lbls = ids.to(device), lengths.to(device), lbls.to(device)
                logits = model(ids, lengths)
                val_loss += criterion(logits, lbls).item()
                preds = (torch.sigmoid(logits) > 0.5).float()
                correct += (preds == lbls).sum().item()
                total += len(lbls)

        val_acc = correct / total
        tl = train_loss / len(train_loader)
        vl = val_loss / len(val_loader)
        marker = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_checkpoint = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "vocab": tokenizer.vocab,
                "model_kwargs": model_kwargs,
                "val_acc": val_acc,
            }
            torch.save(best_checkpoint, output_path)
            marker = " ← saved"
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Epoch {epoch:3d} | train={tl:.4f} | val={vl:.4f} | acc={val_acc:.4f}{marker}")
                print(f"Early stopping (patience={patience})")
                break

        print(f"  Epoch {epoch:3d} | train={tl:.4f} | val={vl:.4f} | acc={val_acc:.4f}{marker}")

    print(f"\nDone. Best val_acc={best_checkpoint.get('val_acc', 0):.4f}")
    print(f"Checkpoint: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Bi-LSTM injection detector")
    parser.add_argument("--data", default=str(DEFAULT_DATA), help="Path to davinci_003_outputs.json")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output checkpoint path (.pt)")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-deepset", action="store_true", help="deepset train split'ini ekleme")
    parser.add_argument("--hf-token", default=None, help="HuggingFace token (rogue-security HF Hub için)")
    parser.add_argument("--rogue-security", default=None, help="rogue-security test.csv local yolu")
    parser.add_argument("--hackaprompt", default=None, help="hackaprompt.parquet local yolu")
    parser.add_argument("--hackaprompt-hf", action="store_true", help="HackAPrompt'u HuggingFace'den indir")
    args = parser.parse_args()
    train(args.data, args.output, args.epochs, args.batch_size, args.lr, args.seed,
          include_deepset=not args.no_deepset,
          hf_token=args.hf_token,
          rogue_security_path=args.rogue_security,
          hackaprompt_path=args.hackaprompt,
          include_hackaprompt=args.hackaprompt_hf)
