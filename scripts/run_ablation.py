import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ablation.variants import ALL_POSITIONS, prepare_position_variant
from src.official_stacks.meta_secalign.config import TARGET_MODEL
from src.official_stacks.meta_secalign.paths import DATA_DIR

OPENAI_CONFIG_PATH = DATA_DIR / "openai_configs.yaml"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "docs" / "raporlar" / "ablation"


def write_position_report(position: str, metrics: dict, output_dir: Path) -> Path:
    report_path = output_dir / f"{position}.json"
    payload = {
        "model": TARGET_MODEL,
        "position": position,
        "n_tokens": 5,
        "metrics": {
            "win_rate": metrics["win_rate"],
            "asr": metrics["asr"],
        },
        "artifacts": metrics["artifacts"],
    }
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    return report_path


def write_summary_csv(results: list[dict], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8", newline="\n") as fh:
        writer = csv.DictWriter(fh, fieldnames=["position", "win_rate", "asr"])
        writer.writeheader()
        for row in results:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Token pozisyon ablation: n=5 sabit, 4 pozisyon karşılaştırması")
    parser.add_argument(
        "--positions",
        nargs="+",
        choices=ALL_POSITIONS,
        default=ALL_POSITIONS,
        metavar="POS",
        help=f"Değerlendirilecek pozisyonlar (varsayılan: hepsi). Seçenekler: {ALL_POSITIONS}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Rapor çıkış dizini",
    )
    parser.add_argument(
        "--model-output-root",
        type=Path,
        default=None,
        help="Model artifact kök dizini (varsayılan: defensivetoken stack root)",
    )
    parser.add_argument(
        "--openai-config",
        type=Path,
        default=OPENAI_CONFIG_PATH,
        help="OpenAI config YAML dosyası",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Model hazırlama ve eval yapmadan konfigürasyonu yazdır",
    )
    args = parser.parse_args()

    if args.dry_run:
        print(json.dumps({
            "model": TARGET_MODEL,
            "positions": args.positions,
            "output_dir": str(args.output_dir),
            "openai_config": str(args.openai_config),
        }, ensure_ascii=False, indent=2))
        return

    if not args.openai_config.exists():
        raise FileNotFoundError(f"OpenAI config bulunamadı: {args.openai_config}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    from src.official_stacks.meta_secalign.qwen_alpaca import run_qwen_alpaca_eval

    results = []
    for position in args.positions:
        print(f"\n{'='*60}")
        print(f"Pozisyon: {position}")
        print(f"{'='*60}")

        model_path = prepare_position_variant(TARGET_MODEL, position, args.model_output_root)
        print(f"Model: {model_path}")

        metrics = run_qwen_alpaca_eval(
            mode="ablation",
            model_name_or_path=str(model_path),
            openai_config_path=str(args.openai_config),
            include_gcg=False,
        )

        report_path = write_position_report(position, metrics, args.output_dir)
        print(f"Rapor: {report_path}")
        print(json.dumps({"win_rate": metrics["win_rate"], "asr": metrics["asr"]}, indent=2))

        results.append({"position": position, "win_rate": metrics["win_rate"], "asr": metrics["asr"]})

    summary_path = args.output_dir / "summary.csv"
    write_summary_csv(results, summary_path)
    print(f"\nÖzet: {summary_path}")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
