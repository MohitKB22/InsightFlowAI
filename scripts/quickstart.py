#!/usr/bin/env python3
"""
Quick-start script — trains the model and runs the full in-process pipeline.
No Kafka, PostgreSQL, or Redis required for local development.

Usage:
    python scripts/quickstart.py
    python scripts/quickstart.py --events 1000
    python scripts/quickstart.py --skip-train
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM Data Pipeline Quickstart")
    parser.add_argument("--events",      type=int, default=300,  help="Events to process")
    parser.add_argument("--skip-train",  action="store_true",    help="Skip model training")
    parser.add_argument("--n-samples",   type=int, default=2000, help="Training samples")
    args = parser.parse_args()

    print("=" * 60)
    print("  LLM Data Pipeline — Quickstart")
    print("=" * 60)

    if not args.skip_train:
        print(f"\n[1/3] Training ML model on {args.n_samples} samples…")
        from backend.ml.train import train_and_save
        metrics = train_and_save(n_samples=args.n_samples)
        print(f"  ✓ ROC-AUC : {metrics.get('roc_auc', '?')}")
        print(f"  ✓ F1 Score: {metrics.get('f1', '?')}")
        print(f"  ✓ Recall  : {metrics.get('recall', '?')}")
    else:
        print("\n[1/3] Skipping model training (--skip-train).")

    print(f"\n[2/3] Running pipeline with {args.events} events…")
    from backend.pipeline import InProcessPipeline
    pipe  = InProcessPipeline(
        n_events    = args.events,
        batch_size  = 50,
        train_model = False,
    )
    stats = pipe.run()

    print("\n[3/3] Pipeline complete!")
    print("=" * 60)
    print("  RESULTS")
    print("=" * 60)
    print(f"  Events processed  : {stats['processed']}")
    print(f"  Fraud detected    : {stats['fraud_detected']}")
    print(f"  Alerts triggered  : {stats['alerts']}")
    print(f"  Insights generated: {stats['insights']}")
    print("=" * 60)
    print("\nNext steps:")
    print("  • Run dashboard : streamlit run dashboard/app.py")
    print("  • Run API server: python -m backend.api.main")
    print("  • Run tests     : pytest tests/ -v")
    print("  • Full stack    : cd infrastructure/docker && docker-compose up")


if __name__ == "__main__":
    main()
