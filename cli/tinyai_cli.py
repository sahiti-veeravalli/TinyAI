#!/usr/bin/env python3
"""
TinyAI CLI: Headless command-line interface for edge model optimization and Pareto analysis.
"""

import sys
import argparse
import json
from tinyai_sdk import TinyAIClient


def main():
    parser = argparse.ArgumentParser(
        description="TinyAI: Domain-Aware Budget-Constrained Tabular Model Compression"
    )
    parser.add_argument("--server", default="http://127.0.0.1:8000", help="TinyAI server URL")

    subparsers = parser.add_subparsers(dest="command", help="Sub-commands")

    # Health command
    subparsers.add_parser("health", help="Check system status and hardware environment")

    # Datasets command
    subparsers.add_parser("datasets", help="List registered datasets")

    # Sample command
    sample_p = subparsers.add_parser("load-sample", help="Load built-in canonical sample dataset")
    sample_p.add_argument("--id", default="fraud_detection_benchmark", help="Sample dataset ID")

    # Optimize command
    opt_p = subparsers.add_parser("optimize", help="Launch Pareto compression experiment")
    opt_p.add_argument("--dataset-id", required=True, help="Dataset ID")
    opt_p.add_argument("--target", required=True, help="Target column name")
    opt_p.add_argument("--baseline", default="lightgbm", help="Baseline model family")
    opt_p.add_argument("--max-size-kb", type=float, default=250.0, help="Max disk size (KB)")
    opt_p.add_argument("--max-lat-ms", type=float, default=1.0, help="Max p50 latency (ms)")
    opt_p.add_argument("--min-quality", type=float, default=0.85, help="Min quality score")
    opt_p.add_argument("--priority", default="balanced", choices=["balanced", "lowest_latency", "smallest_size", "highest_quality"])

    # Export command
    exp_p = subparsers.add_parser("export", help="Export edge deployment bundle for a model")
    exp_p.add_argument("--model-id", required=True, help="Model ID from registry")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Add SDK to sys.path if running locally
    client = TinyAIClient(base_url=args.server)

    try:
        if args.command == "health":
            res = client.get_health()
            print(json.dumps(res, indent=2))

        elif args.command == "datasets":
            res = client.list_datasets()
            print(json.dumps(res, indent=2))

        elif args.command == "load-sample":
            res = client.load_sample_dataset(sample_id=args.id)
            print(f"Loaded: {res.get('message')}")

        elif args.command == "optimize":
            print(f"Launching optimization on dataset '{args.dataset_id}'...")
            res = client.launch_optimization(
                dataset_id=args.dataset_id,
                target_column=args.target,
                baseline_family=args.baseline,
                max_disk_size_kb=args.max_size_kb,
                max_latency_ms=args.max_lat_ms,
                min_quality_score=args.min_quality,
                priority=args.priority
            )
            top = res.get("recommended_candidate")
            print("\n=== OPTIMIZATION COMPLETE ===")
            if top:
                print(f"Recommended Model: {top['name']}")
                print(f"Quality Score:     {top['quality_score']:.4f}")
                print(f"p50 Latency:       {top['latency_p50_ms']:.3f} ms")
                print(f"Disk Size:         {top['disk_size_kb']:.1f} KB")
                print(f"Compression Ratio: {top['compression_ratio']:.1f}x")
                print(f"\nRationale: {top['recommendation_rationale']}")
            else:
                print("No feasible candidate satisfied all constraints.")

        elif args.command == "export":
            res = client.export_bundle(model_id=args.model_id)
            print(f"Bundle Exported to: {res.get('bundle_directory')}")
            for k, v in res.get("generated_files", {}).items():
                print(f"  - {k}: {v}")

    except Exception as e:
        print(f"Error executing command '{args.command}': {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
