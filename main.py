"""
MovieLens 32M Unified Command-Line Interface (CLI)
==================================================
Provides commands for the question pipelines, tests, offline artifact
building, benchmarks, and FastAPI service.
"""

import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


def command_run_all(args):
    from run_all import main as run_all_main
    run_all_main()


def command_run_section(args):
    sec_num = args.number
    sec_map = {
        1: "01_dataset_understanding_validation",
        2: "02_exploratory_data_analysis",
        3: "03_feature_engineering",
        4: "04_content_based_recommendation",
        5: "05_collaborative_filtering",
        6: "06_preference_prediction_supervised",
        7: "07_dimensionality_reduction",
        8: "08_hybrid_recommendation_system",
        9: "09_evaluation_and_ranking",
        10: "10_cold_start_and_production",
        11: "11_advanced_challenges",
    }
    if sec_num not in sec_map:
        print(f"Error: Section {sec_num} does not exist. Choose between 1 and 11.")
        return

    mod_name = f"solutions.{sec_map[sec_num]}"
    import importlib
    mod = importlib.import_module(mod_name)
    func_name = f"run_section_{sec_num}"
    if hasattr(mod, func_name):
        getattr(mod, func_name)()
    else:
        print(f"Error: Function {func_name} not found in {mod_name}.")


def command_test(args):
    import unittest
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir=str(PROJECT_ROOT / "tests"), pattern="test_*.py")
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)


def command_build_artifacts(args):
    from scripts.build_artifacts import build_artifact
    build_artifact(
        output=Path(args.output),
        max_ratings=args.max_ratings,
        n_svd_components=args.svd_components,
    )


def command_api(args):
    import uvicorn
    uvicorn.run(
        "app.api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


def command_benchmark(args):
    from scripts.reproduce_benchmarks import run_benchmark_reproduction
    run_benchmark_reproduction(max_ratings=args.max_ratings)


def command_download_data(args):
    from scripts.download_dataset import download_and_extract
    download_and_extract(dataset_name=args.dataset, target_dir=args.output)


def main():
    parser = argparse.ArgumentParser(
        description="MovieLens 32M Recommendation System Master CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    p_all = subparsers.add_parser("run-all", help="Execute all solution sections")
    p_all.set_defaults(func=command_run_all)

    p_sec = subparsers.add_parser("section", help="Run a specific solution section (1-11)")
    p_sec.add_argument("number", type=int)
    p_sec.set_defaults(func=command_run_section)

    p_test = subparsers.add_parser("test", help="Run unit test suite")
    p_test.set_defaults(func=command_test)

    p_build = subparsers.add_parser("build-artifacts", help="Build deployment-ready model artifacts offline")
    p_build.add_argument("--output", default="artifacts/movielens_runtime.joblib")
    p_build.add_argument("--max-ratings", type=int, default=None)
    p_build.add_argument("--svd-components", type=int, default=16)
    p_build.set_defaults(func=command_build_artifacts)

    p_bm = subparsers.add_parser("benchmark", help="Run offline temporal benchmark reproduction")
    p_bm.add_argument("--max-ratings", type=int, default=100000)
    p_bm.set_defaults(func=command_benchmark)

    p_dl = subparsers.add_parser("download-data", help="Download official MovieLens dataset")
    p_dl.add_argument(
        "--dataset",
        choices=["ml-latest-small", "ml-32m", "ml-25m", "ml-100k"],
        default="ml-latest-small",
    )
    p_dl.add_argument("--output", default="./data")
    p_dl.set_defaults(func=command_download_data)

    p_api = subparsers.add_parser("api", help="Start FastAPI REST service")
    p_api.add_argument("--host", default="0.0.0.0")
    p_api.add_argument("--port", type=int, default=8000)
    p_api.add_argument("--reload", action="store_true", help="Enable development auto-reload")
    p_api.set_defaults(func=command_api)

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
    else:
        args.func(args)


if __name__ == "__main__":
    main()
