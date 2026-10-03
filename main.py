"""
MovieLens 32M Unified Command-Line Interface (CLI)
==================================================
Provides convenient CLI commands to execute question pipelines, run tests,
launch the Streamlit dashboard, or start the FastAPI service.

Usage:
    python main.py run-all                # Run all 11 solution sections and generate reports
    python main.py section <num>          # Run a specific solution section (1 to 11)
    python main.py test                   # Run the unit test suite
    python main.py app                    # Launch interactive Streamlit dashboard
    python main.py api [--port 8000]      # Start FastAPI REST microservice
"""

import sys
import argparse
import subprocess
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


def command_run_all(args):
    """Executes the complete master pipeline across all sections."""
    from run_all import main as run_all_main
    run_all_main()


def command_run_section(args):
    """Executes a single solution section by number."""
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
    print(f"\n>>> Running Section {sec_num} ({sec_map[sec_num]})...\n")
    import importlib
    mod = importlib.import_module(mod_name)
    func_name = f"run_section_{sec_num}"
    if hasattr(mod, func_name):
        getattr(mod, func_name)()
    else:
        print(f"Error: Function {func_name} not found in {mod_name}.")


def command_test(args):
    """Runs the unit test suite."""
    print("\n>>> Running Automated Unit Tests...\n")
    import unittest
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir=str(PROJECT_ROOT / "tests"), pattern="test_*.py")
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)


def command_app(args):
    """Launches the Streamlit interactive dashboard."""
    app_path = PROJECT_ROOT / "app" / "streamlit_app.py"
    print(f"\n>>> Launching Streamlit Web App from {app_path}...\n")
    subprocess.run(["streamlit", "run", str(app_path)])


def command_api(args):
    """Starts the FastAPI production REST service."""
    port = args.port or 8000
    host = args.host or "0.0.0.0"
    print(f"\n>>> Starting FastAPI REST microservice on http://{host}:{port}...\n")
    import uvicorn
    uvicorn.run("app.api:app", host=host, port=port, reload=True)


def command_benchmark(args):
    """Executes automated benchmark reproduction suite."""
    from scripts.reproduce_benchmarks import run_benchmark_reproduction
    run_benchmark_reproduction(max_ratings=args.max_ratings)


def command_download_data(args):
    """Downloads official MovieLens dataset archives."""
    from scripts.download_dataset import download_and_extract
    download_and_extract(dataset_name=args.dataset, target_dir=args.output)


def main():
    parser = argparse.ArgumentParser(
        description="MovieLens 32M Recommendation System Master CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py run-all
    python main.py benchmark
    python main.py download-data --dataset ml-latest-small
    python main.py test
    python main.py app
    python main.py api --port 8000
        """
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Command: run-all
    p_all = subparsers.add_parser("run-all", help="Execute all solution sections and generate reports")
    p_all.set_defaults(func=command_run_all)

    # Command: section <N>
    p_sec = subparsers.add_parser("section", help="Run a specific solution section (1-11)")
    p_sec.add_argument("number", type=int, help="Section number (1 to 11)")
    p_sec.set_defaults(func=command_run_section)

    # Command: test
    p_test = subparsers.add_parser("test", help="Run unit test suite")
    p_test.set_defaults(func=command_test)

    # Command: benchmark
    p_bm = subparsers.add_parser("benchmark", help="Run offline temporal benchmark reproduction")
    p_bm.add_argument("--max-ratings", type=int, default=100000, help="Max ratings to sample for evaluation")
    p_bm.set_defaults(func=command_benchmark)

    # Command: download-data
    p_dl = subparsers.add_parser("download-data", help="Download official MovieLens dataset")
    p_dl.add_argument("--dataset", choices=["ml-latest-small", "ml-32m", "ml-25m", "ml-100k"], default="ml-latest-small", help="Dataset name")
    p_dl.add_argument("--output", default="./data", help="Output directory")
    p_dl.set_defaults(func=command_download_data)

    # Command: app
    p_app = subparsers.add_parser("app", help="Launch Streamlit web dashboard")
    p_app.set_defaults(func=command_app)

    # Command: api
    p_api = subparsers.add_parser("api", help="Start FastAPI REST service")
    p_api.add_argument("--host", type=str, default="0.0.0.0", help="Host address (default: 0.0.0.0)")
    p_api.add_argument("--port", type=int, default=8000, help="Port number (default: 8000)")
    p_api.set_defaults(func=command_api)

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
    else:
        args.func(args)


if __name__ == "__main__":
    main()
