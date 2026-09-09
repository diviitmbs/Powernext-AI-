"""
Main entry point for CPRI PowerNext-AI Hackathon Pipeline.

Usage:
    python main.py
    python main.py --team-name YourTeamName --data-path data/CPRI_Hackathon_Screening_Dataset_PARTICIPANT.xlsx
"""
import argparse
import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import DEFAULT_TEAM_NAME, DELIVERABLES_DIR
from src.pipeline import run_cpri_pipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description="CPRI PowerNext-AI End-to-End Machine Learning Solution."
    )
    parser.add_argument(
        "--team-name",
        type=str,
        default=os.getenv("TEAM_NAME", DEFAULT_TEAM_NAME),
        help="Team name to use for generated submission files (default: PowerNext_AI).",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to CPRI_Hackathon_Screening_Dataset_PARTICIPANT.xlsx.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DELIVERABLES_DIR),
        help="Directory to save output files (default: deliverables).",
    )
    parser.add_argument(
        "--skip-tuning",
        action="store_true",
        help="Skip RandomizedSearchCV hyperparameter search and use calibrated defaults.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    results = run_cpri_pipeline(
        team_name=args.team_name,
        data_path=args.data_path,
        output_dir=args.output_dir,
        tune_regressor=not args.skip_tuning,
    )
    return results


if __name__ == "__main__":
    main()
