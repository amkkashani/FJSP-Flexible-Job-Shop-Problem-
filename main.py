"""Main entry point for FJSP solver."""

import sys
import json
from pathlib import Path
from typing import List

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from models import Problem
from models.remaining import RemainingSection
from solvers import GreedySolver
from evaluation import WeightedEvaluator
from output_generator import generate_outputs
from Visual_Report.flow_animator import FlowAnimator
from Visual_Report.frame_by_frame import generate_frame_by_frame


def load_remaining_sections(file_path: str) -> List[RemainingSection]:
    """
    Load remaining sections from JSON file.

    Args:
        file_path: Path to remaining.json file

    Returns:
        List of RemainingSection objects
    """
    if not Path(file_path).exists():
        return []
    with open(file_path, 'r') as f:
        data = json.load(f)
    return [RemainingSection.from_dict(s) for s in data.get("sections", [])]


def main():
    """Main function to run the FJSP solver."""
    # Configuration path
    config_path = "config/config.json"

    print("=" * 60)
    print("FLEXIBLE JOB SHOP SCHEDULING PROBLEM SOLVER")
    print("=" * 60)

    # Load config to get data file path and sheet dimensions
    with open(config_path, 'r') as f:
        config = json.load(f)

    data_file = config.get("data_file", "data/5693_cleaned.xlsx")
    sheet_x = config.get("sheet_X", 2.0)  # Sheet width in meters
    sheet_y = config.get("sheet_Y", 1.8)  # Sheet height in meters

    # Load evaluator parameters
    evaluator_config = config.get("evaluator", {})
    alpha = evaluator_config.get("alpha", 1.0)
    beta = evaluator_config.get("beta", 0.5)
    gamma = evaluator_config.get("gamma", 0.3)

    # Check if data file exists
    if not Path(data_file).exists():
        print(f"\nWarning: Data file not found at {data_file}")
        print("Please update 'data_file' in config/config.json")
        print("\nRunning with sample data for demonstration...")
        run_with_sample_data()
        return

    # Load problem
    print(f"\nLoading problem from {data_file}...")
    problem = Problem.load_from_files(data_file, config_path)

    print(f"\nProblem loaded:")
    print(f"  - Parts: {problem.num_parts()}")
    print(f"  - Products: {problem.num_products()}")
    print(f"  - Stations: {len(problem.stations)}")
    print(f"  - Sheet capacity: {problem.sheet_capacity} m2")
    print(f"  - Sheet dimensions: {problem.sheet_width}m x {problem.sheet_height}m")
    print(f"  - Total parts area: {problem.total_parts_area():.4f} m2")
    print(f"  - Materials: {len(problem.get_unique_materials())}")

    # Create evaluator
    evaluator = WeightedEvaluator(alpha=alpha, beta=beta, gamma=gamma)
    print(f"\nEvaluator: {evaluator}")

    # Load remaining sections from previous runs
    remaining_file = config.get("remaining_file", "config/remaining.json")
    remaining_sections = load_remaining_sections(remaining_file)
    print(f"  - Remaining sections loaded: {len(remaining_sections)}")

    # Load remaining filter settings
    remaining_filter = config.get("remaining_filter", {})
    remaining_min_width = remaining_filter.get("min_width", 0.1)
    remaining_min_height = remaining_filter.get("min_height", 0.1)
    remaining_min_area = remaining_filter.get("min_area", 0.01)

    # Create and run solver
    solver = GreedySolver(
        sort_by='area_desc',
        remaining_min_width=remaining_min_width,
        remaining_min_height=remaining_min_height,
        remaining_min_area=remaining_min_area
    )
    print(f"Solver: {solver}")
    print(f"  - Remaining filter: min_width={remaining_min_width}m, min_height={remaining_min_height}m, min_area={remaining_min_area}m²")

    print("\nSolving...")
    solution = solver.solve(problem, evaluator, remaining_sections)

    # Print remaining sections summary
    print(solution.get_remaining_summary())

    # Print results
    print(solution.summary(problem))

    # Print fitness
    fitness = evaluator.evaluate(solution, problem)
    print(f"\nFitness (weighted objective): {fitness:.4f}")

    # Print detailed components
    components = evaluator.get_components(solution, problem)
    print("\nEvaluation components:")
    for key, value in components.items():
        print(f"  - {key}: {value:.4f}")

    # Validate solution
    print(f"\nSolution valid: {solution.is_valid(problem)}")

    # Generate outputs (CSV files and sheet images)
    output_folder = generate_outputs(
        solution,
        problem,
        data_file,
        problem.sheet_width,
        problem.sheet_height
    )

    # Save remaining sections to output folder (user can manually copy to config for next run)
    remaining_output_path = output_folder / "remaining.json"
    solution.save_remaining_sections(str(remaining_output_path))
    print(f"\nRemaining sections saved to: {remaining_output_path}")

    # Generate flow animation and Gantt chart based on config
    report_config = config.get("report", {})
    generate_animation = report_config.get("generate_animation", True)
    generate_gantt = report_config.get("generate_gantt_chart", True)
    generate_fbf = report_config.get("generate_frame_by_frame", False)
    generate_flow_frames_v9 = report_config.get("generate_flow_frames_v9", False)

    # Load animation settings
    animation_config = config.get("animation_settings", {})
    fps = animation_config.get("fps", 15)
    duration_seconds = animation_config.get("duration_seconds", 30)
    end_hold_seconds = animation_config.get("end_hold_seconds", 5)
    max_sheets = animation_config.get("max_sheets", None)
    # If max_sheets is None, use all sheets from solution
    if max_sheets is None:
        max_sheets = len(solution.sheets)

    # Load frame-by-frame settings
    fbf_config = config.get("frame_by_frame_settings", {})
    fbf_max_frames = fbf_config.get("max_frames", 100)

    # Load flow-frames settings
    flow_frames_config = config.get("flow_frames_settings", {})
    flow_frames_max_frames = flow_frames_config.get("max_frames", fbf_max_frames)
    flow_frames_zip = flow_frames_config.get("zip", False)

    if generate_animation or generate_gantt or generate_fbf or generate_flow_frames_v9:
        print("\n" + "=" * 60)
        print("GENERATING REPORTS")
        print("=" * 60)
        animator = FlowAnimator(solution, problem)

        if generate_animation:
            print("\nGenerating flow animation...")
            print(f"  Settings: fps={fps}, duration={duration_seconds}s, max_sheets={max_sheets}")
            animator.create_animation(
                output_path=str(output_folder / "flow_animation.gif"),
                fps=fps,
                duration_seconds=duration_seconds,
                end_hold_seconds=end_hold_seconds,
                max_sheets=max_sheets
            )
        else:
            print("\nSkipping animation generation (disabled in config)")

        if generate_gantt:
            print("\nGenerating Gantt chart...")
            animator.create_gantt_chart(
                output_path=str(output_folder / "gantt_chart.png")
                # max_sheets=None means show all sheets
            )
        else:
            print("\nSkipping Gantt chart generation (disabled in config)")

        if generate_fbf:
            print("\nGenerating frame-by-frame visualization...")
            print(f"  Settings: max_frames={fbf_max_frames}")
            generate_frame_by_frame(
                solution, problem,
                output_dir=str(output_folder / "frame_by_frame"),
                max_frames=fbf_max_frames
            )
        else:
            print("\nSkipping frame-by-frame generation (disabled in config)")

        if generate_flow_frames_v9:
            print("\nGenerating flow frames (v9)...")
            from Visual_Report.generate_flow_frames_v9 import generate_flow_frames
            print(f"  Settings: max_frames={flow_frames_max_frames}, zip={flow_frames_zip}")
            sheet_parts_path = output_folder / "sheet_parts.csv"
            events_path = output_folder / "event_summary.csv"
            flow_frames_dir = output_folder / "flow_frames_v9"
            if sheet_parts_path.exists() and events_path.exists():
                generate_flow_frames(
                    config_path=config_path,
                    sheet_parts_path=str(sheet_parts_path),
                    events_path=str(events_path),
                    outdir=str(flow_frames_dir),
                    max_frames=flow_frames_max_frames,
                    zip_output=flow_frames_zip
                )
            else:
                print("\nSkipping flow frames (v9) - missing sheet_parts.csv or event_summary.csv")
        else:
            print("\nSkipping flow frames (v9) generation (disabled in config)")
    else:
        print("\n" + "=" * 60)
        print("Skipping report generation (all reports disabled in config)")
        print("=" * 60)

    print(f"\n" + "=" * 60)
    print(f"COMPLETED - Output folder: {output_folder}")
    print("=" * 60)

    return solution


def run_with_sample_data():
    """Run solver with sample data for demonstration."""
    import pandas as pd

    # Create sample data
    sample_data = {
        'ElemIdent': ['E001', 'E002', 'E003', 'E004', 'E005'],
        'length': [1000, 800, 1200, 600, 900],
        'width': [500, 400, 600, 300, 450],
        'area': [0.5, 0.32, 0.72, 0.18, 0.405],
        'mat': ['MAT_A', 'MAT_A', 'MAT_B', 'MAT_A', 'MAT_B'],
        'quantity': [3, 2, 1, 4, 2],
        'Info8': ['PROD_001', 'PROD_001', 'PROD_002', 'PROD_001', 'PROD_002'],
        'wa': [16, 12, 20, 8, 14],
        'wf': [13, 10, 15, 6, 11],
        'wd': [0, 5, 8, 0, 6],
        'wo': [10, 8, 12, 4, 9],
        'wg': [7, 5, 10, 3, 6],
        'wv': [0, 0, 5, 0, 3],
        'wx': [4, 3, 6, 2, 4]
    }
    df = pd.DataFrame(sample_data)

    # Load station config
    with open("config/config.json", 'r') as f:
        station_config = json.load(f)

    # Create problem
    problem = Problem.load_from_dataframe(df, station_config)

    print(f"\nSample problem created:")
    print(f"  - Parts: {problem.num_parts()}")
    print(f"  - Products: {problem.num_products()}")
    print(f"  - Stations: {len(problem.stations)}")
    print(f"  - Sheet capacity: {problem.sheet_capacity} m2")
    print(f"  - Total parts area: {problem.total_parts_area():.4f} m2")

    # Load evaluator parameters from config
    with open('config/config.json', 'r') as f:
        config = json.load(f)
    evaluator_config = config.get("evaluator", {})
    alpha = evaluator_config.get("alpha", 1.0)
    beta = evaluator_config.get("beta", 0.5)
    gamma = evaluator_config.get("gamma", 0.3)

    # Create evaluator and solver
    evaluator = WeightedEvaluator(alpha=alpha, beta=beta, gamma=gamma)
    solver = GreedySolver(sort_by='area_desc')

    print(f"\nEvaluator: {evaluator}")
    print(f"Solver: {solver}")

    print("\nSolving...")
    # No remaining sections for sample data
    solution = solver.solve(problem, evaluator, [])

    # Print results
    print(solution.summary(problem))

    # Print fitness
    fitness = evaluator.evaluate(solution, problem)
    print(f"\nFitness (weighted objective): {fitness:.4f}")

    # Show all sheets
    print("\nAll sheets:")
    for sheet in solution.sheets:
        print(f"  {sheet}")


if __name__ == "__main__":
    main()
