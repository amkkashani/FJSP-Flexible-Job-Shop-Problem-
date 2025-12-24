"""Main entry point for FJSP solver."""

import sys
import json
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from models import Problem
from solvers import GreedySolver
from evaluation import WeightedEvaluator
from output_generator import generate_outputs


def main():
    """Main function to run the FJSP solver."""
    # Configuration path
    config_path = "config/stations.json"

    print("=" * 60)
    print("FLEXIBLE JOB SHOP SCHEDULING PROBLEM SOLVER")
    print("=" * 60)

    # Load config to get data file path and sheet dimensions
    with open(config_path, 'r') as f:
        config = json.load(f)

    data_file = config.get("data_file", "data/5693_cleaned.xlsx")
    sheet_x = config.get("sheet_X", 2.0)  # Sheet width in meters
    sheet_y = config.get("sheet_Y", 1.8)  # Sheet height in meters

    # Check if data file exists
    if not Path(data_file).exists():
        print(f"\nWarning: Data file not found at {data_file}")
        print("Please update 'data_file' in config/stations.json")
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
    evaluator = WeightedEvaluator(alpha=1.0, beta=0.5)
    print(f"\nEvaluator: {evaluator}")

    # Create and run solver
    solver = GreedySolver(sort_by='area_desc')
    print(f"Solver: {solver}")

    print("\nSolving...")
    solution = solver.solve(problem, evaluator)

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
    with open("config/stations.json", 'r') as f:
        station_config = json.load(f)

    # Create problem
    problem = Problem.load_from_dataframe(df, station_config)

    print(f"\nSample problem created:")
    print(f"  - Parts: {problem.num_parts()}")
    print(f"  - Products: {problem.num_products()}")
    print(f"  - Stations: {len(problem.stations)}")
    print(f"  - Sheet capacity: {problem.sheet_capacity} m2")
    print(f"  - Total parts area: {problem.total_parts_area():.4f} m2")

    # Create evaluator and solver
    evaluator = WeightedEvaluator(alpha=1.0, beta=0.5)
    solver = GreedySolver(sort_by='area_desc')

    print(f"\nEvaluator: {evaluator}")
    print(f"Solver: {solver}")

    print("\nSolving...")
    solution = solver.solve(problem, evaluator)

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
