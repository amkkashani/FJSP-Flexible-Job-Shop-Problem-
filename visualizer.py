"""Visualizer for FJSP solutions."""

import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent))

from models import Problem
from solution import Solution
from models.sheet import Sheet


class SolutionVisualizer:
    """Visualizer for FJSP solutions showing sheet allocations and product distributions."""

    def __init__(self, solution: Solution, problem: Problem):
        self.solution = solution
        self.problem = problem

    def get_sheet_products(self, sheet: Sheet) -> Dict[str, List[str]]:
        """Get products and their parts in a sheet."""
        products = defaultdict(list)
        for part in sheet.assigned_parts:
            products[part.product_id].append(part.id)
        return dict(products)

    def get_product_sheet_distribution(self) -> Dict[str, List[str]]:
        """Get which sheets contain parts of each product."""
        distribution = defaultdict(set)
        for sheet in self.solution.sheets:
            for part in sheet.assigned_parts:
                distribution[part.product_id].add(sheet.id)
        return {k: sorted(list(v)) for k, v in distribution.items()}

    def print_sheet_summary(self):
        """Print summary of all sheets."""
        print("\n" + "=" * 70)
        print("SHEET SUMMARY")
        print("=" * 70)
        print(f"Total sheets: {len(self.solution.sheets)}")
        print(f"Total waste: {self.solution.get_total_waste():.4f} m2")
        print(f"Average utilization: {(1 - self.solution.get_total_waste() / (len(self.solution.sheets) * self.problem.sheet_capacity)) * 100:.2f}%")
        print("=" * 70)

    def print_sheets_detail(self, max_sheets: int = None):
        """Print detailed view of each sheet with products."""
        sheets = self.solution.sheets
        if max_sheets:
            sheets = sheets[:max_sheets]

        print("\n" + "=" * 70)
        print("SHEET DETAILS")
        print("=" * 70)

        for sheet in sheets:
            products = self.get_sheet_products(sheet)
            utilization = (sheet.total_area() / sheet.capacity) * 100

            print(f"\n+{'-' * 68}+")
            print(f"| {sheet.id:<66} |")
            print(f"+{'-' * 68}+")
            print(f"| Parts: {sheet.num_parts():<10} Utilization: {utilization:>6.2f}%{' ' * 32}|")
            print(f"| Area: {sheet.total_area():.4f} / {sheet.capacity:.4f} m2   Waste: {sheet.waste():.4f} m2{' ' * 19}|")
            print(f"+{'-' * 68}+")
            print(f"| {'PRODUCTS:':<66} |")

            for product_id, part_ids in sorted(products.items()):
                line = f"   - {product_id}: {len(part_ids)} part(s)"
                print(f"| {line:<66} |")

            print(f"+{'-' * 68}+")

        if max_sheets and len(self.solution.sheets) > max_sheets:
            print(f"\n... and {len(self.solution.sheets) - max_sheets} more sheets")

    def print_product_distribution(self):
        """Print which sheets contain each product's parts."""
        distribution = self.get_product_sheet_distribution()

        print("\n" + "=" * 70)
        print("PRODUCT DISTRIBUTION ACROSS SHEETS")
        print("=" * 70)
        print(f"{'Product ID':<25} {'Sheets':<8} {'Sheet IDs'}")
        print("-" * 70)

        for product_id in sorted(distribution.keys()):
            sheet_ids = distribution[product_id]
            sheets_str = ", ".join(sheet_ids[:5])
            if len(sheet_ids) > 5:
                sheets_str += f", ... (+{len(sheet_ids) - 5} more)"
            print(f"{product_id:<25} {len(sheet_ids):<8} {sheets_str}")

        print("-" * 70)

    def print_product_completion_times(self):
        """Print completion times for each product."""
        print("\n" + "=" * 70)
        print("PRODUCT COMPLETION TIMES")
        print("=" * 70)
        print(f"{'Product ID':<25} {'Parts':<8} {'Completion Time':<20}")
        print("-" * 70)

        completion_times = self.solution.get_all_product_completion_times(self.problem)

        for product_id in sorted(completion_times.keys()):
            product = self.problem.products[product_id]
            time = completion_times[product_id]
            print(f"{product_id:<25} {len(product.part_ids):<8} {time:<20.2f}")

        print("-" * 70)
        avg_time = sum(completion_times.values()) / len(completion_times) if completion_times else 0
        print(f"{'Average:':<25} {'':<8} {avg_time:<20.2f}")
        print(f"{'Makespan:':<25} {'':<8} {self.solution.get_makespan():<20.2f}")

    def print_material_distribution(self):
        """Print material distribution across sheets."""
        print("\n" + "=" * 70)
        print("MATERIAL DISTRIBUTION PER SHEET")
        print("=" * 70)

        for sheet in self.solution.sheets[:10]:  # Show first 10 sheets
            materials = defaultdict(int)
            for part in sheet.assigned_parts:
                materials[part.material] += 1

            print(f"\n{sheet.id}:")
            for material, count in sorted(materials.items()):
                print(f"  - {material}: {count} part(s)")

        if len(self.solution.sheets) > 10:
            print(f"\n... and {len(self.solution.sheets) - 10} more sheets")

    def print_schedule_gantt_text(self, max_sheets: int = 5):
        """Print text-based Gantt chart for sheet schedules."""
        print("\n" + "=" * 70)
        print("SCHEDULE (Text Gantt Chart)")
        print("=" * 70)

        makespan = self.solution.get_makespan()
        if makespan == 0:
            print("No schedule available.")
            return

        width = 60  # Width of the chart
        scale = width / makespan

        sheets = self.solution.sheets[:max_sheets]

        for sheet in sheets:
            assignments = self.solution.schedule.get(sheet.id, [])
            if not assignments:
                continue

            print(f"\n{sheet.id}:")

            # Create timeline
            timeline = [' '] * width

            for assignment in assignments:
                start_pos = int(assignment.start_time * scale)
                end_pos = int(assignment.end_time * scale)
                start_pos = min(start_pos, width - 1)
                end_pos = min(end_pos, width)

                station_char = assignment.station_name[1].upper()  # Use second char (a, f, d, o, g, v, x)

                for i in range(start_pos, end_pos):
                    timeline[i] = station_char

            print(f"  |{''.join(timeline)}|")
            print(f"  0{' ' * (width - 10)}Makespan: {makespan:.0f}")

        print("\nLegend: A=wa, F=wf, D=wd, O=wo, G=wg, V=wv, X=wx")

    def print_full_report(self, max_sheets_detail: int = 10):
        """Print a full visualization report."""
        self.print_sheet_summary()
        self.print_sheets_detail(max_sheets=max_sheets_detail)
        self.print_product_distribution()
        self.print_product_completion_times()
        self.print_schedule_gantt_text()

    def export_to_csv(self, output_path: str):
        """Export sheet-product mapping to CSV."""
        import csv

        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Sheet ID', 'Part ID', 'Product ID', 'Material', 'Area (m²)',
                           'Length (mm)', 'Width (mm)'])

            for sheet in self.solution.sheets:
                for part in sheet.assigned_parts:
                    writer.writerow([
                        sheet.id,
                        part.id,
                        part.product_id,
                        part.material,
                        f"{part.area:.6f}",
                        part.length,
                        part.width
                    ])

        print(f"\nExported to {output_path}")

    def export_summary_to_csv(self, output_path: str):
        """Export sheet summary to CSV."""
        import csv

        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Sheet ID', 'Num Parts', 'Num Products', 'Used Area (m²)',
                           'Capacity (m²)', 'Waste (m²)', 'Utilization (%)', 'Products'])

            for sheet in self.solution.sheets:
                products = self.get_sheet_products(sheet)
                utilization = (sheet.total_area() / sheet.capacity) * 100
                product_list = "; ".join(sorted(products.keys()))

                writer.writerow([
                    sheet.id,
                    sheet.num_parts(),
                    len(products),
                    f"{sheet.total_area():.6f}",
                    f"{sheet.capacity:.6f}",
                    f"{sheet.waste():.6f}",
                    f"{utilization:.2f}",
                    product_list
                ])

        print(f"\nExported summary to {output_path}")


def visualize_solution(solution: Solution, problem: Problem, export_csv: bool = False):
    """Main function to visualize a solution."""
    viz = SolutionVisualizer(solution, problem)
    viz.print_full_report()

    if export_csv:
        viz.export_to_csv("output/sheet_parts.csv")
        viz.export_summary_to_csv("output/sheet_summary.csv")

    return viz


if __name__ == "__main__":
    import json
    from models import Problem
    from solvers import GreedySolver
    from evaluation import WeightedEvaluator

    # Load config
    config_path = "config/stations.json"
    with open(config_path, 'r') as f:
        config = json.load(f)

    data_file = config.get("data_file", "data/5693_cleaned.xlsx")

    # Load problem and solve
    print("Loading problem...")
    problem = Problem.load_from_files(data_file, config_path)

    print("Solving...")
    solver = GreedySolver(sort_by='area_desc')
    evaluator = WeightedEvaluator(alpha=1.0, beta=0.5)
    solution = solver.solve(problem, evaluator)

    # Visualize
    visualize_solution(solution, problem)
