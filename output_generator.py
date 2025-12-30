"""Output generator for FJSP solutions - creates folder structure and sheet images."""

import os
import sys
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Optional
import csv

sys.path.insert(0, str(Path(__file__).parent))

from models import Problem
from models.sheet import Sheet
from solution import Solution

# Try to import matplotlib for image generation
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.patches import Rectangle
    MATPLOTLIB_AVAILABLE = True
except Exception as exc:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available or backend failed. Sheet images will not be generated.")
    print(f"Reason: {exc}")
    print("Install with: pip install matplotlib")


class OutputGenerator:
    """Generates output files and sheet images for FJSP solutions."""

    def __init__(self, solution: Solution, problem: Problem, data_file: str):
        self.solution = solution
        self.problem = problem
        self.data_file = data_file
        self.output_base = Path("output")
        self.run_folder = None

    def create_output_folder(self) -> Path:
        """Create output folder structure based on input file name."""
        # Get base name from data file (without extension)
        data_name = Path(self.data_file).stem

        # Create timestamp for unique folder
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create folder path: output/{data_name}_{timestamp}/
        self.run_folder = self.output_base / f"{data_name}_{timestamp}"

        # Create directories
        self.run_folder.mkdir(parents=True, exist_ok=True)
        (self.run_folder / "sheets").mkdir(exist_ok=True)

        print(f"\nOutput folder created: {self.run_folder}")
        return self.run_folder

    def generate_sheet_image(self, sheet: Sheet, output_path: Path,
                            sheet_width: float = 2.0, sheet_height: float = 1.8):
        """
        Generate an image for a single sheet showing parts placement.

        Args:
            sheet: The sheet to visualize
            output_path: Path to save the image
            sheet_width: Sheet width in meters (from config sheet_X)
            sheet_height: Sheet height in meters (from config sheet_Y)
        """
        if not MATPLOTLIB_AVAILABLE:
            return

        fig, ax = plt.subplots(1, 1, figsize=(12, 9))
        draw_width = sheet.width or sheet_width
        draw_height = sheet.height or sheet_height

        # Draw sheet boundary
        sheet_rect = Rectangle((0, 0), draw_width, draw_height,
                               linewidth=3, edgecolor='black',
                               facecolor='lightgray', alpha=0.3)
        ax.add_patch(sheet_rect)

        # Get parts and group by product for coloring
        parts = sheet.assigned_parts
        products = {}
        for part in parts:
            if part.product_id not in products:
                products[part.product_id] = []
            products[part.product_id].append(part)

        # Color palette for different products
        colors = plt.cm.Set3(range(len(products)))
        product_colors = {pid: colors[i] for i, pid in enumerate(products.keys())}

        use_placements = bool(sheet.placements)

        if use_placements:
            for part in parts:
                placement = sheet.placements.get(part.id)
                if placement is None:
                    continue
                x, y, w, h, rotated = placement
                color = product_colors[part.product_id]
                part_rect = Rectangle((x, y), w, h,
                                      linewidth=1, edgecolor='black',
                                      facecolor=color, alpha=0.7)
                ax.add_patch(part_rect)

                label = part.id
                if rotated:
                    label = f"{label} (R)"
                ax.text(x + w / 2, y + h / 2,
                        label, ha='center', va='center', fontsize=5,
                        wrap=True)
        else:
            y_offset = 0.05
            x_offset = 0.05
            current_x = x_offset
            current_y = y_offset
            row_height = 0
            # padding = 0.02
            padding = 0

            for part in sorted(parts, key=lambda p: p.area, reverse=True):
                part_w = part.width / 1000
                part_h = part.length / 1000

                scale = min(0.3, (draw_width - 0.1) / max(part_w, 0.1))
                display_w = min(part_w * scale, draw_width - 0.1)
                display_h = min(part_h * scale, draw_height - 0.1)

                if current_x + display_w > draw_width - x_offset:
                    current_x = x_offset
                    current_y += row_height + padding
                    row_height = 0

                if current_y + display_h > draw_height - y_offset:
                    break

                color = product_colors[part.product_id]
                part_rect = Rectangle((current_x, current_y), display_w, display_h,
                                      linewidth=1, edgecolor='black',
                                      facecolor=color, alpha=0.7)
                ax.add_patch(part_rect)

                label = f"{part.product_id[:10]}\n{part.area:.3f}m2"
                ax.text(current_x + display_w/2, current_y + display_h/2,
                       label, ha='center', va='center', fontsize=6,
                       wrap=True)

                current_x += display_w + padding
                row_height = max(row_height, display_h)

        # Set axis properties
        ax.set_xlim(-0.1, draw_width + 0.1)
        ax.set_ylim(-0.1, draw_height + 0.1)
        ax.set_aspect('equal')
        ax.set_xlabel('Width (m)')
        ax.set_ylabel('Height (m)')

        # Title with sheet info
        utilization = (sheet.total_area() / sheet.capacity) * 100
        title = (f"{sheet.id}\n"
                f"Parts: {sheet.num_parts()} | "
                f"Products: {len(products)} | "
                f"Used: {sheet.total_area():.4f}m2 | "
                f"Utilization: {utilization:.1f}%")
        ax.set_title(title, fontsize=10)

        # Add legend for products
        legend_handles = []
        for pid in sorted(products.keys()):
            part_ids = sorted({p.id for p in products[pid]})
            row_ids = ", ".join(part_ids)
            label = f"{pid[:15]} ({row_ids})"
            legend_handles.append(patches.Patch(color=product_colors[pid], label=label))
        legend = ax.legend(handles=legend_handles, loc='upper left',
                          bbox_to_anchor=(1.02, 1), fontsize=8)

        save_kwargs = {"dpi": 150, "bbox_inches": "tight"}
        if legend is not None:
            save_kwargs["bbox_extra_artists"] = (legend,)
        plt.savefig(output_path, **save_kwargs)
        plt.close()

    def generate_all_sheet_images(self, sheet_width: float = 2.0, sheet_height: float = 1.8):
        """Generate images for all sheets."""
        if not MATPLOTLIB_AVAILABLE:
            print("Skipping sheet image generation (matplotlib not available)")
            return

        sheets_folder = self.run_folder / "sheets"
        total = len(self.solution.sheets)

        print(f"\nGenerating {total} sheet images...")

        for i, sheet in enumerate(self.solution.sheets):
            output_path = sheets_folder / f"{sheet.id}.png"
            self.generate_sheet_image(sheet, output_path, sheet_width, sheet_height)

            # Progress indicator
            if (i + 1) % 10 == 0 or i == total - 1:
                print(f"  Generated {i + 1}/{total} images")

        print(f"Sheet images saved to: {sheets_folder}")

    def export_sheet_parts_csv(self):
        """Export detailed sheet-parts mapping to CSV."""
        output_path = self.run_folder / "sheet_parts.csv"

        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Sheet ID', 'Part ID', 'ElemIdent', 'Product ID',
                           'Material', 'Area (m2)', 'Length (mm)', 'Width (mm)'])

            for sheet in self.solution.sheets:
                for part in sheet.assigned_parts:
                    writer.writerow([
                        sheet.id,
                        part.id,
                        part.elem_ident,
                        part.product_id,
                        part.material,
                        f"{part.area:.6f}",
                        part.length,
                        part.width
                    ])

        print(f"Sheet parts CSV saved to: {output_path}")

    def export_sheet_summary_csv(self):
        """Export sheet summary to CSV."""
        output_path = self.run_folder / "sheet_summary.csv"

        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Sheet ID', 'Num Parts', 'Num Products', 'Used Area (m2)',
                           'Capacity (m2)', 'Waste (m2)', 'Utilization (%)', 'Products'])

            for sheet in self.solution.sheets:
                products = {}
                for part in sheet.assigned_parts:
                    if part.product_id not in products:
                        products[part.product_id] = 0
                    products[part.product_id] += 1

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

        print(f"Sheet summary CSV saved to: {output_path}")

    def export_product_summary_csv(self):
        """Export product summary to CSV."""
        output_path = self.run_folder / "product_summary.csv"

        # Get product completion times
        completion_times = self.solution.get_all_product_completion_times(self.problem)

        # Get product-sheet distribution
        product_sheets = {}
        for sheet in self.solution.sheets:
            for part in sheet.assigned_parts:
                if part.product_id not in product_sheets:
                    product_sheets[part.product_id] = set()
                product_sheets[part.product_id].add(sheet.id)

        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Product ID', 'Num Parts', 'Num Sheets',
                           'Completion Time', 'Sheet IDs'])

            for product_id in sorted(self.problem.products.keys()):
                product = self.problem.products[product_id]
                sheets = sorted(product_sheets.get(product_id, []))
                completion = completion_times.get(product_id, 0)

                writer.writerow([
                    product_id,
                    len(product.part_ids),
                    len(sheets),
                    f"{completion:.2f}",
                    "; ".join(sheets)
                ])

        print(f"Product summary CSV saved to: {output_path}")

    def export_schedule_csv(self):
        """Export schedule to CSV."""
        output_path = self.run_folder / "schedule.csv"

        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Sheet ID', 'Station', 'Machine', 'Start Time', 'End Time', 'Duration'])

            for sheet_id, assignments in self.solution.schedule.items():
                for assignment in sorted(assignments, key=lambda a: a.start_time):
                    writer.writerow([
                        sheet_id,
                        assignment.station_name,
                        assignment.machine_index,
                        f"{assignment.start_time:.2f}",
                        f"{assignment.end_time:.2f}",
                        f"{assignment.duration:.2f}"
                    ])

        print(f"Schedule CSV saved to: {output_path}")

    def export_solution_summary(self):
        """Export overall solution summary to text file."""
        output_path = self.run_folder / "solution_summary.txt"

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("FJSP SOLUTION SUMMARY\n")
            f.write("=" * 60 + "\n\n")

            f.write(f"Input file: {self.data_file}\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("-" * 40 + "\n")
            f.write("PROBLEM STATISTICS\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total parts: {self.problem.num_parts()}\n")
            f.write(f"Total products: {self.problem.num_products()}\n")
            f.write(f"Total parts area: {self.problem.total_parts_area():.4f} m2\n")
            f.write(f"Sheet capacity: {self.problem.sheet_capacity} m2\n")
            f.write(f"Unique materials: {len(self.problem.get_unique_materials())}\n\n")

            f.write("-" * 40 + "\n")
            f.write("SOLUTION STATISTICS\n")
            f.write("-" * 40 + "\n")
            f.write(f"Sheets used: {len(self.solution.sheets)}\n")
            f.write(f"Total waste: {self.solution.get_total_waste():.4f} m2\n")
            f.write(f"Total used area: {self.solution.get_total_used_area():.4f} m2\n")

            total_capacity = len(self.solution.sheets) * self.problem.sheet_capacity
            utilization = (self.solution.get_total_used_area() / total_capacity) * 100
            f.write(f"Average utilization: {utilization:.2f}%\n")
            f.write(f"Makespan: {self.solution.get_makespan():.2f} time units\n")
            f.write(f"Avg product completion: {self.solution.get_avg_product_completion_time(self.problem):.2f} time units\n")
            f.write(f"Solution valid: {self.solution.is_valid(self.problem)}\n\n")

            f.write("-" * 40 + "\n")
            f.write("OUTPUT FILES\n")
            f.write("-" * 40 + "\n")
            f.write("- sheet_parts.csv: Detailed part assignments\n")
            f.write("- sheet_summary.csv: Sheet statistics\n")
            f.write("- product_summary.csv: Product statistics\n")
            f.write("- schedule.csv: Station scheduling\n")
            f.write("- sheets/: Individual sheet images\n")

        print(f"Solution summary saved to: {output_path}")

    def generate_all_outputs(self, sheet_width: float = 2.0, sheet_height: float = 1.8):
        """Generate all output files and images."""
        # Create output folder
        self.create_output_folder()

        # Export CSV files
        self.export_sheet_parts_csv()
        self.export_sheet_summary_csv()
        self.export_product_summary_csv()
        self.export_schedule_csv()
        self.export_solution_summary()

        # Generate sheet images
        self.generate_all_sheet_images(sheet_width, sheet_height)

        print(f"\nAll outputs generated in: {self.run_folder}")
        return self.run_folder


def generate_outputs(solution: Solution, problem: Problem, data_file: str,
                    sheet_width: float = 2.0, sheet_height: float = 1.8) -> Path:
    """
    Main function to generate all outputs for a solution.

    Args:
        solution: The solved solution
        problem: The problem instance
        data_file: Path to the input data file
        sheet_width: Sheet width in meters (from config)
        sheet_height: Sheet height in meters (from config)

    Returns:
        Path to the output folder
    """
    generator = OutputGenerator(solution, problem, data_file)
    return generator.generate_all_outputs(sheet_width, sheet_height)


if __name__ == "__main__":
    import json
    from models import Problem
    from solvers import GreedySolver
    from evaluation import WeightedEvaluator

    # Load config
    config_path = "config/config.json"
    with open(config_path, 'r') as f:
        config = json.load(f)

    data_file = config.get("data_file", "data/5693_cleaned.xlsx")
    sheet_x = config.get("sheet_X", 2.0)
    sheet_y = config.get("sheet_Y", 1.8)

    # Load evaluator parameters
    evaluator_config = config.get("evaluator", {})
    alpha = evaluator_config.get("alpha", 1.0)
    beta = evaluator_config.get("beta", 0.5)
    gamma = evaluator_config.get("gamma", 0.3)

    # Load problem and solve
    print("Loading problem...")
    problem = Problem.load_from_files(data_file, config_path)

    print("Solving...")
    solver = GreedySolver(sort_by='area_desc')
    evaluator = WeightedEvaluator(alpha=alpha, beta=beta, gamma=gamma)
    solution = solver.solve(problem, evaluator)

    # Generate outputs
    generate_outputs(solution, problem, data_file, sheet_x, sheet_y)
