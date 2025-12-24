"""Solution model for FJSP."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from models.sheet import Sheet
from models.problem import Problem
from .assignment import SheetAssignment


@dataclass
class Solution:
    """
    Represents a complete solution: which parts go to which sheets,
    and the schedule of sheets through stations.

    Attributes:
        sheets: All sheets with their assigned parts
        schedule: For each sheet_id: list of assignments at each station
        metrics: Computed metrics (populated after evaluation)
    """
    sheets: List[Sheet] = field(default_factory=list)
    schedule: Dict[str, List[SheetAssignment]] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)

    def get_makespan(self) -> float:
        """Time when last sheet finishes last station."""
        if not self.schedule:
            return 0.0

        max_end_time = 0.0
        for sheet_id, assignments in self.schedule.items():
            if assignments:
                sheet_end = max(a.end_time for a in assignments)
                max_end_time = max(max_end_time, sheet_end)

        return max_end_time

    def get_total_waste(self) -> float:
        """Sum of waste across all sheets."""
        return sum(sheet.waste() for sheet in self.sheets)

    def get_total_used_area(self) -> float:
        """Sum of used area across all sheets."""
        return sum(sheet.total_area() for sheet in self.sheets)

    def get_product_completion_time(self, product_id: str, problem: Problem) -> float:
        """
        When all parts of a product finish processing.

        Args:
            product_id: The product ID to check
            problem: The problem instance (to get product-part mapping)

        Returns:
            The time when the last part of the product completes
        """
        if product_id not in problem.products:
            return 0.0

        product = problem.products[product_id]
        part_ids = set(product.part_ids)

        max_completion = 0.0

        # Find all sheets containing parts from this product
        for sheet in self.sheets:
            sheet_part_ids = set(sheet.get_part_ids())
            if sheet_part_ids & part_ids:  # If intersection exists
                # Get the completion time for this sheet
                if sheet.id in self.schedule and self.schedule[sheet.id]:
                    sheet_completion = max(a.end_time for a in self.schedule[sheet.id])
                    max_completion = max(max_completion, sheet_completion)

        return max_completion

    def get_all_product_completion_times(self, problem: Problem) -> Dict[str, float]:
        """Get completion times for all products."""
        return {
            product_id: self.get_product_completion_time(product_id, problem)
            for product_id in problem.products.keys()
        }

    def get_avg_product_completion_time(self, problem: Problem) -> float:
        """Average completion time across all products."""
        times = self.get_all_product_completion_times(problem)
        if not times:
            return 0.0
        return sum(times.values()) / len(times)

    def num_sheets(self) -> int:
        """Total number of sheets used."""
        return len(self.sheets)

    def is_valid(self, problem: Problem) -> bool:
        """
        Check if solution satisfies all constraints.

        Validates:
        1. Sheet capacity constraint
        2. Part assignment constraint (each part assigned exactly once)
        3. Station order constraint
        4. Non-preemption (implicit in assignment structure)
        """
        # 1. Check sheet capacity constraint
        for sheet in self.sheets:
            if sheet.total_area() > sheet.capacity + 1e-9:  # Small tolerance for floating point
                return False

        # 2. Check part assignment constraint
        assigned_parts: Set[str] = set()
        for sheet in self.sheets:
            for part in sheet.assigned_parts:
                if part.id in assigned_parts:
                    return False  # Part assigned twice
                assigned_parts.add(part.id)

        # Check all parts are assigned
        all_part_ids = {part.id for part in problem.parts}
        if assigned_parts != all_part_ids:
            return False

        # 3. Check station order constraint
        station_order = {s.name: s.order_index for s in problem.stations}
        for sheet_id, assignments in self.schedule.items():
            prev_end_time = 0.0
            prev_order = -1
            for assignment in sorted(assignments, key=lambda a: a.start_time):
                order = station_order.get(assignment.station_name, 0)
                # Station order must be non-decreasing
                if order < prev_order:
                    return False
                # Start time must be >= previous end time
                if assignment.start_time < prev_end_time - 1e-9:
                    return False
                prev_end_time = assignment.end_time
                prev_order = order

        return True

    def get_sheet_by_id(self, sheet_id: str) -> Optional[Sheet]:
        """Get a sheet by its ID."""
        for sheet in self.sheets:
            if sheet.id == sheet_id:
                return sheet
        return None

    def add_sheet(self, sheet: Sheet) -> None:
        """Add a sheet to the solution."""
        self.sheets.append(sheet)
        self.schedule[sheet.id] = []

    def add_assignment(self, sheet_id: str, assignment: SheetAssignment) -> None:
        """Add a station assignment for a sheet."""
        if sheet_id not in self.schedule:
            self.schedule[sheet_id] = []
        self.schedule[sheet_id].append(assignment)

    def compute_metrics(self, problem: Problem) -> Dict[str, float]:
        """Compute and store all metrics."""
        self.metrics = {
            'total_waste': self.get_total_waste(),
            'makespan': self.get_makespan(),
            'num_sheets': self.num_sheets(),
            'total_used_area': self.get_total_used_area(),
            'avg_product_completion': self.get_avg_product_completion_time(problem),
            'is_valid': float(self.is_valid(problem))
        }
        return self.metrics

    def summary(self, problem: Problem) -> str:
        """Generate a summary string of the solution."""
        self.compute_metrics(problem)
        lines = [
            "=" * 50,
            "SOLUTION SUMMARY",
            "=" * 50,
            f"Sheets used: {self.metrics['num_sheets']}",
            f"Total waste: {self.metrics['total_waste']:.4f} m²",
            f"Total used area: {self.metrics['total_used_area']:.4f} m²",
            f"Makespan: {self.metrics['makespan']:.2f} time units",
            f"Avg product completion: {self.metrics['avg_product_completion']:.2f} time units",
            f"Valid solution: {bool(self.metrics['is_valid'])}",
            "=" * 50
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"Solution(sheets={self.num_sheets()}, makespan={self.get_makespan():.2f})"
