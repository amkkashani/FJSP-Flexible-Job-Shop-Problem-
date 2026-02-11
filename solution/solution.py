"""Solution model for FJSP."""

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, TYPE_CHECKING

from models.sheet import Sheet
from models.problem import Problem
from .assignment import SheetAssignment
from .part_assignment import PartAssignment

if TYPE_CHECKING:
    from models.remaining import RemainingSection


@dataclass
class Solution:
    """
    Represents a complete solution: which parts go to which sheets,
    and the schedule of sheets through stations.

    Attributes:
        sheets: All sheets with their assigned parts
        schedule: For each sheet_id: list of assignments at each station (for sheet-based stations)
        part_schedule: For each part_id: list of assignments at each station (for part-based stations)
        metrics: Computed metrics (populated after evaluation)
        remaining_sections: Remaining sections to save for future runs
        used_remaining_sections: Remaining sections that were used from previous runs
    """
    sheets: List[Sheet] = field(default_factory=list)
    schedule: Dict[str, List[SheetAssignment]] = field(default_factory=dict)
    part_schedule: Dict[str, List[PartAssignment]] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    remaining_sections: List['RemainingSection'] = field(default_factory=list)
    used_remaining_sections: List['RemainingSection'] = field(default_factory=list)

    def get_makespan(self) -> float:
        """Time when last sheet/part finishes last station."""
        max_end_time = 0.0

        # Check sheet-level schedules
        for sheet_id, assignments in self.schedule.items():
            if assignments:
                sheet_end = max(a.end_time for a in assignments)
                max_end_time = max(max_end_time, sheet_end)

        # Check part-level schedules
        for part_id, assignments in self.part_schedule.items():
            if assignments:
                part_end = max(a.end_time for a in assignments)
                max_end_time = max(max_end_time, part_end)

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

        # Check part-level schedules first (for parts that went through part stations)
        for part_id in part_ids:
            if part_id in self.part_schedule and self.part_schedule[part_id]:
                part_completion = max(a.end_time for a in self.part_schedule[part_id])
                max_completion = max(max_completion, part_completion)

        # Also check sheet-level schedules (for parts that only went through sheet stations)
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
        2. Sheet size/placement constraint (within sheet bounds, no overlaps)
        3. Part assignment constraint (each part assigned exactly once)
        4. Station order constraint
        5. Non-preemption (implicit in assignment structure)
        """
        # 1. Check sheet capacity constraint
        for sheet in self.sheets:
            if sheet.total_area() > sheet.capacity + 1e-9:  # Small tolerance for floating point
                return False
            materials = {part.material for part in sheet.assigned_parts}
            if len(materials) > 1:
                return False
            if sheet.material is not None and materials and sheet.material not in materials:
                return False

            if sheet.placements:
                if len(sheet.placements) != len(sheet.assigned_parts):
                    return False
                rects = []
                for part in sheet.assigned_parts:
                    placement = sheet.placements.get(part.id)
                    if placement is None:
                        return False
                    x, y, w, h, _ = placement
                    if x < -1e-9 or y < -1e-9:
                        return False
                    if x + w > sheet.width + 1e-9 or y + h > sheet.height + 1e-9:
                        return False
                    rects.append((x, y, w, h))

                for i in range(len(rects)):
                    ax, ay, aw, ah = rects[i]
                    for j in range(i + 1, len(rects)):
                        bx, by, bw, bh = rects[j]
                        if not (ax + aw <= bx + 1e-9 or bx + bw <= ax + 1e-9 or
                                ay + ah <= by + 1e-9 or by + bh <= ay + 1e-9):
                            return False
            else:
                for part in sheet.assigned_parts:
                    w = part.width / 1000.0
                    h = part.length / 1000.0
                    if not ((w <= sheet.width and h <= sheet.height) or (h <= sheet.width and w <= sheet.height)):
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

        # 3. Check station order constraint (per-sheet sequence-aware)
        sheet_station_names = [s.name for s in problem.stations if s.sheet]
        default_sheet_order = [s.name for s in sorted(problem.stations) if s.sheet]
        sheet_station_set = set(sheet_station_names)

        def get_sheet_order_map(sheet: Sheet) -> Dict[str, int]:
            sequences = [p.sequence for p in sheet.assigned_parts if p.sequence]
            if sequences:
                ordered: List[str] = []
                seen = set()
                for seq in sequences:
                    for name in seq:
                        if name in sheet_station_set and name not in seen:
                            ordered.append(name)
                            seen.add(name)
                return {name: idx for idx, name in enumerate(ordered)}
            ordered = list(default_sheet_order)
            return {name: idx for idx, name in enumerate(ordered)}

        for sheet_id, assignments in self.schedule.items():
            prev_end_time = 0.0
            prev_order = -1
            sheet = self.get_sheet_by_id(sheet_id)
            order_map = get_sheet_order_map(sheet) if sheet else {n: i for i, n in enumerate(default_sheet_order)}

            for assignment in sorted(assignments, key=lambda a: a.start_time):
                order = order_map.get(assignment.station_name)
                if order is None:
                    return False
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

    def add_part_assignment(self, part_id: str, assignment: PartAssignment) -> None:
        """Add a station assignment for a part."""
        if part_id not in self.part_schedule:
            self.part_schedule[part_id] = []
        self.part_schedule[part_id].append(assignment)

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

    def save_remaining_sections(self, file_path: str) -> None:
        """
        Save remaining sections to a JSON file for use in future runs.

        Args:
            file_path: Path to the remaining.json file
        """
        data = {
            "_comment": "Remaining sheet sections from previous runs. These can be reused in future runs before creating new sheets.",
            "sections": [section.to_dict() for section in self.remaining_sections]
        }
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)

    def get_remaining_summary(self) -> str:
        """Get a summary of remaining sections."""
        lines = [
            f"Used {len(self.used_remaining_sections)} remaining sections from previous runs",
            f"Generated {len(self.remaining_sections)} new remaining sections for future runs"
        ]
        if self.remaining_sections:
            total_area = sum(s.area for s in self.remaining_sections)
            lines.append(f"Total remaining area: {total_area:.4f} m²")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"Solution(sheets={self.num_sheets()}, makespan={self.get_makespan():.2f})"
