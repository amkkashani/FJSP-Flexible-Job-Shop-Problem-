"""Greedy solver implementation for FJSP."""

from typing import Dict, List

from .base import Solver
from solution.solution import Solution
from solution.assignment import SheetAssignment
from models.problem import Problem
from models.sheet import Sheet
from models.part import Part
from evaluation.base import Evaluator


class GreedySolver(Solver):
    """
    Greedy solver using first-fit decreasing for bin packing
    and FIFO scheduling for station assignments.

    Algorithm:
    1. Sort parts by area (descending) - First Fit Decreasing
    2. Pack parts into sheets using first-fit
    3. Schedule sheets through stations in FIFO order
    """

    def __init__(self, sort_by: str = 'area_desc'):
        """
        Initialize the greedy solver.

        Args:
            sort_by: Sorting strategy for parts
                - 'area_desc': Sort by area descending (default, FFD)
                - 'area_asc': Sort by area ascending
                - 'product': Group by product, then by area descending
                - 'none': No sorting (original order)
        """
        self.sort_by = sort_by

    def solve(self, problem: Problem, evaluator: Evaluator) -> Solution:
        """
        Solve the FJSP using greedy approach.

        Args:
            problem: The problem instance
            evaluator: The evaluator (used for final evaluation)

        Returns:
            A complete solution
        """
        # Step 1: Sort parts
        sorted_parts = self._sort_parts(problem.parts)

        # Step 2: Pack parts into sheets (bin packing)
        sheets = self._pack_parts(sorted_parts, problem.sheet_capacity)

        # Step 3: Schedule sheets through stations
        solution = self._schedule_sheets(sheets, problem)

        # Compute final metrics
        solution.compute_metrics(problem)

        return solution

    def _sort_parts(self, parts: List[Part]) -> List[Part]:
        """Sort parts according to the configured strategy."""
        if self.sort_by == 'area_desc':
            return sorted(parts, key=lambda p: p.area, reverse=True)
        elif self.sort_by == 'area_asc':
            return sorted(parts, key=lambda p: p.area)
        elif self.sort_by == 'product':
            # Group by product, then sort by area descending within each group
            return sorted(parts, key=lambda p: (p.product_id, -p.area))
        else:  # 'none'
            return list(parts)

    def _pack_parts(self, parts: List[Part], sheet_capacity: float) -> List[Sheet]:
        """
        Pack parts into sheets using First Fit algorithm.

        Args:
            parts: Sorted list of parts
            sheet_capacity: Maximum capacity per sheet

        Returns:
            List of sheets with assigned parts
        """
        sheets: List[Sheet] = []
        sheet_counter = 0

        for part in parts:
            placed = False

            # Try to fit in existing sheets
            for sheet in sheets:
                if sheet.can_fit(part):
                    sheet.add_part(part)
                    placed = True
                    break

            # Create new sheet if needed
            if not placed:
                new_sheet = Sheet(
                    id=f"sheet_{sheet_counter:05d}",
                    capacity=sheet_capacity
                )
                new_sheet.add_part(part)
                sheets.append(new_sheet)
                sheet_counter += 1

        return sheets

    def _schedule_sheets(self, sheets: List[Sheet], problem: Problem) -> Solution:
        """
        Schedule sheets through stations using FIFO.

        Args:
            sheets: List of sheets to schedule
            problem: The problem instance

        Returns:
            Solution with complete schedule
        """
        solution = Solution()

        # Initialize machine availability for each station
        # machine_availability[station_name] = list of times when each machine becomes free
        machine_availability: Dict[str, List[float]] = {
            station.name: [0.0] * station.num_machines
            for station in problem.stations
        }

        # Add all sheets to solution
        for sheet in sheets:
            solution.add_sheet(sheet)

        # Schedule each sheet through all stations (FIFO order)
        for sheet in sheets:
            current_time = 0.0

            # Process through each station in order
            for station in sorted(problem.stations):
                process_time = sheet.get_station_time(station.name)

                # Skip station if process time is 0
                if process_time <= 0:
                    continue

                # Find earliest available machine
                earliest_machine = 0
                earliest_free = machine_availability[station.name][0]
                for i, free_time in enumerate(machine_availability[station.name]):
                    if free_time < earliest_free:
                        earliest_free = free_time
                        earliest_machine = i

                # Sheet can start when: (1) previous station done AND (2) machine is free
                start_time = max(current_time, earliest_free)
                end_time = start_time + process_time

                # Create assignment
                assignment = SheetAssignment(
                    station_name=station.name,
                    machine_index=earliest_machine,
                    start_time=start_time,
                    end_time=end_time
                )
                solution.add_assignment(sheet.id, assignment)

                # Update machine availability
                machine_availability[station.name][earliest_machine] = end_time

                # Update current time for next station
                current_time = end_time

        return solution

    def __repr__(self) -> str:
        return f"GreedySolver(sort_by={self.sort_by})"
