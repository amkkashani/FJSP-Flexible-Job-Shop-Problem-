"""Greedy solver implementation for FJSP."""

from typing import Dict, List, Tuple

from .base import Solver
from solution.solution import Solution
from solution.assignment import SheetAssignment
from solution.part_assignment import PartAssignment
from models.problem import Problem
from models.sheet import Sheet
from models.part import Part
from models.station import Station
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
        sheets = self._pack_parts(
            sorted_parts,
            problem.sheet_capacity,
            problem.sheet_width,
            problem.sheet_height
        )

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

    def _pack_parts(
        self,
        parts: List[Part],
        sheet_capacity: float,
        sheet_width: float,
        sheet_height: float
    ) -> List[Sheet]:
        """
        Pack parts into sheets using First Fit algorithm.

        Args:
            parts: Sorted list of parts
            sheet_capacity: Maximum capacity per sheet
            sheet_width: Sheet width in meters
            sheet_height: Sheet height in meters

        Returns:
            List of sheets with assigned parts
        """
        sheets: List[Sheet] = []
        sheet_counter = 0

        for part in parts:
            placed = False

            # Try to fit in existing sheets
            for sheet in sheets:
                if sheet.add_part(part):
                    placed = True
                    break

            # Create new sheet if needed
            if not placed:
                new_sheet = Sheet(
                    id=f"sheet_{sheet_counter:05d}",
                    capacity=sheet_capacity,
                    width=sheet_width,
                    height=sheet_height
                )
                if not new_sheet.add_part(part):
                    raise ValueError(
                        f"Part {part.id} ({part.length}x{part.width}mm) "
                        f"cannot fit in sheet {sheet_width}x{sheet_height}m."
                    )
                sheets.append(new_sheet)
                sheet_counter += 1

        return sheets

    def _schedule_sheets(self, sheets: List[Sheet], problem: Problem) -> Solution:
        """
        Schedule sheets through stations using FIFO.

        For stations with sheet=True, entire sheets are scheduled.
        For stations with sheet=False, individual parts are scheduled separately.

        Args:
            sheets: List of sheets to schedule
            problem: The problem instance

        Returns:
            Solution with complete schedule
        """
        solution = Solution()

        # Separate stations into sheet-based and part-based
        sorted_stations = sorted(problem.stations)
        sheet_stations = [s for s in sorted_stations if s.sheet]
        part_stations = [s for s in sorted_stations if not s.sheet]
        station_by_name = {s.name: s for s in sorted_stations}
        sheet_station_names = [s.name for s in sheet_stations]
        part_station_names = [s.name for s in part_stations]
        sheet_station_set = set(sheet_station_names)
        part_station_set = set(part_station_names)

        def get_sheet_sequence(sheet: Sheet) -> List[Station]:
            sequences = [p.sequence for p in sheet.assigned_parts if p.sequence]
            if sequences:
                ordered: List[str] = []
                seen = set()
                for seq in sequences:
                    for name in seq:
                        if name in sheet_station_set and name not in seen:
                            ordered.append(name)
                            seen.add(name)
                return [station_by_name[name] for name in ordered]
            return [station_by_name[name] for name in sheet_station_names]

        def get_part_sequence(part: Part) -> List[Station]:
            if part.sequence:
                ordered = [name for name in part.sequence if name in part_station_set]
                return [station_by_name[name] for name in ordered]
            return [station_by_name[name] for name in part_station_names]

        # Initialize machine availability for each station
        machine_availability: Dict[str, List[float]] = {
            station.name: [0.0] * station.num_machines
            for station in problem.stations
        }

        # Add all sheets to solution
        for sheet in sheets:
            solution.add_sheet(sheet)

        # Track when each part becomes available (after all sheet stations complete)
        # part_available_time[part_id] = time when part is released from sheet processing
        part_available_time: Dict[str, float] = {}

        # Phase 1: Schedule sheets through sheet-based stations
        for sheet in sheets:
            current_time = 0.0

            for station in get_sheet_sequence(sheet):
                workers = max(1, station.workers_per_machine)
                process_time = sheet.get_station_time(station.name) / workers

                # Skip station if process time is 0
                if process_time <= 0:
                    continue

                # Find earliest available machine
                earliest_machine, earliest_free = self._find_earliest_machine(
                    machine_availability[station.name]
                )

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

            # After all sheet stations, record when each part becomes available
            for part in sheet.assigned_parts:
                part_available_time[part.id] = current_time

        # Phase 2: Schedule individual parts through part-based stations
        if part_stations:
            # Collect all parts with their availability times
            all_parts_with_times: List[Tuple[Part, float, Sheet]] = []
            for sheet in sheets:
                for part in sheet.assigned_parts:
                    all_parts_with_times.append((part, part_available_time[part.id], sheet))

            # Sort parts by availability time (FIFO based on when they become free)
            all_parts_with_times.sort(key=lambda x: x[1])

            # Track current time for each part through part stations
            part_current_time: Dict[str, float] = {
                part.id: avail_time for part, avail_time, _ in all_parts_with_times
            }

            # Schedule each part through part-based stations
            for part, avail_time, sheet in all_parts_with_times:
                current_time = part_current_time[part.id]

                for station in get_part_sequence(part):
                    workers = max(1, station.workers_per_machine)
                    process_time = part.get_process_time(station.name) / workers

                    # Skip station if process time is 0
                    if process_time <= 0:
                        continue

                    # Find earliest available machine
                    earliest_machine, earliest_free = self._find_earliest_machine(
                        machine_availability[station.name]
                    )

                    # Part can start when: (1) previous station done AND (2) machine is free
                    start_time = max(current_time, earliest_free)
                    end_time = start_time + process_time

                    # Create part assignment
                    assignment = PartAssignment(
                        part_id=part.id,
                        station_name=station.name,
                        machine_index=earliest_machine,
                        start_time=start_time,
                        end_time=end_time
                    )
                    solution.add_part_assignment(part.id, assignment)

                    # Update machine availability
                    machine_availability[station.name][earliest_machine] = end_time

                    # Update current time for next station
                    current_time = end_time

                # Update part's current time
                part_current_time[part.id] = current_time

        return solution

    def _find_earliest_machine(self, machine_times: List[float]) -> Tuple[int, float]:
        """Find the machine with earliest availability."""
        earliest_machine = 0
        earliest_free = machine_times[0]
        for i, free_time in enumerate(machine_times):
            if free_time < earliest_free:
                earliest_free = free_time
                earliest_machine = i
        return earliest_machine, earliest_free

    def __repr__(self) -> str:
        return f"GreedySolver(sort_by={self.sort_by})"
