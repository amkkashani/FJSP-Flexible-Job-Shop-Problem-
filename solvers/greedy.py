"""Greedy solver implementation for FJSP."""

from datetime import datetime
from typing import Dict, List, Optional, Tuple

from .base import Solver
from solution.solution import Solution
from solution.assignment import SheetAssignment
from solution.part_assignment import PartAssignment
from models.problem import Problem
from models.sheet import Sheet
from models.part import Part
from models.station import Station
from models.remaining import RemainingSection
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

    def __init__(
        self,
        sort_by: str = 'area_desc',
        remaining_min_width: float = 0.1,
        remaining_min_height: float = 0.1,
        remaining_min_area: float = 0.01
    ):
        """
        Initialize the greedy solver.

        Args:
            sort_by: Sorting strategy for parts
                - 'area_desc': Sort by area descending (default, FFD)
                - 'area_asc': Sort by area ascending
                - 'product': Group by product, then by area descending
                - 'none': No sorting (original order)
            remaining_min_width: Minimum width (m) for remaining sections
            remaining_min_height: Minimum height (m) for remaining sections
            remaining_min_area: Minimum area (m²) for remaining sections
        """
        self.sort_by = sort_by
        self.remaining_min_width = remaining_min_width
        self.remaining_min_height = remaining_min_height
        self.remaining_min_area = remaining_min_area

    def solve(
        self,
        problem: Problem,
        evaluator: Evaluator,
        remaining_sections: Optional[List[RemainingSection]] = None
    ) -> Solution:
        """
        Solve the FJSP using greedy approach.

        Args:
            problem: The problem instance
            evaluator: The evaluator (used for final evaluation)
            remaining_sections: Optional list of remaining sections from previous runs

        Returns:
            A complete solution
        """
        # Step 1: Sort parts
        sorted_parts = self._sort_parts(problem.parts)

        # Step 2: Pack parts into sheets (bin packing) with material-specific sizes
        # Also handles remaining sections from previous runs
        sheets, used_remaining, new_remaining = self._pack_parts(
            sorted_parts,
            problem,
            remaining_sections or []
        )

        # Step 3: Schedule sheets through stations
        solution = self._schedule_sheets(sheets, problem)

        # Step 4: Store remaining sections in solution for later saving
        solution.remaining_sections = new_remaining
        solution.used_remaining_sections = used_remaining

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
        problem: Problem,
        remaining_sections: List[RemainingSection]
    ) -> Tuple[List[Sheet], List[RemainingSection], List[RemainingSection]]:
        """
        Pack parts into sheets using First Fit algorithm.
        Uses material-specific sheet sizes from problem config.
        Tries to use remaining sections from previous runs first.

        Args:
            parts: Sorted list of parts
            problem: Problem instance with sheet size configuration
            remaining_sections: List of remaining sections from previous runs

        Returns:
            Tuple of (sheets, used_remaining_sections, new_remaining_sections)
        """
        sheets: List[Sheet] = []
        sheet_counter = 0

        # Track which remaining sections are still available
        available_remaining = list(remaining_sections)
        used_remaining: List[RemainingSection] = []

        for part in parts:
            placed = False

            # Try to fit in existing sheets
            for sheet in sheets:
                if sheet.add_part(part):
                    placed = True
                    break

            # Create new sheet if needed
            if not placed:
                # First, try to use a remaining section
                remaining_sheet = None
                part_width_m = part.width / 1000.0
                part_height_m = part.length / 1000.0

                for i, remaining in enumerate(available_remaining):
                    if remaining.can_fit_part(part_width_m, part_height_m, part.material):
                        # Create a sheet from this remaining section
                        remaining_sheet = Sheet(
                            id=f"sheet_{sheet_counter:05d}_from_{remaining.id}",
                            capacity=remaining.area,
                            width=remaining.width,
                            height=remaining.height,
                            material=remaining.material
                        )
                        if remaining_sheet.add_part(part):
                            sheets.append(remaining_sheet)
                            sheet_counter += 1
                            used_remaining.append(remaining)
                            available_remaining.pop(i)
                            placed = True
                            break

                # If no remaining section worked, create a new sheet
                if not placed:
                    sheet_width, sheet_height, sheet_capacity = problem.get_sheet_size_for_material(
                        part.material
                    )
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

        # Calculate new remaining sections from sheets
        new_remaining = self._calculate_remaining_sections(
            sheets,
            min_width=self.remaining_min_width,
            min_height=self.remaining_min_height,
            min_area=self.remaining_min_area
        )

        # Add unused remaining sections back to new_remaining
        new_remaining.extend(available_remaining)

        return sheets, used_remaining, new_remaining

    def _calculate_remaining_sections(
        self,
        sheets: List[Sheet],
        min_width: float = 0.1,
        min_height: float = 0.1,
        min_area: float = 0.01
    ) -> List[RemainingSection]:
        """
        Calculate remaining sections from the waste of each sheet.
        Captures multiple remaining areas per sheet:
        1. Bottom remaining (below all shelves)
        2. Right-side remaining for each shelf

        Args:
            sheets: List of sheets after packing
            min_width: Minimum width threshold (meters)
            min_height: Minimum height threshold (meters)
            min_area: Minimum area threshold (m²)

        Returns:
            List of remaining sections
        """
        remaining_sections: List[RemainingSection] = []
        timestamp = datetime.now().isoformat()
        section_counter = 0

        for sheet in sheets:
            if not sheet._shelves:
                continue

            material = sheet.get_material()

            # 1. Calculate right-side remaining for each shelf
            for shelf_idx, shelf in enumerate(sheet._shelves):
                shelf_remaining_width = sheet.width - shelf["x"]
                shelf_height = shelf["height"]

                if shelf_remaining_width >= min_width and shelf_height >= min_height:
                    shelf_area = shelf_remaining_width * shelf_height
                    if shelf_area >= min_area:
                        section = RemainingSection(
                            id=f"rem_{sheet.id}_shelf{shelf_idx}_{section_counter:03d}",
                            material=material,
                            width=shelf_remaining_width,
                            height=shelf_height,
                            area=shelf_area,
                            original_sheet_id=sheet.id,
                            created_at=timestamp
                        )
                        remaining_sections.append(section)
                        section_counter += 1

            # 2. Calculate the bottom remaining section (below all shelves)
            last_shelf = sheet._shelves[-1]
            used_height = last_shelf["y"] + last_shelf["height"]
            remaining_height = sheet.height - used_height

            if remaining_height >= min_height and sheet.width >= min_width:
                remaining_area = sheet.width * remaining_height
                if remaining_area >= min_area:
                    section = RemainingSection(
                        id=f"rem_{sheet.id}_bottom_{section_counter:03d}",
                        material=material,
                        width=sheet.width,
                        height=remaining_height,
                        area=remaining_area,
                        original_sheet_id=sheet.id,
                        created_at=timestamp
                    )
                    remaining_sections.append(section)
                    section_counter += 1

        return remaining_sections

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
