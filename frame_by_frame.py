"""Frame-by-frame visualization for FJSP solutions.

Creates individual frames at each start/end event for sheets and parts,
showing the flow from sheet processing through part separation and individual processing.
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple, Set
import json

sys.path.insert(0, str(Path(__file__).parent))

from models import Problem
from models.part import Part
from solution import Solution
from solution.part_assignment import PartAssignment
from solution.assignment import SheetAssignment

# Try to import matplotlib
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np
    MATPLOTLIB_AVAILABLE = True
except Exception as exc:
    MATPLOTLIB_AVAILABLE = False
    print(f"Warning: matplotlib not available: {exc}")


class FrameByFrameGenerator:
    """Generates frame-by-frame visualization at each event time."""

    def __init__(self, solution: Solution, problem: Problem):
        self.solution = solution
        self.problem = problem
        self.makespan = solution.get_makespan()

        # Build station info
        self.stations = sorted(problem.stations)
        self.station_machines = {s.name: s.num_machines for s in self.stations}
        self.station_is_sheet = {s.name: s.sheet for s in self.stations}

        # Build part lookup
        self.all_parts: Dict[str, Part] = {}
        self.part_to_sheet: Dict[str, str] = {}  # part_id -> sheet_id
        for sheet in solution.sheets:
            for part in sheet.assigned_parts:
                self.all_parts[part.id] = part
                self.part_to_sheet[part.id] = sheet.id

    def collect_event_times(self) -> List[Tuple[float, str]]:
        """Collect all event times (start/end of sheet and part processing)."""
        events: Set[Tuple[float, str]] = set()

        # Add time 0
        events.add((0.0, "start"))

        # Sheet events
        for sheet_id, assignments in self.solution.schedule.items():
            for a in assignments:
                events.add((a.start_time, f"sheet_{sheet_id}_start_{a.station_name}"))
                events.add((a.end_time, f"sheet_{sheet_id}_end_{a.station_name}"))

        # Part events
        for part_id, assignments in self.solution.part_schedule.items():
            for a in assignments:
                events.add((a.start_time, f"part_{part_id}_start_{a.station_name}"))
                events.add((a.end_time, f"part_{part_id}_end_{a.station_name}"))

        # Sort by time
        sorted_events = sorted(events, key=lambda x: x[0])
        return sorted_events

    def collect_part_station_events(self) -> List[Tuple[float, int, str, str, str]]:
        """
        Collect part-centric events: start and end at each station.

        Returns list of (time, sequence_order, event_type, part_id, station_name)
        where sequence_order is used to order events at the same time:
        - end events come before start events at same time (to show transition)

        This creates exactly 2 frames per part per station: start frame and end frame.
        When end of one station equals start of next, we create two sequential frames.
        """
        events: List[Tuple[float, int, str, str, str]] = []

        # For each part, collect its journey through all stations
        for part_id, part in self.all_parts.items():
            sheet_id = self.part_to_sheet.get(part_id)

            # Sheet station events (part travels with sheet)
            if sheet_id and sheet_id in self.solution.schedule:
                for a in self.solution.schedule[sheet_id]:
                    # sequence_order: 0 for end (comes first), 1 for start (comes second)
                    # This way at same time: previous station end shows before next station start
                    events.append((a.start_time, 1, "start", part_id, a.station_name))
                    events.append((a.end_time, 0, "end", part_id, a.station_name))

            # Part station events (part travels individually after separation)
            if part_id in self.solution.part_schedule:
                for a in self.solution.part_schedule[part_id]:
                    events.append((a.start_time, 1, "start", part_id, a.station_name))
                    events.append((a.end_time, 0, "end", part_id, a.station_name))

        # Sort by: time, then sequence_order (end=0 before start=1), then part_id, then station
        events.sort(key=lambda x: (x[0], x[1], x[3], x[4]))

        return events

    def generate_part_frames(self, output_dir: str = "output/frame_by_frame",
                             max_frames: int = None):
        """
        Generate frames showing start/end states for each part at each station.

        Creates exactly 2 frames per part per station:
        - Frame when part STARTS at station (just arrived)
        - Frame when part ENDS at station (about to leave)

        When end time of one station equals start time of next station,
        two sequential frames are created at the same time value.

        Args:
            output_dir: Directory to save frames
            max_frames: Maximum number of frames to generate (None = all)
        """
        if not MATPLOTLIB_AVAILABLE:
            print("Cannot generate frames - matplotlib not available")
            return

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Collect part-centric events
        events = self.collect_part_station_events()
        total_events = len(events)

        if max_frames and len(events) > max_frames:
            # Sample events but try to keep start/end pairs together
            step = max(1, len(events) // max_frames)
            events = events[::step][:max_frames]

        print(f"\nGenerating {len(events)} part-station frames (from {total_events} total events)...")
        print(f"Output directory: {output_dir}")

        for frame_idx, (time, seq, event_type, part_id, station_name) in enumerate(events):
            fig, ax = plt.subplots(figsize=(18, 10))

            # Create descriptive event string
            event_desc = f"{part_id} {event_type.upper()} at {station_name}"

            self._draw_part_frame(ax, time, frame_idx, event_desc, part_id, station_name, event_type)

            frame_path = output_path / f"frame_{frame_idx:05d}_t{time:.1f}_{event_type}.png"
            plt.savefig(frame_path, dpi=120, bbox_inches='tight')
            plt.close()

            if frame_idx % 50 == 0 or frame_idx == len(events) - 1:
                print(f"  Generated frame {frame_idx + 1}/{len(events)} (t={time:.1f}, {event_type})")

        print(f"\nPart frames saved to: {output_dir}")

    def _draw_part_frame(self, ax, current_time: float, frame_idx: int, event_desc: str,
                         highlight_part: str = None, highlight_station: str = None,
                         event_type: str = None):
        """Draw a single frame with optional highlighting of the current event's part/station."""

        # Layout parameters
        num_stations = len(self.stations)
        station_width = 1.8
        machine_height = 0.8
        station_spacing = 0.6
        machine_spacing = 0.25
        waiting_area_height = 2.0
        info_panel_width = 3.0

        max_machines = max(self.station_machines.values())
        machine_area_height = max_machines * (machine_height + machine_spacing)
        total_height = waiting_area_height + machine_area_height
        layout_width = num_stations * (station_width + station_spacing)

        # For start events, we show the part AT the station (just arrived)
        # For end events, we show the part AT the station (about to leave)
        # We use slightly different time interpretations:
        # - start: show state at exactly start_time (part is processing)
        # - end: show state at end_time - epsilon (part still processing, about to finish)

        effective_time = current_time
        if event_type == "end":
            # For end frame, we want to show the part still at the station
            effective_time = current_time - 0.001  # tiny epsilon before end

        # Collect current state at effective_time
        # Active sheets
        active_sheets: Dict[str, Tuple[str, int]] = {}
        for sheet_id, assignments in self.solution.schedule.items():
            for a in assignments:
                if a.start_time <= effective_time < a.end_time:
                    active_sheets[sheet_id] = (a.station_name, a.machine_index)

        # Active parts
        active_parts: Dict[str, Tuple[str, int]] = {}
        for part_id, assignments in self.solution.part_schedule.items():
            for a in assignments:
                if a.start_time <= effective_time < a.end_time:
                    active_parts[part_id] = (a.station_name, a.machine_index)

        # For the highlighted part, ensure it shows at the highlighted station
        if highlight_part and highlight_station:
            # Check if this part is on a sheet at this station
            sheet_id = self.part_to_sheet.get(highlight_part)
            if sheet_id and sheet_id in self.solution.schedule:
                for a in self.solution.schedule[sheet_id]:
                    if a.station_name == highlight_station:
                        if event_type == "start" and a.start_time == current_time:
                            active_sheets[sheet_id] = (a.station_name, a.machine_index)
                        elif event_type == "end" and a.end_time == current_time:
                            active_sheets[sheet_id] = (a.station_name, a.machine_index)

            # Check if this part has individual assignment at this station
            if highlight_part in self.solution.part_schedule:
                for a in self.solution.part_schedule[highlight_part]:
                    if a.station_name == highlight_station:
                        if event_type == "start" and a.start_time == current_time:
                            active_parts[highlight_part] = (a.station_name, a.machine_index)
                        elif event_type == "end" and a.end_time == current_time:
                            active_parts[highlight_part] = (a.station_name, a.machine_index)

        # Waiting sheets at each station
        waiting_sheets: Dict[str, List[str]] = {s.name: [] for s in self.stations}
        for sheet_id, assignments in self.solution.schedule.items():
            sorted_assignments = sorted(assignments, key=lambda x: x.start_time)
            for idx, a in enumerate(sorted_assignments):
                if a.start_time > effective_time:
                    prev_end = sorted_assignments[idx - 1].end_time if idx > 0 else 0.0
                    if prev_end <= effective_time:
                        waiting_sheets[a.station_name].append(sheet_id)
                    break

        # Waiting parts at each station
        waiting_parts: Dict[str, List[str]] = {s.name: [] for s in self.stations}
        for part_id, assignments in self.solution.part_schedule.items():
            if not assignments:
                continue

            sheet_id = self.part_to_sheet.get(part_id)
            sheet_end_time = 0.0
            if sheet_id and sheet_id in self.solution.schedule:
                sheet_assignments = self.solution.schedule[sheet_id]
                if sheet_assignments:
                    sheet_end_time = max(a.end_time for a in sheet_assignments)

            if effective_time < sheet_end_time:
                continue

            sorted_assignments = sorted(assignments, key=lambda x: x.start_time)
            for idx, a in enumerate(sorted_assignments):
                if a.start_time > effective_time:
                    prev_end = sorted_assignments[idx - 1].end_time if idx > 0 else sheet_end_time
                    if prev_end <= effective_time:
                        waiting_parts[a.station_name].append(part_id)
                    break

        # Draw stations
        station_positions = {}
        for i, station in enumerate(self.stations):
            x = i * (station_width + station_spacing)
            num_machines = self.station_machines[station.name]
            is_sheet = self.station_is_sheet[station.name]

            y_offset = waiting_area_height + (
                machine_area_height - num_machines * (machine_height + machine_spacing)
            ) / 2

            # Highlight the current station
            is_highlighted_station = (station.name == highlight_station)

            # Station label
            station_type = "SHEET" if is_sheet else "PART"
            label_color = 'darkblue' if is_sheet else 'darkgreen'
            if is_highlighted_station:
                label_color = 'red'
            ax.text(x + station_width/2, total_height + 0.4,
                   f"{station.name}\n[{station_type}]\n({num_machines} M)",
                   ha='center', va='bottom', fontsize=9, fontweight='bold', color=label_color)

            # Draw machines
            for m in range(num_machines):
                y = y_offset + m * (machine_height + machine_spacing)
                box_color = 'lightblue' if is_sheet else 'lightyellow'
                edge_color = 'black'
                edge_width = 2

                # Highlight station border
                if is_highlighted_station:
                    edge_color = 'red'
                    edge_width = 4

                rect = mpatches.Rectangle((x, y), station_width, machine_height,
                                         linewidth=edge_width, edgecolor=edge_color,
                                         facecolor=box_color, alpha=0.4)
                ax.add_patch(rect)
                ax.text(x + 0.1, y + machine_height - 0.1, f"M{m}",
                       va='top', fontsize=8, color='gray')

            # Waiting area separator
            separator_y = waiting_area_height - 0.15
            ax.plot([x, x + station_width], [separator_y, separator_y],
                   linestyle='--', color='gray', linewidth=1)

            # Waiting list
            if is_sheet:
                wait_list = waiting_sheets.get(station.name, [])
                wait_label = f"Waiting Sheets: {len(wait_list)}"
                if wait_list:
                    display = "\n".join([s.replace('sheet_', 'S') for s in wait_list[:4]])
                    if len(wait_list) > 4:
                        display += f"\n+{len(wait_list) - 4} more"
                else:
                    display = "None"
                ax.text(x + station_width/2, waiting_area_height/2,
                       f"{wait_label}\n{display}",
                       ha='center', va='center', fontsize=7, color='darkblue')
            else:
                wait_list = waiting_parts.get(station.name, [])
                wait_label = f"Waiting Parts: {len(wait_list)}"
                if wait_list:
                    display = "\n".join([p.replace('row_', 'P')[:8] for p in wait_list[:4]])
                    if len(wait_list) > 4:
                        display += f"\n+{len(wait_list) - 4} more"
                else:
                    display = "None"
                ax.text(x + station_width/2, waiting_area_height/2,
                       f"{wait_label}\n{display}",
                       ha='center', va='center', fontsize=7, color='darkgreen')

            station_positions[station.name] = (x, y_offset)

        # Draw active sheets on machines
        for sheet_id, (station_name, machine_idx) in active_sheets.items():
            if station_name not in station_positions:
                continue
            x_base, y_base = station_positions[station_name]
            y = y_base + machine_idx * (machine_height + machine_spacing)

            # Get progress
            assignments = self.solution.schedule.get(sheet_id, [])
            current_assignment = None
            for a in assignments:
                if a.station_name == station_name:
                    # Check if this is the right assignment
                    if a.start_time <= effective_time < a.end_time:
                        current_assignment = a
                        break
                    # Also match for exact end time (for end events)
                    if event_type == "end" and a.end_time == current_time and a.station_name == highlight_station:
                        current_assignment = a
                        break

            progress = 0.5
            if current_assignment:
                if event_type == "start" and current_assignment.start_time == current_time:
                    progress = 0.0  # Just started
                elif event_type == "end" and current_assignment.end_time == current_time:
                    progress = 1.0  # Just finished
                else:
                    progress = (effective_time - current_assignment.start_time) / current_assignment.duration
                    progress = max(0, min(1, progress))

            # Check if this sheet contains the highlighted part
            sheet_obj = self.solution.get_sheet_by_id(sheet_id)
            contains_highlight = False
            if sheet_obj and highlight_part:
                for p in sheet_obj.assigned_parts:
                    if p.id == highlight_part:
                        contains_highlight = True
                        break

            # Draw sheet
            sw, sh = station_width * 0.85, machine_height * 0.7
            sheet_x = x_base + (station_width - sw) / 2
            sheet_y = y + (machine_height - sh) / 2

            # Color based on progress and highlight
            if contains_highlight and station_name == highlight_station:
                if event_type == "start":
                    color = 'lime'  # Bright green for start
                else:
                    color = 'orange'  # Orange for end
                edge_color = 'red'
                edge_width = 3
            else:
                color = plt.cm.RdYlGn(1 - progress)
                edge_color = 'darkblue'
                edge_width = 2

            sheet_rect = mpatches.Rectangle((sheet_x, sheet_y), sw, sh,
                                           linewidth=edge_width, edgecolor=edge_color,
                                           facecolor=color, alpha=0.9)
            ax.add_patch(sheet_rect)

            # Sheet label with part count
            num_parts = len(sheet_obj.assigned_parts) if sheet_obj else 0
            label = f"{sheet_id.replace('sheet_', 'S')}\n({num_parts} parts)"
            ax.text(sheet_x + sw/2, sheet_y + sh/2, label,
                   ha='center', va='center', fontsize=6, fontweight='bold', color='black')

        # Draw active parts on machines
        for part_id, (station_name, machine_idx) in active_parts.items():
            if station_name not in station_positions:
                continue
            x_base, y_base = station_positions[station_name]
            y = y_base + machine_idx * (machine_height + machine_spacing)

            # Get progress
            assignments = self.solution.part_schedule.get(part_id, [])
            current_assignment = None
            for a in assignments:
                if a.station_name == station_name:
                    if a.start_time <= effective_time < a.end_time:
                        current_assignment = a
                        break
                    if event_type == "end" and a.end_time == current_time and a.station_name == highlight_station:
                        current_assignment = a
                        break

            progress = 0.5
            if current_assignment:
                if event_type == "start" and current_assignment.start_time == current_time:
                    progress = 0.0
                elif event_type == "end" and current_assignment.end_time == current_time:
                    progress = 1.0
                else:
                    progress = (effective_time - current_assignment.start_time) / current_assignment.duration
                    progress = max(0, min(1, progress))

            # Draw part (smaller, rounded)
            pw, ph = station_width * 0.65, machine_height * 0.5
            part_x = x_base + (station_width - pw) / 2
            part_y = y + (machine_height - ph) / 2

            # Highlight the specific part
            is_highlighted = (part_id == highlight_part and station_name == highlight_station)
            if is_highlighted:
                if event_type == "start":
                    color = 'lime'  # Bright green for start
                else:
                    color = 'orange'  # Orange for end
                edge_color = 'red'
                edge_width = 3
            else:
                color = plt.cm.RdYlGn(1 - progress)
                edge_color = 'darkgreen'
                edge_width = 1.5

            part_rect = mpatches.FancyBboxPatch(
                (part_x, part_y), pw, ph,
                boxstyle="round,pad=0.02,rounding_size=0.15",
                linewidth=edge_width, edgecolor=edge_color,
                facecolor=color, alpha=0.9)
            ax.add_patch(part_rect)

            label = part_id.replace('row_', 'P')[:7]
            ax.text(part_x + pw/2, part_y + ph/2, label,
                   ha='center', va='center', fontsize=5, fontweight='bold', color='black')

        # Draw arrows between stations
        arrow_y = waiting_area_height + machine_area_height / 2
        for i in range(len(self.stations) - 1):
            x1 = i * (station_width + station_spacing) + station_width
            x2 = (i + 1) * (station_width + station_spacing)

            curr_is_sheet = self.station_is_sheet[self.stations[i].name]
            next_is_sheet = self.station_is_sheet[self.stations[i + 1].name]

            if curr_is_sheet and not next_is_sheet:
                ax.annotate('', xy=(x2, arrow_y), xytext=(x1, arrow_y),
                           arrowprops=dict(arrowstyle='->', lw=3, color='red', alpha=0.8))
                ax.text((x1 + x2) / 2, arrow_y + 0.2, "SPLIT",
                       ha='center', va='bottom', fontsize=8, color='red', fontweight='bold')
            else:
                ax.annotate('', xy=(x2, arrow_y), xytext=(x1, arrow_y),
                           arrowprops=dict(arrowstyle='->', lw=2, color='gray', alpha=0.5))

        # Info panel
        panel_x = layout_width + station_spacing + info_panel_width / 2
        ax.text(panel_x, total_height + 0.5, "Statistics",
               ha='center', va='bottom', fontsize=11, fontweight='bold')

        stats_text = [
            f"Active Sheets: {len(active_sheets)}",
            f"Active Parts: {len(active_parts)}",
            f"Total Waiting Sheets: {sum(len(v) for v in waiting_sheets.values())}",
            f"Total Waiting Parts: {sum(len(v) for v in waiting_parts.values())}",
            "",
            f"Total Sheets: {len(self.solution.sheets)}",
            f"Total Parts: {len(self.all_parts)}",
            "",
            f"Event: {event_type.upper() if event_type else 'N/A'}",
            f"Part: {highlight_part or 'N/A'}",
        ]
        for idx, txt in enumerate(stats_text):
            ax.text(panel_x, total_height - 0.1 - idx * 0.3, txt,
                   ha='center', va='top', fontsize=8)

        # Set limits
        ax.set_xlim(-0.5, layout_width + info_panel_width + station_spacing * 2)
        ax.set_ylim(-0.5, total_height + 1.5)
        ax.set_aspect('equal')
        ax.axis('off')

        # Title with event type indicator
        time_mins = int(current_time // 60)
        time_secs = int(current_time % 60)
        makespan_mins = int(self.makespan // 60)
        makespan_secs = int(self.makespan % 60)

        event_indicator = "START" if event_type == "start" else "END"
        event_color = 'green' if event_type == "start" else 'orange'

        title = f"Frame {frame_idx} | Time: {time_mins}m {time_secs}s / {makespan_mins}m {makespan_secs}s | [{event_indicator}]"
        title += f"\n{event_desc}"
        ax.set_title(title, fontsize=12, fontweight='bold', pad=15, color=event_color)

        # Legend
        legend_elements = [
            mpatches.Patch(facecolor='lightblue', edgecolor='black', label='Sheet Station', alpha=0.5),
            mpatches.Patch(facecolor='lightyellow', edgecolor='black', label='Part Station', alpha=0.5),
            mpatches.Patch(facecolor='lime', edgecolor='red', label='START Event', linewidth=2),
            mpatches.Patch(facecolor='orange', edgecolor='red', label='END Event', linewidth=2),
        ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=8,
                 bbox_to_anchor=(0, 1.15))

    def generate_frames(self, output_dir: str = "output/frame_by_frame",
                       max_frames: int = None):
        """
        Generate frame images at each start/end event time.

        Each frame captures the exact moment when a sheet or part starts
        or ends processing at a station.

        Args:
            output_dir: Directory to save frames
            max_frames: Maximum number of frames to generate (None = all events)
        """
        if not MATPLOTLIB_AVAILABLE:
            print("Cannot generate frames - matplotlib not available")
            return

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Collect all start/end event times
        events = self.collect_event_times()
        total_events = len(events)

        if max_frames and len(events) > max_frames:
            # Sample events evenly to limit frame count
            step = len(events) // max_frames
            events = events[::step][:max_frames]

        print(f"\nGenerating {len(events)} frames (from {total_events} total events)...")
        print(f"Output directory: {output_dir}")

        for frame_idx, (time, event_desc) in enumerate(events):
            fig, ax = plt.subplots(figsize=(18, 10))
            self._draw_frame(ax, time, frame_idx, event_desc)

            frame_path = output_path / f"frame_{frame_idx:05d}_t{time:.1f}.png"
            plt.savefig(frame_path, dpi=120, bbox_inches='tight')
            plt.close()

            if frame_idx % 50 == 0 or frame_idx == len(events) - 1:
                print(f"  Generated frame {frame_idx + 1}/{len(events)} (t={time:.1f})")

        print(f"\nFrames saved to: {output_dir}")

    def _draw_frame(self, ax, current_time: float, frame_idx: int, event_desc: str):
        """Draw a single frame showing current state."""

        # Layout parameters
        num_stations = len(self.stations)
        station_width = 1.8
        machine_height = 0.8
        station_spacing = 0.6
        machine_spacing = 0.25
        waiting_area_height = 2.0
        info_panel_width = 3.0

        max_machines = max(self.station_machines.values())
        machine_area_height = max_machines * (machine_height + machine_spacing)
        total_height = waiting_area_height + machine_area_height
        layout_width = num_stations * (station_width + station_spacing)

        # Collect current state
        # Active sheets (currently processing in sheet stations)
        active_sheets: Dict[str, Tuple[str, int]] = {}  # sheet_id -> (station, machine)
        for sheet_id, assignments in self.solution.schedule.items():
            for a in assignments:
                if a.start_time <= current_time < a.end_time:
                    active_sheets[sheet_id] = (a.station_name, a.machine_index)

        # Active parts (currently processing in part stations)
        active_parts: Dict[str, Tuple[str, int]] = {}  # part_id -> (station, machine)
        for part_id, assignments in self.solution.part_schedule.items():
            for a in assignments:
                if a.start_time <= current_time < a.end_time:
                    active_parts[part_id] = (a.station_name, a.machine_index)

        # Waiting sheets at each station
        waiting_sheets: Dict[str, List[str]] = {s.name: [] for s in self.stations}
        for sheet_id, assignments in self.solution.schedule.items():
            sorted_assignments = sorted(assignments, key=lambda x: x.start_time)
            for idx, a in enumerate(sorted_assignments):
                if a.start_time > current_time:
                    prev_end = sorted_assignments[idx - 1].end_time if idx > 0 else 0.0
                    if prev_end <= current_time:
                        waiting_sheets[a.station_name].append(sheet_id)
                    break

        # Waiting parts at each station
        # Parts can only wait AFTER their sheet completes all sheet-stations
        waiting_parts: Dict[str, List[str]] = {s.name: [] for s in self.stations}
        for part_id, assignments in self.solution.part_schedule.items():
            if not assignments:
                continue

            # Find when this part's sheet finished (part becomes available)
            sheet_id = self.part_to_sheet.get(part_id)
            sheet_end_time = 0.0
            if sheet_id and sheet_id in self.solution.schedule:
                sheet_assignments = self.solution.schedule[sheet_id]
                if sheet_assignments:
                    sheet_end_time = max(a.end_time for a in sheet_assignments)

            # Part can only be waiting if sheet has finished
            if current_time < sheet_end_time:
                continue

            sorted_assignments = sorted(assignments, key=lambda x: x.start_time)
            for idx, a in enumerate(sorted_assignments):
                if a.start_time > current_time:
                    # Part is waiting for this station
                    prev_end = sorted_assignments[idx - 1].end_time if idx > 0 else sheet_end_time
                    if prev_end <= current_time:
                        waiting_parts[a.station_name].append(part_id)
                    break

        # Draw stations
        station_positions = {}
        for i, station in enumerate(self.stations):
            x = i * (station_width + station_spacing)
            num_machines = self.station_machines[station.name]
            is_sheet = self.station_is_sheet[station.name]

            y_offset = waiting_area_height + (
                machine_area_height - num_machines * (machine_height + machine_spacing)
            ) / 2

            # Station label
            station_type = "SHEET" if is_sheet else "PART"
            label_color = 'darkblue' if is_sheet else 'darkgreen'
            ax.text(x + station_width/2, total_height + 0.4,
                   f"{station.name}\n[{station_type}]\n({num_machines} M)",
                   ha='center', va='bottom', fontsize=9, fontweight='bold', color=label_color)

            # Draw machines
            for m in range(num_machines):
                y = y_offset + m * (machine_height + machine_spacing)
                box_color = 'lightblue' if is_sheet else 'lightyellow'
                rect = mpatches.Rectangle((x, y), station_width, machine_height,
                                         linewidth=2, edgecolor='black',
                                         facecolor=box_color, alpha=0.4)
                ax.add_patch(rect)
                ax.text(x + 0.1, y + machine_height - 0.1, f"M{m}",
                       va='top', fontsize=8, color='gray')

            # Waiting area separator
            separator_y = waiting_area_height - 0.15
            ax.plot([x, x + station_width], [separator_y, separator_y],
                   linestyle='--', color='gray', linewidth=1)

            # Waiting list
            if is_sheet:
                wait_list = waiting_sheets.get(station.name, [])
                wait_label = f"Waiting Sheets: {len(wait_list)}"
                if wait_list:
                    display = "\n".join([s.replace('sheet_', 'S') for s in wait_list[:4]])
                    if len(wait_list) > 4:
                        display += f"\n+{len(wait_list) - 4} more"
                else:
                    display = "None"
                ax.text(x + station_width/2, waiting_area_height/2,
                       f"{wait_label}\n{display}",
                       ha='center', va='center', fontsize=7, color='darkblue')
            else:
                wait_list = waiting_parts.get(station.name, [])
                wait_label = f"Waiting Parts: {len(wait_list)}"
                if wait_list:
                    display = "\n".join([p.replace('row_', 'P')[:8] for p in wait_list[:4]])
                    if len(wait_list) > 4:
                        display += f"\n+{len(wait_list) - 4} more"
                else:
                    display = "None"
                ax.text(x + station_width/2, waiting_area_height/2,
                       f"{wait_label}\n{display}",
                       ha='center', va='center', fontsize=7, color='darkgreen')

            station_positions[station.name] = (x, y_offset)

        # Draw active sheets on machines
        for sheet_id, (station_name, machine_idx) in active_sheets.items():
            if station_name not in station_positions:
                continue
            x_base, y_base = station_positions[station_name]
            y = y_base + machine_idx * (machine_height + machine_spacing)

            # Get progress
            assignments = self.solution.schedule.get(sheet_id, [])
            current_assignment = None
            for a in assignments:
                if a.station_name == station_name and a.start_time <= current_time < a.end_time:
                    current_assignment = a
                    break

            progress = 0.5
            if current_assignment:
                progress = (current_time - current_assignment.start_time) / current_assignment.duration
                progress = max(0, min(1, progress))

            # Draw sheet
            sw, sh = station_width * 0.85, machine_height * 0.7
            sheet_x = x_base + (station_width - sw) / 2
            sheet_y = y + (machine_height - sh) / 2
            color = plt.cm.RdYlGn(1 - progress)

            sheet_rect = mpatches.Rectangle((sheet_x, sheet_y), sw, sh,
                                           linewidth=2, edgecolor='darkblue',
                                           facecolor=color, alpha=0.9)
            ax.add_patch(sheet_rect)

            # Sheet label with part count
            sheet_obj = self.solution.get_sheet_by_id(sheet_id)
            num_parts = len(sheet_obj.assigned_parts) if sheet_obj else 0
            label = f"{sheet_id.replace('sheet_', 'S')}\n({num_parts} parts)"
            ax.text(sheet_x + sw/2, sheet_y + sh/2, label,
                   ha='center', va='center', fontsize=6, fontweight='bold', color='white')

        # Draw active parts on machines
        for part_id, (station_name, machine_idx) in active_parts.items():
            if station_name not in station_positions:
                continue
            x_base, y_base = station_positions[station_name]
            y = y_base + machine_idx * (machine_height + machine_spacing)

            # Get progress
            assignments = self.solution.part_schedule.get(part_id, [])
            current_assignment = None
            for a in assignments:
                if a.station_name == station_name and a.start_time <= current_time < a.end_time:
                    current_assignment = a
                    break

            progress = 0.5
            if current_assignment:
                progress = (current_time - current_assignment.start_time) / current_assignment.duration
                progress = max(0, min(1, progress))

            # Draw part (smaller, rounded)
            pw, ph = station_width * 0.65, machine_height * 0.5
            part_x = x_base + (station_width - pw) / 2
            part_y = y + (machine_height - ph) / 2
            color = plt.cm.RdYlGn(1 - progress)

            part_rect = mpatches.FancyBboxPatch(
                (part_x, part_y), pw, ph,
                boxstyle="round,pad=0.02,rounding_size=0.15",
                linewidth=1.5, edgecolor='darkgreen',
                facecolor=color, alpha=0.9)
            ax.add_patch(part_rect)

            label = part_id.replace('row_', 'P')[:7]
            ax.text(part_x + pw/2, part_y + ph/2, label,
                   ha='center', va='center', fontsize=5, fontweight='bold', color='black')

        # Draw arrows between stations
        arrow_y = waiting_area_height + machine_area_height / 2
        for i in range(len(self.stations) - 1):
            x1 = i * (station_width + station_spacing) + station_width
            x2 = (i + 1) * (station_width + station_spacing)

            curr_is_sheet = self.station_is_sheet[self.stations[i].name]
            next_is_sheet = self.station_is_sheet[self.stations[i + 1].name]

            if curr_is_sheet and not next_is_sheet:
                ax.annotate('', xy=(x2, arrow_y), xytext=(x1, arrow_y),
                           arrowprops=dict(arrowstyle='->', lw=3, color='red', alpha=0.8))
                ax.text((x1 + x2) / 2, arrow_y + 0.2, "SPLIT",
                       ha='center', va='bottom', fontsize=8, color='red', fontweight='bold')
            else:
                ax.annotate('', xy=(x2, arrow_y), xytext=(x1, arrow_y),
                           arrowprops=dict(arrowstyle='->', lw=2, color='gray', alpha=0.5))

        # Info panel
        panel_x = layout_width + station_spacing + info_panel_width / 2
        ax.text(panel_x, total_height + 0.5, "Statistics",
               ha='center', va='bottom', fontsize=11, fontweight='bold')

        stats_text = [
            f"Active Sheets: {len(active_sheets)}",
            f"Active Parts: {len(active_parts)}",
            f"Total Waiting Sheets: {sum(len(v) for v in waiting_sheets.values())}",
            f"Total Waiting Parts: {sum(len(v) for v in waiting_parts.values())}",
            "",
            f"Total Sheets: {len(self.solution.sheets)}",
            f"Total Parts: {len(self.all_parts)}",
        ]
        for idx, txt in enumerate(stats_text):
            ax.text(panel_x, total_height - 0.1 - idx * 0.3, txt,
                   ha='center', va='top', fontsize=8)

        # Set limits
        ax.set_xlim(-0.5, layout_width + info_panel_width + station_spacing * 2)
        ax.set_ylim(-0.5, total_height + 1.5)
        ax.set_aspect('equal')
        ax.axis('off')

        # Title
        time_mins = int(current_time // 60)
        time_secs = int(current_time % 60)
        makespan_mins = int(self.makespan // 60)
        makespan_secs = int(self.makespan % 60)
        title = f"Frame {frame_idx} | Time: {time_mins}m {time_secs}s / {makespan_mins}m {makespan_secs}s"
        title += f"\n{event_desc[:60]}"
        ax.set_title(title, fontsize=12, fontweight='bold', pad=15)

        # Legend
        legend_elements = [
            mpatches.Patch(facecolor='lightblue', edgecolor='black', label='Sheet Station', alpha=0.5),
            mpatches.Patch(facecolor='lightyellow', edgecolor='black', label='Part Station', alpha=0.5),
            mpatches.Patch(facecolor='lightgreen', edgecolor='darkblue', label='Starting'),
            mpatches.Patch(facecolor='lightcoral', edgecolor='darkblue', label='Finishing'),
        ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=8,
                 bbox_to_anchor=(0, 1.15))


def generate_frame_by_frame(solution: Solution, problem: Problem,
                           output_dir: str = "output/frame_by_frame",
                           max_frames: int = 200,
                           use_part_frames: bool = True):
    """
    Main function to generate frame-by-frame visualization and CSV files.

    Generates frames at each start/end event for sheets and parts,
    making it easy to track the flow through stations.

    Args:
        solution: The solution to visualize
        problem: The problem instance
        output_dir: Directory for output files (default: output/frame_by_frame)
        max_frames: Maximum frames to generate (None = all events)
        use_part_frames: If True, use part-centric frames (2 frames per part per station:
                        start and end). If False, use original event-based frames.
    """
    generator = FrameByFrameGenerator(solution, problem)

    # Generate frames
    if use_part_frames:
        # Part-centric frames: exactly 2 frames per part per station (start + end)
        # When end of station N equals start of station N+1, creates sequential frames
        generator.generate_part_frames(output_dir, max_frames=max_frames)
    else:
        # Original: frames at all start/end events
        generator.generate_frames(output_dir, max_frames=max_frames)

    return generator


if __name__ == "__main__":
    from models import Problem
    from solvers import GreedySolver
    from evaluation import WeightedEvaluator

    # Load config
    config_path = "config/config.json"
    with open(config_path, 'r') as f:
        config = json.load(f)

    data_file = config.get("data_file", "data/5693_cleaned.xlsx")

    # Load evaluator parameters
    evaluator_config = config.get("evaluator", {})
    alpha = evaluator_config.get("alpha", 1.0)
    beta = evaluator_config.get("beta", 0.5)
    gamma = evaluator_config.get("gamma", 0.3)

    # Load problem and solve
    print("Loading problem...")
    problem = Problem.load_from_files(data_file, config_path)
    print(f"Problem: {problem}")

    print("\nSolving...")
    solver = GreedySolver(sort_by='area_desc')
    evaluator = WeightedEvaluator(alpha=alpha, beta=beta, gamma=gamma)
    solution = solver.solve(problem, evaluator)

    print(f"Solution: {solution.num_sheets()} sheets, Makespan: {solution.get_makespan():.2f}")
    print(f"Part schedules: {len(solution.part_schedule)}")

    # Generate frame-by-frame output
    print("\n" + "="*50)
    print("Generating frame-by-frame visualization...")
    print("="*50)

    generator = generate_frame_by_frame(
        solution, problem,
        output_dir="output/frame_by_frame",
        max_frames=100  # Limit frames for initial test
    )

    print("\nDone!")
