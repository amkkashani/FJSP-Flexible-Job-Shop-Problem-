"""Flow animation for FJSP solutions - shows sheets moving through stations and machines."""

import sys
from pathlib import Path
from typing import Dict, List, Tuple
import json

sys.path.insert(0, str(Path(__file__).parent))

from models import Problem
from solution import Solution

# Try to import matplotlib for animation
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.animation import FuncAnimation, PillowWriter
    import numpy as np
    MATPLOTLIB_AVAILABLE = True
except Exception as exc:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available. Animation cannot be generated.")
    print(f"Reason: {exc}")
    print("Install with: pip install matplotlib pillow")


class FlowAnimator:
    """Animates the flow of sheets through stations and machines."""

    def __init__(self, solution: Solution, problem: Problem):
        self.solution = solution
        self.problem = problem
        self.makespan = solution.get_makespan()

        # Build station-machine structure
        self.stations = []
        self.station_machines = {}  # station_name -> num_machines
        for station in problem.stations:
            self.stations.append(station)
            self.station_machines[station.name] = station.num_machines

    def create_animation(self, output_path: str = "output/flow_animation.gif",
                        fps: int = 10, duration_seconds: int = 20,
                        max_sheets: int = None):
        """
        Create an animated GIF showing sheet flow through stations.

        Args:
            output_path: Path to save the GIF
            fps: Frames per second
            duration_seconds: Total animation duration
            max_sheets: Maximum number of sheets to show (for performance)
        """
        if not MATPLOTLIB_AVAILABLE:
            print("Cannot create animation - matplotlib not available")
            return

        # Calculate time parameters
        total_frames = fps * duration_seconds
        time_step = self.makespan / total_frames

        # Select sheets to animate
        sheets_to_show = self.solution.sheets
        if max_sheets and len(sheets_to_show) > max_sheets:
            sheets_to_show = sheets_to_show[:max_sheets]

        print(f"\nCreating flow animation...")
        print(f"  Sheets: {len(sheets_to_show)}")
        print(f"  Stations: {len(self.stations)}")
        print(f"  Makespan: {self.makespan:.2f}")
        print(f"  Total frames: {total_frames}")

        # Create figure
        fig, ax = plt.subplots(figsize=(16, 10))

        def animate(frame):
            ax.clear()
            current_time = frame * time_step

            # Draw the animation for this time frame
            self._draw_frame(ax, current_time, sheets_to_show)

            # Progress indicator
            if frame % 10 == 0:
                progress = (frame / total_frames) * 100
                print(f"  Progress: {progress:.1f}%", end='\r')

            return ax,

        # Create animation
        anim = FuncAnimation(fig, animate, frames=total_frames,
                           interval=1000/fps, blit=False, repeat=True)

        # Save as GIF
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        writer = PillowWriter(fps=fps)
        anim.save(output_path, writer=writer)

        plt.close()
        print(f"\n  Animation saved to: {output_path}")

    def _draw_frame(self, ax, current_time: float, sheets_to_show: List):
        """Draw a single frame of the animation."""

        # Setup layout
        num_stations = len(self.stations)
        station_width = 1.5
        machine_height = 0.6
        station_spacing = 0.5
        machine_spacing = 0.2
        waiting_area_height = 1.0

        # Calculate total height needed
        max_machines = max(self.station_machines.values())
        machine_area_height = max_machines * (machine_height + machine_spacing)
        total_height = waiting_area_height + machine_area_height

        # Determine sheets waiting at each station (finished previous step but not yet started here)
        waiting_by_station: Dict[str, List[str]] = {s.name: [] for s in self.stations}
        for sheet in sheets_to_show:
            assignments = sorted(self.solution.schedule.get(sheet.id, []),
                                 key=lambda a: a.start_time)
            for idx, assignment in enumerate(assignments):
                if assignment.start_time > current_time:
                    prev_end = assignments[idx - 1].end_time if idx > 0 else 0.0
                    if prev_end <= current_time + 1e-9:
                        waiting_by_station[assignment.station_name].append(sheet.id)
                    break

        # Draw stations and machines
        station_positions = {}  # station_name -> (x, y_base)

        for i, station in enumerate(self.stations):
            x = i * (station_width + station_spacing)
            num_machines = self.station_machines[station.name]

            # Center machines vertically
            y_offset = waiting_area_height + (
                machine_area_height - num_machines * (machine_height + machine_spacing)
            ) / 2

            # Draw station label
            ax.text(x + station_width/2, total_height + 0.3,
                   f"{station.name}\n({num_machines} machines)",
                   ha='center', va='bottom', fontsize=10, fontweight='bold')

            # Draw each machine
            for m in range(num_machines):
                y = y_offset + m * (machine_height + machine_spacing)

                # Machine box
                rect = mpatches.Rectangle((x, y), station_width, machine_height,
                                         linewidth=2, edgecolor='black',
                                         facecolor='lightgray', alpha=0.3)
                ax.add_patch(rect)

                # Machine label
                ax.text(x + 0.1, y + machine_height/2, f"M{m}",
                       va='center', fontsize=8, color='gray')

            # Waiting section under the machines
            separator_y = waiting_area_height - 0.1
            ax.plot([x, x + station_width], [separator_y, separator_y],
                    linestyle='--', color='gray', linewidth=1, alpha=0.7)
            ax.text(x + station_width/2, separator_y - 0.05, "-------",
                    ha='center', va='top', fontsize=8, color='gray')

            # Skip waiting list for first station; list others line-by-line
            if i > 0:
                waiting_labels = waiting_by_station.get(station.name, [])
                waiting_display = "\n".join(
                    s.replace('SHEET_', '') for s in waiting_labels
                ) if waiting_labels else "None"
                ax.text(x + station_width/2, waiting_area_height / 2,
                       f"Waiting:\n{waiting_display}",
                       ha='center', va='center', fontsize=8, color='black')

            station_positions[station.name] = (x, y_offset)

        # Draw sheets on machines
        sheets_drawn = 0
        for sheet in sheets_to_show:
            if sheet.id not in self.solution.schedule:
                continue

            assignments = self.solution.schedule[sheet.id]

            # Find current assignment for this sheet
            for assignment in assignments:
                if assignment.start_time <= current_time <= assignment.end_time:
                    # Sheet is currently being processed
                    station_name = assignment.station_name
                    machine_idx = assignment.machine_index

                    if station_name not in station_positions:
                        continue

                    x_base, y_base = station_positions[station_name]
                    y = y_base + machine_idx * (machine_height + machine_spacing)

                    # Progress through the machine
                    progress = (current_time - assignment.start_time) / assignment.duration
                    progress = max(0, min(1, progress))

                    # Sheet representation
                    sheet_width = station_width * 0.8
                    sheet_height = machine_height * 0.6
                    sheet_x = x_base + (station_width - sheet_width) / 2
                    sheet_y = y + (machine_height - sheet_height) / 2

                    # Color based on progress (green -> blue)
                    color = plt.cm.RdYlGn(progress)

                    # Draw sheet
                    sheet_rect = mpatches.Rectangle((sheet_x, sheet_y),
                                                    sheet_width, sheet_height,
                                                    linewidth=2, edgecolor='darkblue',
                                                    facecolor=color, alpha=0.8)
                    ax.add_patch(sheet_rect)

                    # Sheet ID
                    sheet_label = sheet.id.replace('SHEET_', '')
                    ax.text(sheet_x + sheet_width/2, sheet_y + sheet_height/2,
                           sheet_label, ha='center', va='center',
                           fontsize=7, fontweight='bold', color='white')

                    sheets_drawn += 1
                    break

        # Draw arrows between stations
        arrow_y = waiting_area_height + (machine_area_height / 2)
        for i in range(len(self.stations) - 1):
            x1 = i * (station_width + station_spacing) + station_width
            x2 = (i + 1) * (station_width + station_spacing)
            y = arrow_y

            ax.annotate('', xy=(x2, y), xytext=(x1, y),
                       arrowprops=dict(arrowstyle='->', lw=2, color='gray', alpha=0.5))

        # Set axis properties
        ax.set_xlim(-0.5, num_stations * (station_width + station_spacing))
        ax.set_ylim(-0.5, total_height + 1.2)
        ax.set_aspect('equal')
        ax.axis('off')

        # Title with current time
        title = f"Sheet Flow Through Stations - Time: {current_time:.1f} / {self.makespan:.1f}"
        title += f"\nActive Sheets: {sheets_drawn}"
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

        # Add legend
        legend_elements = [
            mpatches.Patch(facecolor='lightgreen', edgecolor='darkblue', label='Starting'),
            mpatches.Patch(facecolor='yellow', edgecolor='darkblue', label='Processing'),
            mpatches.Patch(facecolor='lightcoral', edgecolor='darkblue', label='Finishing')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

    def create_gantt_chart(self, output_path: str = "output/gantt_chart.png",
                          max_sheets: int = 20):
        """
        Create a static Gantt chart showing all sheet schedules.

        Args:
            output_path: Path to save the image
            max_sheets: Maximum number of sheets to display
        """
        if not MATPLOTLIB_AVAILABLE:
            print("Cannot create Gantt chart - matplotlib not available")
            return

        sheets_to_show = self.solution.sheets[:max_sheets]

        print(f"\nCreating Gantt chart for {len(sheets_to_show)} sheets...")

        fig, ax = plt.subplots(figsize=(16, max(10, len(sheets_to_show) * 0.3)))

        # Color map for stations
        station_colors = {}
        colors = plt.cm.Set3(np.linspace(0, 1, len(self.stations)))
        for i, station in enumerate(self.stations):
            station_colors[station.name] = colors[i]

        # Draw bars for each sheet
        for idx, sheet in enumerate(sheets_to_show):
            y_pos = idx

            if sheet.id not in self.solution.schedule:
                continue

            assignments = sorted(self.solution.schedule[sheet.id],
                               key=lambda a: a.start_time)

            for assignment in assignments:
                duration = assignment.end_time - assignment.start_time
                color = station_colors[assignment.station_name]

                # Draw bar
                ax.barh(y_pos, duration, left=assignment.start_time,
                       height=0.8, color=color, edgecolor='black', linewidth=0.5)

                # Add machine label
                label = f"{assignment.station_name}:M{assignment.machine_index}"
                ax.text(assignment.start_time + duration/2, y_pos, label,
                       ha='center', va='center', fontsize=6, fontweight='bold')

        # Formatting
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('Sheet ID', fontsize=12)
        ax.set_title('Gantt Chart - Sheet Processing Schedule', fontsize=14, fontweight='bold')
        ax.set_yticks(range(len(sheets_to_show)))
        ax.set_yticklabels([s.id for s in sheets_to_show], fontsize=8)
        ax.grid(axis='x', alpha=0.3)

        # Legend
        legend_elements = [mpatches.Patch(facecolor=station_colors[s.name],
                                         edgecolor='black', label=f"{s.name} ({self.station_machines[s.name]}m)")
                          for s in self.stations]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=9)

        plt.tight_layout()
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"  Gantt chart saved to: {output_path}")


def create_flow_animation(solution: Solution, problem: Problem,
                         output_path: str = "output/flow_animation.gif",
                         fps: int = 10, duration: int = 20, max_sheets: int = 50):
    """
    Main function to create flow animation.

    Args:
        solution: The solution to animate
        problem: The problem instance
        output_path: Path to save the GIF
        fps: Frames per second
        duration: Animation duration in seconds
        max_sheets: Maximum sheets to show
    """
    animator = FlowAnimator(solution, problem)
    animator.create_animation(output_path, fps, duration, max_sheets)
    return animator


if __name__ == "__main__":
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
    evaluator = WeightedEvaluator(alpha=1.0, beta=0.5, gamma=0.3)
    solution = solver.solve(problem, evaluator)

    print(f"\nSolution: {solution.num_sheets()} sheets, Makespan: {solution.get_makespan():.2f}")

    # Create animation
    animator = FlowAnimator(solution, problem)
    animator.create_animation("output/flow_animation.gif", fps=15, duration_seconds=30, max_sheets=50)
    animator.create_gantt_chart("output/gantt_chart.png", max_sheets=30)

    print("\nDone!")
