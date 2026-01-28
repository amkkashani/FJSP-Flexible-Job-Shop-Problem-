#!/usr/bin/env python3
"""Generate station-flow frames (PNG sequence) from config + CSV event log.

Key behaviors (per user spec):
- Sheet stays together only while station config has `"sheet": true`.
- After sheet completion, parts split and travel independently.
- A part is 'waiting for a station' if:
  1) its sheet has completed (sheet_end_time <= current_time)
  2) the part is not currently active in any station
  3) the part has a future station event
  Then it is queued under the *next* station.
- Queues list Part IDs (not just counts).
- Product progress shows products with at least one completed part, but not fully completed.
- Statistics include:
  1. Total Completed Parts (count of 'part_complete' events).
  2. Total Completed Products (count of products where completed parts == total parts).
- Simultaneous events are aggregated into a single frame.
- **Consistent Logic:** Progress list and Statistics now BOTH rely on 'part_complete' events.
- **Fixed:** In-Progress list now limits to 8 items and shows "+ X more" if there are overflow items.
"""

from __future__ import annotations
import argparse
import bisect
import json
import math
import os
import zipfile
from dataclasses import dataclass
from typing import Dict, List, Tuple, Set

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import pandas as pd

# Type Alias for Part Intervals:
# (start_times, end_times, station_names, machine_indices)
PartIntervals = Tuple[List[float], List[float], List[str], List[int]]


def format_hms(seconds: float) -> str:
    total_seconds = int(seconds)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


@dataclass
class Station:
    """Represents a processing station configuration."""
    name: str
    order_index: int
    num_machines: int
    is_sheet: bool


def load_stations(config: dict) -> List[Station]:
    """Parses station configurations from the loaded JSON dictionary."""
    stations: List[Station] = []
    for station_cfg in config["stations"]:
        stations.append(
            Station(
                name=station_cfg["name"],
                order_index=int(station_cfg["order_index"]),
                num_machines=int(station_cfg["num_machines"]),
                is_sheet=bool(station_cfg.get("sheet", False)),
            )
        )
    stations.sort(key=lambda s: s.order_index)
    return stations


def preprocess(events: pd.DataFrame, sheet_parts: pd.DataFrame):
    """
    Pre-calculates mappings and intervals to speed up frame generation.
    """
    # Create mappings from the static sheet definition file
    part_to_sheet = dict(zip(sheet_parts["Part ID"], sheet_parts["Sheet ID"]))
    part_to_product = dict(zip(sheet_parts["Part ID"], sheet_parts["Product ID"]))
    prod_total = sheet_parts.groupby("Product ID")["Part ID"].nunique().to_dict()

    # Map areas from static sheet definition file
    part_to_area = dict(zip(sheet_parts["Part ID"], sheet_parts["Area (m2)"].astype(float)))
    sheet_to_area = (
        sheet_parts.groupby("Sheet ID")["Area (m2)"]
        .sum()
        .astype(float)
        .to_dict()
    )

    # Calculate when each sheet finishes processing
    sheet_end_times = (
        events.loc[events.event_type == "sheet_end", ["sheet_id", "time"]]
        .groupby("sheet_id")["time"]
        .max()
        .to_dict()
    )

    # Filter for part start events to build timelines
    part_starts_df = events[events.event_type == "part_start"].copy()
    part_starts_df = part_starts_df.sort_values(["entity_id", "time"])

    # Map each entity (part) to its sheet based on the first event occurrence
    part_initial_sheet = part_starts_df.groupby("entity_id")["sheet_id"].first().to_dict()

    # Build the interval structures for efficient querying during animation
    part_intervals: Dict[str, PartIntervals] = {}

    for part_id, group in part_starts_df.groupby("entity_id"):
        part_intervals[part_id] = (
            group["time"].tolist(),
            group["end_time"].tolist(),
            group["station"].tolist(),
            group["machine"].astype(int).tolist(),
        )

    # Determine when each part is fully done based on machine intervals (for visualization/queueing)
    interval_completion_times = {
        part_id: max(intervals[1])
        for part_id, intervals in part_intervals.items()
    }

    # --- NEW LOGIC START ---
    # Create a specific map for part completion based ONLY on 'part_complete' events.
    # This ensures the progress list matches the statistics perfectly.
    part_complete_events = events[events.event_type == 'part_complete']

    # Map Part ID -> Time of part_complete event
    # If a part has multiple complete events (unlikely but possible), this takes the last one.
    part_completion_times = part_complete_events.sort_values('time').groupby('entity_id')['time'].last().to_dict()

    # Also keep the sorted timestamps for the quick "Total Completed Parts" counter
    part_complete_timestamps = sorted(part_complete_events['time'].tolist())
    # --- NEW LOGIC END ---

    return (
        part_to_sheet,
        part_to_product,
        prod_total,
        sheet_end_times,
        part_intervals,
        interval_completion_times,
        part_initial_sheet,
        part_complete_timestamps,
        part_completion_times,
        part_to_area,
        sheet_to_area,
    )


def compute_sheet_state(
        current_time: float,
        events: pd.DataFrame,
        sheet_station_name: str
) -> Tuple[Dict[str, List[str]], Dict[str, Tuple[str, int, float, int]]]:
    """
    Determines which sheets are waiting or active at the specific 'sheet' station.
    """
    sheet_starts = events[events.event_type == "sheet_start"]
    active_sheets = {}

    for _, row in sheet_starts.iterrows():
        # Check if the sheet is currently being processed
        if row.time <= current_time < row.end_time:
            active_sheets[row.entity_id] = (
                row.station,
                int(row.machine),
                float(row.end_time),
                int(row.num_parts)
            )

    # Sheets that haven't started yet are considered "waiting"
    waiting_sheets_list = sorted(set(sheet_starts.loc[sheet_starts.time > current_time, "entity_id"].tolist()))

    waiting_sheets = {sheet_station_name: waiting_sheets_list}
    return waiting_sheets, active_sheets


def compute_frame_state(
        current_time: float,
        stations: List[Station],
        part_sheet_map: Dict[str, str],
        sheet_end_times: Dict[str, float],
        part_intervals: Dict[str, PartIntervals],
        interval_completion_times: Dict[str, float],
):
    """
    Calculates the status (Waiting, Active) of every part at `current_time`.
    """
    # Initialize waiting queues for non-sheet stations
    waiting_parts: Dict[str, List[str]] = {s.name: [] for s in stations if not s.is_sheet}
    active_parts: Dict[str, Tuple[str, int, float]] = {}

    for part_id, (start_times, end_times, station_names, machines) in part_intervals.items():
        # 1. Check if part is already fully completed based on intervals
        # We still use interval_completion_times here to stop drawing the part
        # once it physically leaves the last machine.
        final_time = interval_completion_times.get(part_id, math.inf)
        if final_time <= current_time:
            continue

        # 2. Check if part is currently ACTIVE in a machine
        idx_active = bisect.bisect_right(start_times, current_time) - 1

        if idx_active >= 0 and current_time < end_times[idx_active]:
            active_parts[part_id] = (
                station_names[idx_active],
                machines[idx_active],
                float(end_times[idx_active])
            )
            continue

        # 3. Check if part is WAITING for the next station
        idx_next = bisect.bisect_right(start_times, current_time)

        if idx_next < len(start_times):
            next_station = station_names[idx_next]

            # Check prerequisites for waiting:
            # The sheet containing this part must have finished processing
            associated_sheet = part_sheet_map.get(part_id)
            sheet_finish_time = sheet_end_times.get(associated_sheet, math.inf)

            if sheet_finish_time <= current_time:
                # Add to the queue of the *next* station
                if next_station in waiting_parts:
                    waiting_parts.setdefault(next_station, []).append(part_id)

    # Sort queues for consistent rendering
    for station_name in list(waiting_parts.keys()):
        waiting_parts[station_name] = sorted(waiting_parts[station_name])

    return waiting_parts, active_parts


def compute_product_progress(
        current_time: float,
        part_to_product: Dict[str, str],
        prod_total: Dict[str, int],
        part_completion_times: Dict[str, float],
        # display_limit: int = 12, # Removed limit here, handled in draw_frame now
) -> List[Tuple[str, int, int]]:
    """Calculates progress (parts done / total parts) for products currently in production."""
    completed_counts_by_prod: Dict[str, int] = {}

    # Count completed parts per product based on the 'part_complete' events
    for part_id, product_id in part_to_product.items():
        # Using the map derived from 'part_complete' events specifically
        if part_completion_times.get(part_id, math.inf) <= current_time:
            completed_counts_by_prod[product_id] = completed_counts_by_prod.get(product_id, 0) + 1

    progress_list = []
    for product_id, count_done in completed_counts_by_prod.items():
        total_parts = prod_total.get(product_id, 0)
        # Only show products that are partially done (started but not finished)
        # If count_done == total_parts, it will now be correctly excluded
        if total_parts and 0 < count_done < total_parts:
            progress_list.append((product_id, count_done, total_parts))

    progress_list.sort(key=lambda x: (-x[1] / x[2], x[0]))
    return progress_list


def draw_frame(
        ax,
        stations: List[Station],
        frame_idx: int,
        current_time: float,
        current_events: pd.DataFrame,
        waiting_parts: Dict[str, List[str]],
        active_parts: Dict[str, Tuple[str, int, float]],
        total_completed_parts: int,
        total_completed_products: int,
        sheet_waiting: Dict[str, List[str]],
        sheet_active: Dict[str, Tuple[str, int, float, int]],
        product_progress: List[Tuple[str, int, int]],
        part_area_map: Dict[str, float],
        sheet_area_map: Dict[str, float],
):
    """Renders a single frame of the simulation onto the provided Matplotlib axis."""
    ax.set_axis_off()

    # --- Layout Configuration ---
    LEFT_MARGIN = 0.04
    RIGHT_STATS_START = 0.80
    STATION_Y_BASE = 0.40
    STATION_HEIGHT = 0.28

    num_stations = len(stations)
    station_slot_width = (RIGHT_STATS_START - LEFT_MARGIN) / num_stations

    # --- Logic to Summarize Simultaneous Events ---
    if len(current_events) == 0:
        event_type_str = "SIMULATION COMPLETE"
        entity_str = "All Tasks Finished"
        current_station_highlight = None
        unique_stations = []
    else:
        unique_types = current_events['event_type'].unique()
        unique_entities = current_events['entity_id'].unique()

        if len(unique_types) == 1:
            event_type_str = str(unique_types[0]).upper()
            if len(current_events) > 1:
                event_type_str += f" ({len(current_events)})"
        else:
            event_type_str = f"MULTIPLE TYPES ({len(current_events)})"

        if len(unique_entities) == 1:
            entity_str = str(unique_entities[0])
        else:
            entity_str = f"{len(unique_entities)} Entities"
            if len(unique_entities) <= 2:
                entity_str = ", ".join(unique_entities)

        unique_stations = current_events['station'].unique()
        current_station_highlight = unique_stations[0] if len(unique_stations) == 1 else None

    # --- Title & Header ---
    ax.text(
        0.5, 0.90,
        f"Frame {frame_idx} | Time: {format_hms(current_time)} | {event_type_str}",
        ha="center", va="center", fontsize=16, fontweight="bold",
        color="darkgreen", transform=ax.transAxes,
    )
    ax.text(
        0.5, 0.865,
        f"{entity_str} {'at ' + str(unique_stations[0]) if len(unique_stations) == 1 else ''}",
        ha="center", va="center", fontsize=13,
        color="darkgreen", transform=ax.transAxes,
    )

    # --- Legend ---
    ax.add_patch(
        patches.Rectangle((0.02, 0.94), 0.015, 0.015, transform=ax.transAxes, facecolor="#e8f4ff", edgecolor="gray"))
    ax.text(0.04, 0.947, "Sheet Station", transform=ax.transAxes, va="center", fontsize=9)
    ax.add_patch(
        patches.Rectangle((0.12, 0.94), 0.015, 0.015, transform=ax.transAxes, facecolor="#fffbe6", edgecolor="gray"))
    ax.text(0.14, 0.947, "Part Station", transform=ax.transAxes, va="center", fontsize=9)
    ax.add_patch(
        patches.Rectangle((0.22, 0.94), 0.015, 0.015, transform=ax.transAxes, facecolor="none", edgecolor="red",
                          linewidth=2))
    ax.text(0.24, 0.947, "Current Station", transform=ax.transAxes, va="center", fontsize=9)

    # --- Draw Stations ---
    station_names = [s.name for s in stations]

    for i, station in enumerate(stations):
        x_pos = LEFT_MARGIN + i * station_slot_width + 0.005
        width = station_slot_width - 0.01

        ax.text(
            x_pos + width / 2,
            STATION_Y_BASE + STATION_HEIGHT + 0.08,
            f"{station.name}\n[{'SHEET' if station.is_sheet else 'PART'}]\n({station.num_machines} M)",
            ha="center", va="bottom", fontsize=11, fontweight="bold",
            color=("red" if station.is_sheet else "darkgreen"),
            transform=ax.transAxes,
        )

        machine_box_height = STATION_HEIGHT / max(station.num_machines, 1)

        for m_idx in range(station.num_machines):
            y_pos = STATION_Y_BASE + (station.num_machines - 1 - m_idx) * machine_box_height
            fill_color = "#e8f4ff" if station.is_sheet else "#fffbe6"

            rect = patches.Rectangle(
                (x_pos, y_pos), width, machine_box_height - 0.01,
                transform=ax.transAxes, facecolor=fill_color,
                edgecolor="gray", linewidth=2,
            )
            ax.add_patch(rect)
            ax.text(x_pos + 0.005, y_pos + machine_box_height - 0.03, f"M{m_idx}", transform=ax.transAxes, fontsize=9,
                    color="gray")

            if station.is_sheet:
                for sheet_id, (st_name, mach_idx, _, num_parts) in sheet_active.items():
                    if st_name == station.name and mach_idx == m_idx:
                        ax.text(
                            x_pos + width / 2, y_pos + (machine_box_height - 0.01) / 2,
                            f"{sheet_id}\n({num_parts} parts)",
                            ha="center", va="center", transform=ax.transAxes,
                            fontsize=10, fontweight="bold",
                        )
            else:
                for part_id, (st_name, mach_idx, _) in active_parts.items():
                    if st_name == station.name and mach_idx == m_idx:
                        ax.text(
                            x_pos + width / 2, y_pos + (machine_box_height - 0.01) / 2,
                            f"{part_id}",
                            ha="center", va="center", transform=ax.transAxes,
                            fontsize=10, fontweight="bold",
                        )

        # --- Queues ---
        queue_y_start = STATION_Y_BASE - 0.11
        if station.is_sheet:
            queue = sheet_waiting.get(station.name, [])
            queue_area = sum(sheet_area_map.get(sheet_id, 0.0) for sheet_id in queue)
            header_text = f"Waiting Sheets: {len(queue)}\nArea: {queue_area:.3f} m2"
            color = "navy"
            display_items = queue[:4]
            max_items = 4
        else:
            queue = waiting_parts.get(station.name, [])
            queue_area = sum(part_area_map.get(part_id, 0.0) for part_id in queue)
            header_text = f"Waiting Parts: {len(queue)}\nArea: {queue_area:.3f} m2"
            color = "darkgreen"
            display_items = queue[:10]
            max_items = 10

        ax.text(x_pos + width / 2, queue_y_start, header_text, ha="center", va="top", fontsize=9,
                transform=ax.transAxes, color=color)

        line_height = 0.03 if station.is_sheet else 0.02
        font_size = 8 if station.is_sheet else 7.5
        header_lines = header_text.count("\n") + 1
        items_start = queue_y_start - (line_height * header_lines)

        for j, item in enumerate(display_items):
            ax.text(x_pos + width / 2, items_start - line_height * j, item, ha="center", va="top",
                    fontsize=font_size, transform=ax.transAxes, color=color)

        if len(queue) > max_items:
            ax.text(x_pos + width / 2, items_start - line_height * max_items, f"+{len(queue) - max_items} more",
                    ha="center", va="top", fontsize=font_size, transform=ax.transAxes, color=color)

        if i < num_stations - 1:
            next_x_pos = LEFT_MARGIN + (i + 1) * station_slot_width + 0.005
            ax.annotate("", xy=(next_x_pos - 0.005, STATION_Y_BASE + STATION_HEIGHT / 2),
                        xytext=(x_pos + width + 0.005, STATION_Y_BASE + STATION_HEIGHT / 2), xycoords=ax.transAxes,
                        arrowprops=dict(arrowstyle="->", color="gray", lw=2))

    if current_station_highlight and current_station_highlight in station_names:
        idx = station_names.index(current_station_highlight)
        x_pos = LEFT_MARGIN + idx * station_slot_width + 0.005
        width = station_slot_width - 0.01
        ax.add_patch(
            patches.Rectangle((x_pos, STATION_Y_BASE), width, STATION_HEIGHT, transform=ax.transAxes, fill=False,
                              edgecolor="red", linewidth=3))

    # --- Statistics Side Panel ---
    stats_x = 0.86
    ax.text(stats_x, 0.78, "Statistics", transform=ax.transAxes, fontsize=14, fontweight="bold")
    ax.text(stats_x, 0.72, f"Active Sheets: {len(sheet_active)}", transform=ax.transAxes, fontsize=10)
    ax.text(stats_x, 0.69, f"Active Parts: {len(active_parts)}", transform=ax.transAxes, fontsize=10)
    ax.text(stats_x, 0.66, f"Total Waiting Sheets: {sum(len(v) for v in sheet_waiting.values())}",
            transform=ax.transAxes, fontsize=10)
    ax.text(stats_x, 0.63, f"Total Waiting Parts: {sum(len(v) for v in waiting_parts.values())}",
            transform=ax.transAxes, fontsize=10)

    # --- COMPLETION STATS ---
    ax.text(stats_x, 0.60, f"Completed Parts: {total_completed_parts}", transform=ax.transAxes, fontsize=10,
            color="darkgreen", fontweight="bold")
    ax.text(stats_x, 0.57, f"Completed Products: {total_completed_products}", transform=ax.transAxes, fontsize=10,
            color="darkgreen", fontweight="bold")

    ax.text(stats_x, 0.53, f"Event: {event_type_str}", transform=ax.transAxes, fontsize=10)
    ax.text(stats_x, 0.50, f"Entity: {entity_str}", transform=ax.transAxes, fontsize=10)

    # --- Product Progress ---
    limit = 8  # Limit to show 8 lines
    ax.text(stats_x, 0.45, "In-Progress Products", transform=ax.transAxes, fontsize=11, fontweight="bold")

    # Loop only through the first 'limit' items
    for k, (prod, done, total) in enumerate(product_progress[:limit]):
        ax.text(stats_x, 0.42 - 0.025 * k, f"{prod}: {done}/{total}", transform=ax.transAxes, fontsize=9)

    # Check if there are more products than the limit
    if len(product_progress) > limit:
        remaining = len(product_progress) - limit
        ax.text(stats_x, 0.42 - 0.025 * limit, f"+ {remaining} more", transform=ax.transAxes, fontsize=9,
                color="darkgreen")


def generate_flow_frames(
        config_path: str,
        sheet_parts_path: str,
        events_path: str,
        outdir: str,
        max_frames: int | None = None,
        zip_output: bool = False,
) -> str:
    """Generate flow frames using explicit input paths."""
    with open(config_path, "r") as f:
        config_data = json.load(f)

    stations = load_stations(config_data)
    sheet_station_name = next((s.name for s in stations if s.is_sheet), None)
    if not sheet_station_name:
        raise ValueError("No sheet station found in config.")

    sheet_parts_df = pd.read_csv(sheet_parts_path)
    events_df = pd.read_csv(events_path).sort_values(["time", "event_type"]).reset_index(drop=True)

    (
        part_to_sheet,
        part_to_product,
        prod_total,
        sheet_end,
        part_intervals,
        interval_completion_times,
        part_sheet,
        part_complete_timestamps,
        part_completion_times,
        part_to_area,
        sheet_to_area,
    ) = preprocess(events_df, sheet_parts_df)
    part_area_map = part_to_area
    sheet_area_map = sheet_to_area

    # Extract all 'part_complete' events once for efficiency
    all_complete_events = events_df[events_df.event_type == 'part_complete'][['time', 'product_id']]

    os.makedirs(outdir, exist_ok=True)

    unique_times = events_df['time'].sort_values().unique()

    if max_frames:
        process_times = unique_times[:max_frames].tolist()
    else:
        process_times = unique_times.tolist()

    # Add final frame
    final_time = unique_times[-1] + 1.0
    process_times.append(final_time)

    for idx, t in enumerate(process_times):
        current_events = events_df[events_df.time == t]

        sheet_waiting, sheet_active = compute_sheet_state(t, events_df, sheet_station_name)

        waiting_parts, active_parts = compute_frame_state(
            t, stations, part_sheet, sheet_end, part_intervals, interval_completion_times
        )

        product_progress = compute_product_progress(
            t, part_to_product, prod_total, part_completion_times
        )

        # 1. Total Completed Parts
        total_completed_parts = bisect.bisect_right(part_complete_timestamps, t)

        # 2. Total Completed Products
        finished_events_now = all_complete_events[all_complete_events.time <= t]
        counts_per_product = finished_events_now['product_id'].value_counts()

        total_completed_products = 0
        for pid, count in counts_per_product.items():
            if count >= prod_total.get(pid, 0):
                total_completed_products += 1

        fig = plt.figure(figsize=(17, 7), dpi=100)
        ax = fig.add_axes([0, 0, 1, 1])

        draw_frame(
            ax=ax,
            stations=stations,
            frame_idx=idx,
            current_time=t,
            current_events=current_events,
            waiting_parts=waiting_parts,
            active_parts=active_parts,
            total_completed_parts=total_completed_parts,
            total_completed_products=total_completed_products,
            sheet_waiting=sheet_waiting,
            sheet_active=sheet_active,
            product_progress=product_progress,
            part_area_map=part_area_map,
            sheet_area_map=sheet_area_map,
        )

        output_filename = os.path.join(outdir, f"frame_{idx:05d}.png")
        fig.savefig(output_filename)
        plt.close(fig)

    if zip_output:
        zip_path = f"{outdir}.zip"
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
            for filename in sorted(os.listdir(outdir)):
                if filename.lower().endswith(".png"):
                    z.write(os.path.join(outdir, filename), arcname=filename)
        print(f"ZIP written: {zip_path}")

    return outdir


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_config = os.path.join(script_dir, "config", "config.json")
    default_sheet_parts = os.path.join(script_dir, "sheet_parts.csv")
    default_events = os.path.join(script_dir, "event_summary.csv")
    default_outdir = os.path.join(script_dir, "flow_frames_v3")

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=default_config)
    parser.add_argument("--sheet_parts", default=default_sheet_parts)
    parser.add_argument("--events", default=default_events)
    parser.add_argument("--outdir", default=default_outdir)
    parser.add_argument("--max_frames", type=int, default=None, help="Limit frames for debugging. If unset, runs all.")
    parser.add_argument("--zip", action="store_true")
    args = parser.parse_args()

    generate_flow_frames(
        config_path=args.config,
        sheet_parts_path=args.sheet_parts,
        events_path=args.events,
        outdir=args.outdir,
        max_frames=args.max_frames,
        zip_output=args.zip,
    )


if __name__ == "__main__":
    main()
