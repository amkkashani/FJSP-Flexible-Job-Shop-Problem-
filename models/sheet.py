"""Sheet model for FJSP."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from .part import Part


@dataclass
class Sheet:
    """
    A container (bin) that holds multiple parts.
    Parts are assigned to sheets based on area capacity.
    The sheet is the unit that moves through stations.

    Attributes:
        id: Unique identifier (e.g., "sheet_001")
        capacity: Maximum area in m²
        width: Sheet width in meters
        height: Sheet height in meters
        material: Material code for all parts in this sheet
        assigned_parts: Parts assigned to this sheet
    """
    id: str
    capacity: float
    width: float
    height: float
    material: Optional[str] = None
    assigned_parts: List[Part] = field(default_factory=list)
    placements: Dict[str, Tuple[float, float, float, float, bool]] = field(default_factory=dict)
    _shelves: List[Dict[str, float]] = field(default_factory=list, repr=False)

    def total_area(self) -> float:
        """Sum of areas of all assigned parts."""
        return sum(part.area for part in self.assigned_parts)

    def get_material(self) -> str:
        """Return the sheet material (single value or joined list if mixed)."""
        if self.material is not None:
            return self.material
        materials = {part.material for part in self.assigned_parts}
        if not materials:
            return ""
        if len(materials) == 1:
            return next(iter(materials))
        return "; ".join(sorted(materials))

    def remaining_capacity(self) -> float:
        """Remaining capacity = capacity - total_area()."""
        return self.capacity - self.total_area()

    def waste(self) -> float:
        """Unused area (same as remaining_capacity)."""
        return self.remaining_capacity()

    def get_station_time(self, station_name: str) -> float:
        """
        Sum of process times of all parts for the specified station.

        If all parts have 0 process time at a station, returns 0 (sheet skips station).
        """
        return sum(part.get_process_time(station_name) for part in self.assigned_parts)

    def is_empty(self) -> bool:
        """True if no parts assigned."""
        return len(self.assigned_parts) == 0

    def can_fit(self, part: Part, allow_rotate: bool = True) -> bool:
        """Check if a part can fit based on area and 2D placement."""
        if self.material is not None and part.material != self.material:
            return False
        if part.area > self.remaining_capacity():
            return False
        return self._find_placement(part, allow_rotate) is not None

    def _get_part_dims(self, part: Part, rotated: bool) -> Tuple[float, float]:
        width_m = part.width / 1000.0
        height_m = part.length / 1000.0
        if rotated:
            return height_m, width_m
        return width_m, height_m

    def _find_placement(
        self,
        part: Part,
        allow_rotate: bool
    ) -> Optional[Tuple[float, float, float, float, bool, Optional[int]]]:
        dims = [(self._get_part_dims(part, False), False)]
        if allow_rotate and part.length != part.width:
            dims.append((self._get_part_dims(part, True), True))

        dims = [
            (w, h, rotated)
            for (w, h), rotated in dims
            if w > 0 and h > 0 and w <= self.width and h <= self.height
        ]
        if not dims:
            return None

        best = None
        for shelf_index, shelf in enumerate(self._shelves):
            for w, h, rotated in dims:
                if h <= shelf["height"] and shelf["x"] + w <= self.width:
                    remaining = self.width - (shelf["x"] + w)
                    score = (remaining, shelf["height"] - h)
                    if best is None or score < best[0]:
                        best = (score, shelf_index, shelf["x"], shelf["y"], w, h, rotated)

        if best is not None:
            _, shelf_index, x, y, w, h, rotated = best
            return (x, y, w, h, rotated, shelf_index)

        new_y = self._shelves[-1]["y"] + self._shelves[-1]["height"] if self._shelves else 0.0
        candidates = []
        for w, h, rotated in dims:
            if new_y + h <= self.height:
                score = (h, self.width - w)
                candidates.append((score, w, h, rotated))

        if not candidates:
            return None

        _, w, h, rotated = min(candidates, key=lambda c: c[0])
        return (0.0, new_y, w, h, rotated, None)

    def add_part(self, part: Part, allow_rotate: bool = True) -> bool:
        """
        Add a part to this sheet if it fits (area + 2D placement).

        Returns:
            True if part was added, False if it doesn't fit.
        """
        if not self.can_fit(part):
            return False

        placement = self._find_placement(part, allow_rotate)
        if placement is None:
            return False

        if self.material is None:
            self.material = part.material

        x, y, w, h, rotated, shelf_index = placement
        if shelf_index is None:
            self._shelves.append({"y": y, "height": h, "x": w})
        else:
            self._shelves[shelf_index]["x"] += w

        self.assigned_parts.append(part)
        self.placements[part.id] = (x, y, w, h, rotated)
        return True

    def num_parts(self) -> int:
        """Return the number of parts in this sheet."""
        return len(self.assigned_parts)

    def get_part_ids(self) -> List[str]:
        """Return list of part IDs in this sheet."""
        return [part.id for part in self.assigned_parts]

    def __repr__(self) -> str:
        material = self.get_material()
        material_label = material if material else "unknown"
        return (f"Sheet(id={self.id}, parts={self.num_parts()}, "
                f"material={material_label}, "
                f"used={self.total_area():.4f}/{self.capacity:.4f}mAı, "
                f"waste={self.waste():.4f}mAı, "
                f"size={self.width}x{self.height}m)")
