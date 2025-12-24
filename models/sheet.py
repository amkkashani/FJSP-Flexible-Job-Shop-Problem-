"""Sheet model for FJSP."""

from dataclasses import dataclass, field
from typing import List, Optional
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
        assigned_parts: Parts assigned to this sheet
    """
    id: str
    capacity: float
    assigned_parts: List[Part] = field(default_factory=list)

    def total_area(self) -> float:
        """Sum of areas of all assigned parts."""
        return sum(part.area for part in self.assigned_parts)

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

    def can_fit(self, part: Part) -> bool:
        """Check if a part can fit in the remaining capacity."""
        return part.area <= self.remaining_capacity()

    def add_part(self, part: Part) -> bool:
        """
        Add a part to this sheet if it fits.

        Returns:
            True if part was added, False if it doesn't fit.
        """
        if self.can_fit(part):
            self.assigned_parts.append(part)
            return True
        return False

    def num_parts(self) -> int:
        """Return the number of parts in this sheet."""
        return len(self.assigned_parts)

    def get_part_ids(self) -> List[str]:
        """Return list of part IDs in this sheet."""
        return [part.id for part in self.assigned_parts]

    def __repr__(self) -> str:
        return (f"Sheet(id={self.id}, parts={self.num_parts()}, "
                f"used={self.total_area():.4f}/{self.capacity:.4f}m², "
                f"waste={self.waste():.4f}m²)")
