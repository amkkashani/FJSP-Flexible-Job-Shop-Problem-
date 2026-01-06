"""PartAssignment model for FJSP - tracks individual parts after sheet separation."""

from dataclasses import dataclass


@dataclass
class PartAssignment:
    """
    Represents the timing of a part at a specific station (after sheet separation).

    Attributes:
        part_id: The part identifier
        station_name: Station name
        machine_index: Which machine (0 to M-1)
        start_time: When processing starts
        end_time: When processing ends
    """
    part_id: str
    station_name: str
    machine_index: int
    start_time: float
    end_time: float

    @property
    def duration(self) -> float:
        """Processing duration at this station."""
        return self.end_time - self.start_time

    def __repr__(self) -> str:
        return (f"PartAssignment(part={self.part_id}, station={self.station_name}, "
                f"machine={self.machine_index}, time=[{self.start_time:.2f}, {self.end_time:.2f}])")
