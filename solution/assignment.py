"""SheetAssignment model for FJSP."""

from dataclasses import dataclass


@dataclass
class SheetAssignment:
    """
    Represents the timing of a sheet at a specific station.

    Attributes:
        station_name: Station name
        machine_index: Which machine (0 to M-1)
        start_time: When processing starts
        end_time: When processing ends
    """
    station_name: str
    machine_index: int
    start_time: float
    end_time: float

    @property
    def duration(self) -> float:
        """Processing duration at this station."""
        return self.end_time - self.start_time

    def __repr__(self) -> str:
        return (f"SheetAssignment(station={self.station_name}, machine={self.machine_index}, "
                f"time=[{self.start_time:.2f}, {self.end_time:.2f}])")
