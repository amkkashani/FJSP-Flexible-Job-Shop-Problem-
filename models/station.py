"""Station model for FJSP."""

from dataclasses import dataclass


@dataclass
class Station:
    """
    A processing station with one or more parallel machines.

    Attributes:
        name: Station code: "wa", "wf", "wd", "wo", "wg", "wv", "wx"
        order_index: Position in sequence: wa=0, wf=1, wd=2, wo=3, wg=4, wv=5, wx=6
        num_machines: Number of parallel machines (M). If M=2, two sheets can process simultaneously
    """
    name: str
    order_index: int
    num_machines: int = 1

    def __repr__(self) -> str:
        return f"Station(name={self.name}, order={self.order_index}, machines={self.num_machines})"

    def __lt__(self, other: 'Station') -> bool:
        """Allow sorting stations by order_index."""
        return self.order_index < other.order_index
