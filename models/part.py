"""Part model for FJSP."""

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class Part:
    """
    A single physical piece to be manufactured.

    Attributes:
        id: Unique identifier (generated, e.g., "part_001")
        elem_ident: Original ElemIdent from data
        length: Length in mm
        width: Width in mm
        area: Area in m²
        material: Material code (e.g., "NC007 M/AM114")
        product_id: Info8 value - which product this part belongs to
        process_times: Processing time at each station: {"wa": 16, "wf": 13, ...}
    """
    id: str
    elem_ident: str
    length: float
    width: float
    area: float
    material: str
    product_id: str
    process_times: Dict[str, float] = field(default_factory=dict)

    def get_process_time(self, station_name: str) -> float:
        """Get processing time for a specific station."""
        return self.process_times.get(station_name, 0.0)

    def __repr__(self) -> str:
        return f"Part(id={self.id}, product={self.product_id}, area={self.area:.4f}m²)"
