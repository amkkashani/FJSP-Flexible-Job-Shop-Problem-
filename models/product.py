"""Product model for FJSP."""

from dataclasses import dataclass, field
from typing import List


@dataclass
class Product:
    """
    A complete product composed of multiple parts.
    A product is only complete when all its parts have finished processing.

    Attributes:
        id: Info8 value (e.g., "ABH24558-11")
        part_ids: List of part IDs belonging to this product
    """
    id: str
    part_ids: List[str] = field(default_factory=list)

    def add_part(self, part_id: str) -> None:
        """Add a part ID to this product."""
        if part_id not in self.part_ids:
            self.part_ids.append(part_id)

    def num_parts(self) -> int:
        """Return the number of parts in this product."""
        return len(self.part_ids)

    def __repr__(self) -> str:
        return f"Product(id={self.id}, num_parts={self.num_parts()})"
