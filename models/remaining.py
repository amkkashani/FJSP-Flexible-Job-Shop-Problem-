"""Remaining section model for FJSP."""

from dataclasses import dataclass
from typing import Dict, List, Optional
from datetime import datetime


@dataclass
class RemainingSection:
    """
    Represents a remaining (leftover) section from a previously used sheet.
    These sections can be reused in future runs before creating new sheets.

    Attributes:
        id: Unique identifier for this remaining section
        material: Material code (must match part material)
        width: Width in meters
        height: Height in meters
        area: Area in m² (width * height)
        original_sheet_id: ID of the sheet this section came from
        created_at: Timestamp when this section was created
    """
    id: str
    material: str
    width: float
    height: float
    area: float
    original_sheet_id: Optional[str] = None
    created_at: Optional[str] = None

    def __post_init__(self):
        if self.area == 0:
            self.area = self.width * self.height

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "material": self.material,
            "width": self.width,
            "height": self.height,
            "area": self.area,
            "original_sheet_id": self.original_sheet_id,
            "created_at": self.created_at
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'RemainingSection':
        """Create RemainingSection from dictionary."""
        return cls(
            id=data["id"],
            material=data["material"],
            width=float(data["width"]),
            height=float(data["height"]),
            area=float(data.get("area", 0)),
            original_sheet_id=data.get("original_sheet_id"),
            created_at=data.get("created_at")
        )

    def can_fit_part(self, part_width: float, part_height: float, material: str) -> bool:
        """
        Check if a part can fit in this remaining section.

        Args:
            part_width: Part width in meters
            part_height: Part height in meters
            material: Part material code

        Returns:
            True if part can fit (with or without rotation)
        """
        if material != self.material:
            return False

        # Check normal orientation
        if part_width <= self.width and part_height <= self.height:
            return True

        # Check rotated orientation
        if part_height <= self.width and part_width <= self.height:
            return True

        return False

    def __repr__(self) -> str:
        return (f"RemainingSection(id={self.id}, material={self.material}, "
                f"size={self.width}x{self.height}m, area={self.area:.4f}m²)")
