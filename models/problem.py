"""Problem model for FJSP."""

import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from pathlib import Path

import pandas as pd

from .part import Part
from .product import Product
from .station import Station


@dataclass
class Problem:
    """
    Encapsulates the entire problem instance.
    This is the input to any solver.

    Attributes:
        parts: All parts (expanded by quantity)
        products: Products indexed by product_id
        stations: Ordered list of stations
        sheet_capacity: Maximum area per sheet in m²
        sheet_width: Default sheet width in meters
        sheet_height: Default sheet height in meters
        material_sheet_sizes: Material-specific sheet sizes {material: {"width": w, "height": h}}
    """
    parts: List[Part] = field(default_factory=list)
    products: Dict[str, Product] = field(default_factory=dict)
    stations: List[Station] = field(default_factory=list)
    sheet_capacity: float = 5.0
    sheet_width: float = 2.0
    sheet_height: float = 1.8
    material_sheet_sizes: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def get_parts_by_product(self, product_id: str) -> List[Part]:
        """Get all parts for a product."""
        return [part for part in self.parts if part.product_id == product_id]

    def get_parts_by_material(self, material: str) -> List[Part]:
        """Get all parts with specific material."""
        return [part for part in self.parts if part.material == material]

    def get_station(self, name: str) -> Optional[Station]:
        """Get station by name."""
        for station in self.stations:
            if station.name == name:
                return station
        return None

    def get_station_names(self) -> List[str]:
        """Get ordered list of station names."""
        return [station.name for station in sorted(self.stations)]

    def total_parts_area(self) -> float:
        """Get total area of all parts."""
        return sum(part.area for part in self.parts)

    def num_parts(self) -> int:
        """Get total number of parts."""
        return len(self.parts)

    def num_products(self) -> int:
        """Get total number of products."""
        return len(self.products)

    def get_unique_materials(self) -> List[str]:
        """Get list of unique materials."""
        return list(set(part.material for part in self.parts))

    def get_sheet_size_for_material(self, material: str) -> tuple:
        """
        Get sheet size (width, height, capacity) for a specific material.
        Falls back to default sheet size if material not found in config.

        Args:
            material: Material code

        Returns:
            Tuple of (width, height, capacity)
        """
        if material in self.material_sheet_sizes:
            size = self.material_sheet_sizes[material]
            width = float(size.get("width", self.sheet_width))
            height = float(size.get("height", self.sheet_height))
            capacity = min(self.sheet_capacity, width * height)
            return width, height, capacity
        return self.sheet_width, self.sheet_height, self.sheet_capacity

    @classmethod
    def load_from_dataframe(
        cls,
        df: pd.DataFrame,
        station_config: dict,
        sheet_sizes: Optional[Dict[str, Dict[str, float]]] = None
    ) -> 'Problem':
        """
        Create Problem from pandas DataFrame and station config dict.

        Args:
            df: DataFrame with columns: ElemIdent, length, width, area, mat, quantity, Info8,
                and station columns (wa, wf, wd, wo, wg, wv, wx)
            station_config: Dict with 'stations' list, 'sheet_capacity', 'sheet_X', 'sheet_Y'
            sheet_sizes: Optional dict of material-specific sheet sizes from sheet_sizes.json

        Returns:
            Problem instance
        """
        # 1. Create stations
        stations = [
            Station(
                name=s["name"],
                order_index=s["order_index"],
                num_machines=s.get("num_machines", 1),
                workers_per_machine=max(1, int(s.get("workerPerMachine", 1))),
                sheet=s.get("sheet", True)
            )
            for s in station_config["stations"]
        ]
        stations.sort(key=lambda s: s.order_index)

        # Get station names for process times
        station_names = [s.name for s in stations]
        station_lookup = {name.upper(): name for name in station_names}

        def extract_sequence(code_value: object) -> List[str]:
            if code_value is None or pd.isna(code_value):
                return []
            code_str = str(code_value)
            slash_idx = code_str.find("/")
            if slash_idx == -1:
                return []
            backslash_idx = code_str.find("\\", slash_idx + 1)
            if backslash_idx == -1:
                return []
            seq_raw = code_str[slash_idx + 1:backslash_idx]
            if not seq_raw:
                return []
            tokens = [t for t in re.split(r"[^A-Za-z]+", seq_raw) if t]
            return [station_lookup[t.upper()] for t in tokens if t.upper() in station_lookup]

        # 2. Expand parts by quantity
        parts = []

        for row_pos, (_, row) in enumerate(df.iterrows(), start=1):
            quantity = int(row.get("quantity", 1))
            # Area per individual part (divide total area by quantity)
            area_per_part = float(row["area"]) / quantity if quantity > 0 else float(row["area"])
            row_id_val = row.get("row_id", None)
            row_id = None
            if row_id_val is not None and pd.notna(row_id_val):
                if isinstance(row_id_val, (int, float)) and not isinstance(row_id_val, bool):
                    if float(row_id_val).is_integer():
                        row_id = f"row_{int(row_id_val):05d}"
                elif isinstance(row_id_val, str):
                    row_id = row_id_val.strip()
                    if row_id.isdigit():
                        row_id = f"row_{int(row_id):05d}"
            if not row_id:
                row_id = f"row_{row_pos:05d}"

            for i in range(quantity):
                # Build process times dict
                process_times = {}
                for station_name in station_names:
                    if station_name in row:
                        process_times[station_name] = (float(row[station_name]) / quantity) if pd.notna(row[station_name]) else 0.0
                    else:
                        process_times[station_name] = 0.0

                part_id = row_id if quantity == 1 else f"{row_id}_{i + 1:03d}"
                part_sequence = extract_sequence(row.get("code", ""))
                if not part_sequence:
                    part_sequence = list(station_names)

                part = Part(
                    id=part_id,
                    elem_ident=str(row.get("ElemIdent", "")),
                    length=float(row.get("length", 0)),
                    width=float(row.get("width", 0)),
                    area=area_per_part,
                    material=str(row.get("mat", "")),
                    product_id=str(row.get("Info8", "")),
                    process_times=process_times,
                    sequence=part_sequence
                )
                parts.append(part)

        # 3. Create products
        products = {}
        for part in parts:
            if part.product_id not in products:
                products[part.product_id] = Product(id=part.product_id, part_ids=[])
            products[part.product_id].add_part(part.id)

        # 4. Create and return problem
        sheet_width = float(station_config.get("sheet_X", 2.0))
        sheet_height = float(station_config.get("sheet_Y", 1.8))
        raw_capacity = float(station_config.get("sheet_capacity", sheet_width * sheet_height))
        sheet_capacity = min(raw_capacity, sheet_width * sheet_height)

        # 5. Load material-specific sheet sizes
        material_sheet_sizes = {}
        if sheet_sizes:
            for material, size in sheet_sizes.items():
                if material != "_comment" and isinstance(size, dict):
                    material_sheet_sizes[material] = size

        return cls(
            parts=parts,
            products=products,
            stations=stations,
            sheet_capacity=sheet_capacity,
            sheet_width=sheet_width,
            sheet_height=sheet_height,
            material_sheet_sizes=material_sheet_sizes
        )

    @classmethod
    def load_from_files(cls, excel_path: str, config_path: str) -> 'Problem':
        """
        Load problem from Excel file and JSON config file.

        Args:
            excel_path: Path to Excel file with parts data
            config_path: Path to JSON config file with station configuration

        Returns:
            Problem instance
        """
        # Load Excel data
        df = pd.read_excel(excel_path)

        # Load station config
        with open(config_path, 'r') as f:
            station_config = json.load(f)

        # Load sheet sizes from separate file if specified
        sheet_sizes = None
        sheet_sizes_file = station_config.get("sheet_sizes_file")
        if sheet_sizes_file:
            config_dir = Path(config_path).parent.parent
            sheet_sizes_path = config_dir / sheet_sizes_file
            if sheet_sizes_path.exists():
                with open(sheet_sizes_path, 'r') as f:
                    sheet_sizes = json.load(f)

        return cls.load_from_dataframe(df, station_config, sheet_sizes)

    def __repr__(self) -> str:
        return (f"Problem(parts={self.num_parts()}, products={self.num_products()}, "
                f"stations={len(self.stations)}, sheet_capacity={self.sheet_capacity}m², "
                f"sheet_size={self.sheet_width}x{self.sheet_height}m)")
