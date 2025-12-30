# Configuration Guide

This document explains all configuration options for the FJSP Solver.

---

## Configuration File

All configuration is stored in `config/config.json`.

### Complete Example

```json
{
  "data_file": "data/5693_cleaned.xlsx",
  "stations": [
    {"name": "wa", "order_index": 0, "num_machines": 2},
    {"name": "wf", "order_index": 1, "num_machines": 2},
    {"name": "wd", "order_index": 2, "num_machines": 1},
    {"name": "wo", "order_index": 3, "num_machines": 1},
    {"name": "wg", "order_index": 4, "num_machines": 2},
    {"name": "wv", "order_index": 5, "num_machines": 1},
    {"name": "wx", "order_index": 6, "num_machines": 1}
  ],
  "sheet_capacity": 3.6,
  "sheet_X": 2,
  "sheet_Y": 1.8,
  "evaluator": {
    "alpha": 1.0,
    "beta": 0.5,
    "gamma": 0.3
  },
  "report": {
    "generate_gantt_chart": true,
    "generate_animation": true
  },
  "animation_settings": {
    "fps": 15,
    "duration_seconds": 30,
    "end_hold_seconds": 5,
    "max_sheets": null
  }
}
```

---

## Configuration Parameters

### data_file

**Type:** string
**Required:** Yes
**Example:** `"data/5693_cleaned.xlsx"`

Path to the input data file (Excel or CSV). The path is relative to the project root directory.

```json
"data_file": "data/my_production_data.xlsx"
```

---

### stations

**Type:** array of objects
**Required:** Yes

Defines the processing stations in the manufacturing system.

Each station object has:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | Yes | Station code (must match column names in data file) |
| `order_index` | int | Yes | Processing sequence position (0 = first) |
| `num_machines` | int | No | Number of parallel machines (default: 1) |

#### Station Order

Sheets must visit stations in the order defined by `order_index`:

```
wa (0) → wf (1) → wd (2) → wo (3) → wg (4) → wv (5) → wx (6)
```

#### Parallel Machines

If a station has `num_machines > 1`, multiple sheets can be processed simultaneously at that station.

```json
{"name": "wa", "order_index": 0, "num_machines": 2}  // 2 parallel machines
{"name": "wd", "order_index": 2, "num_machines": 1}  // 1 machine (bottleneck)
```

---

### sheet_capacity

**Type:** float
**Required:** Yes
**Unit:** square meters (m²)
**Example:** `3.6`

Maximum area that can be packed into a single sheet.

```json
"sheet_capacity": 3.6
```

This value is used for bin packing. Parts are assigned to sheets such that:
```
sum(part.area for part in sheet) <= sheet_capacity
```

If `sheet_capacity` is larger than `sheet_X * sheet_Y`, the solver caps it to the sheet area.

---

### sheet_X

**Type:** float
**Required:** No (used for packing and visualization)
**Unit:** meters
**Default:** `2.0`
**Example:** `2`

Sheet width in meters. Used for:
- 2D placement inside the sheet
- Sheet image generation

```json
"sheet_X": 2
```

---

### sheet_Y

**Type:** float
**Required:** No (used for packing and visualization)
**Unit:** meters
**Default:** `1.8`
**Example:** `1.8`

Sheet height in meters. Used for:
- 2D placement inside the sheet
- Sheet image generation

```json
"sheet_Y": 1.8
```

**Note:** `sheet_X * sheet_Y` should be at least `sheet_capacity`. If not, the solver caps capacity to the sheet area.

---

### evaluator

**Type:** object
**Required:** No

Defines weights for the multi-objective evaluator.

```json
"evaluator": {
  "alpha": 1.0,
  "beta": 0.5,
  "gamma": 0.3
}
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `alpha` | float | 1.0 | Weight for waste minimization |
| `beta` | float | 0.5 | Weight for makespan minimization |
| `gamma` | float | 0.3 | Weight for average product completion time |

**Higher values prioritize:**
- `alpha` → Less waste (better sheet utilization)
- `beta` → Faster overall completion (lower makespan)
- `gamma` → Faster individual product completion

---

### report

**Type:** object
**Required:** No

Controls which output reports and visualizations are generated.

```json
"report": {
  "generate_gantt_chart": true,
  "generate_animation": false
}
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `generate_gantt_chart` | boolean | true | Generate Gantt chart PNG showing schedule |
| `generate_animation` | boolean | true | Generate animated GIF of sheet flow |

**Performance tip:** Set `generate_animation` to `false` to significantly speed up execution time.

---

### animation_settings

**Type:** object
**Required:** No

Configures animation generation parameters. These settings affect runtime performance.

```json
"animation_settings": {
  "fps": 15,
  "duration_seconds": 30,
  "end_hold_seconds": 5,
  "max_sheets": null
}
```

| Parameter | Type | Default | Description | Runtime Impact |
|-----------|------|---------|-------------|----------------|
| `fps` | int | 15 | Frames per second | Higher = longer runtime |
| `duration_seconds` | int | 30 | Total animation duration | Higher = longer runtime |
| `end_hold_seconds` | int | 5 | Hold on final frame | Minor impact |
| `max_sheets` | int or null | null | Maximum sheets to animate (null = all sheets) | Higher = longer runtime |

**Performance recommendations:**
- For faster generation: `fps=10`, `duration_seconds=20`, `max_sheets=30`
- For high quality: `fps=20`, `duration_seconds=40`, `max_sheets=null` (all sheets)
- For quick testing: Set `generate_animation=false` in report section

**Note about max_sheets:**
- Set to `null` to animate all sheets in the solution (recommended for final reports)
- Set to a number (e.g., `30` or `50`) to limit sheets for faster generation during development

**Note:** Total frames = `fps × (duration_seconds + end_hold_seconds)`. More frames = longer generation time.

---

## Input Data File Requirements

The input Excel/CSV file specified in `data_file` must have these columns:

### Required Columns

| Column | Type | Description |
|--------|------|-------------|
| `ElemIdent` | string | Element identifier |
| `length` | float | Part length in mm |
| `width` | float | Part width in mm |
| `area` | float | Part area in m² |
| `mat` | string | Material code |
| `quantity` | int | Number of parts |
| `Info8` | string | Product ID |

### Station Columns

One column for each station defined in the config:

| Column | Type | Description |
|--------|------|-------------|
| `wa` | float | Processing time at station wa |
| `wf` | float | Processing time at station wf |
| `wd` | float | Processing time at station wd |
| `wo` | float | Processing time at station wo |
| `wg` | float | Processing time at station wg |
| `wv` | float | Processing time at station wv |
| `wx` | float | Processing time at station wx |

### Optional Columns

| Column | Type | Description |
|--------|------|-------------|
| `row_id` | int/string | Custom row identifier |

---

## Part ID Generation

Part IDs are generated based on the `row_id` column or row position:

1. If `row_id` column exists and has a value:
   - Numeric values: `row_00001`, `row_00002`, etc.
   - String values: Used as-is (or formatted if numeric string)

2. If `row_id` is missing:
   - Generated from row position: `row_00001`, `row_00002`, etc.

3. If `quantity > 1`:
   - Parts are expanded with suffix: `row_00001_001`, `row_00001_002`, etc.

---

## Quantity Expansion

When a row has `quantity > 1`:

1. The row is expanded into multiple individual parts
2. The `area` is divided by `quantity` for each part
3. Each part gets a unique ID with a suffix

**Example:**
```
Original: ElemIdent=E001, area=1.5, quantity=3
Expanded:
  - row_00001_001 (area=0.5)
  - row_00001_002 (area=0.5)
  - row_00001_003 (area=0.5)
```

---

## Solver Configuration

Solver settings are configured in `main.py`:

### Evaluator Weights

```python
evaluator = WeightedEvaluator(alpha=1.0, beta=0.5)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `alpha` | 1.0 | Weight for waste minimization |
| `beta` | 0.5 | Weight for makespan minimization |

Higher `alpha` → prioritize less waste
Higher `beta` → prioritize faster completion

### Solver Strategy

```python
solver = GreedySolver(sort_by='area_desc')
```

| Strategy | Description | Best For |
|----------|-------------|----------|
| `'area_desc'` | Largest parts first (FFD) | Minimal waste |
| `'area_asc'` | Smallest parts first | - |
| `'product'` | Group by product | Product completion |
| `'none'` | Original order | Testing |

---

## Example Configurations

### Minimal Configuration

```json
{
  "data_file": "data/parts.xlsx",
  "stations": [
    {"name": "wa", "order_index": 0, "num_machines": 1}
  ],
  "sheet_capacity": 5.0
}
```

### High-Throughput Configuration

```json
{
  "data_file": "data/production.xlsx",
  "stations": [
    {"name": "wa", "order_index": 0, "num_machines": 4},
    {"name": "wf", "order_index": 1, "num_machines": 4},
    {"name": "wd", "order_index": 2, "num_machines": 2},
    {"name": "wo", "order_index": 3, "num_machines": 2},
    {"name": "wg", "order_index": 4, "num_machines": 4},
    {"name": "wv", "order_index": 5, "num_machines": 2},
    {"name": "wx", "order_index": 6, "num_machines": 2}
  ],
  "sheet_capacity": 3.6,
  "sheet_X": 2,
  "sheet_Y": 1.8
}
```

### Custom Sheet Size

```json
{
  "data_file": "data/large_parts.xlsx",
  "stations": [
    {"name": "wa", "order_index": 0, "num_machines": 2}
  ],
  "sheet_capacity": 6.0,
  "sheet_X": 3,
  "sheet_Y": 2
}
```

---

## Troubleshooting

### "Data file not found"

Check that `data_file` path is correct and relative to project root.

### "Column not found" errors

Ensure your Excel file has all required columns with exact names (case-sensitive).

### Station processing times are 0

Check that station column names in Excel match the `name` values in config.

### Parts not fitting in sheets

- Check `sheet_capacity` is large enough
- Ensure `sheet_X` and `sheet_Y` can hold the part length/width (rotation allowed)
- Verify `area` values in Excel are in m² (not mm²)
- Ensure `area / quantity` for each row is less than `sheet_capacity`
