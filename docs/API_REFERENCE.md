# API Reference

Complete API documentation for the FJSP Solver.

---

## Models Package

### Part

```python
from models import Part
```

A single physical piece to be manufactured.

#### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `id` | str | Unique identifier (e.g., "row_00001" or "row_00001_001") |
| `elem_ident` | str | Original ElemIdent from input data |
| `length` | float | Part length in millimeters |
| `width` | float | Part width in millimeters |
| `area` | float | Part area in square meters |
| `material` | str | Material code |
| `product_id` | str | Product ID this part belongs to (Info8) |
| `process_times` | Dict[str, float] | Processing time at each station |

#### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `get_process_time(station_name)` | float | Get processing time for a specific station |

#### Example

```python
part = Part(
    id="row_00001",
    elem_ident="E001",
    length=1000.0,
    width=500.0,
    area=0.5,
    material="MAT_A",
    product_id="PROD_001",
    process_times={"wa": 16, "wf": 13, "wd": 0, "wo": 10, "wg": 7, "wv": 0, "wx": 4}
)

print(part.get_process_time("wa"))  # 16.0
```

---

### Product

```python
from models import Product
```

A complete product composed of multiple parts.

#### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `id` | str | Product ID (Info8 value) |
| `part_ids` | List[str] | List of part IDs belonging to this product |

#### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `add_part(part_id)` | None | Add a part ID to this product |
| `num_parts()` | int | Return the number of parts |

#### Example

```python
product = Product(id="PROD_001", part_ids=["row_00001", "row_00002"])
product.add_part("row_00003")
print(product.num_parts())  # 3
```

---

### Station

```python
from models import Station
```

A processing station with one or more parallel machines.

#### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `name` | str | Station code (wa, wf, wd, wo, wg, wv, wx) |
| `order_index` | int | Position in processing sequence (0-6) |
| `num_machines` | int | Number of parallel machines (default: 1) |

#### Example

```python
station = Station(name="wa", order_index=0, num_machines=2)
```

---

### Sheet

```python
from models import Sheet
```

A container (bin) that holds multiple parts based on area and size.

#### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `id` | str | Unique identifier (e.g., "sheet_00001") |
| `capacity` | float | Maximum area in m² |
| `width` | float | Sheet width in meters |
| `height` | float | Sheet height in meters |
| `assigned_parts` | List[Part] | Parts assigned to this sheet |
| `placements` | Dict[str, tuple] | Part placement (x, y, w, h, rotated) |

#### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `total_area()` | float | Sum of areas of all assigned parts |
| `remaining_capacity()` | float | Available space (capacity - total_area) |
| `waste()` | float | Unused area (same as remaining_capacity) |
| `get_station_time(station_name)` | float | Sum of process times for all parts at station |
| `is_empty()` | bool | True if no parts assigned |
| `can_fit(part, allow_rotate=True)` | bool | Check if part can fit in remaining capacity and sheet size |
| `add_part(part, allow_rotate=True)` | bool | Add part if it fits, returns success status |
| `num_parts()` | int | Number of parts in sheet |
| `get_part_ids()` | List[str] | List of part IDs in sheet |

#### Example

```python
sheet = Sheet(id="sheet_00001", capacity=3.6, width=2.0, height=1.8)

part1 = Part(id="p1", ..., area=1.5)
part2 = Part(id="p2", ..., area=1.0)

sheet.add_part(part1)  # True
sheet.add_part(part2)  # True

print(sheet.total_area())        # 2.5
print(sheet.remaining_capacity()) # 1.1
print(sheet.waste())             # 1.1
print(sheet.can_fit(Part(..., area=0.5)))  # True
```

---

### Problem

```python
from models import Problem
```

Encapsulates the entire problem instance.

#### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `parts` | List[Part] | All parts (expanded by quantity) |
| `products` | Dict[str, Product] | Products indexed by product_id |
| `stations` | List[Station] | Ordered list of stations |
| `sheet_capacity` | float | Maximum area per sheet in m² |
| `sheet_width` | float | Sheet width in meters |
| `sheet_height` | float | Sheet height in meters |

#### Class Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `load_from_files(excel_path, config_path)` | Problem | Load from Excel and JSON config |
| `load_from_dataframe(df, station_config)` | Problem | Load from DataFrame and config dict |

#### Instance Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `get_parts_by_product(product_id)` | List[Part] | Get all parts for a product |
| `get_parts_by_material(material)` | List[Part] | Get all parts with specific material |
| `get_station(name)` | Station | Get station by name |
| `get_station_names()` | List[str] | Get ordered list of station names |
| `total_parts_area()` | float | Get total area of all parts |
| `num_parts()` | int | Get total number of parts |
| `num_products()` | int | Get total number of products |
| `get_unique_materials()` | List[str] | Get list of unique materials |

#### Example

```python
# Load from files
problem = Problem.load_from_files(
    excel_path="data/5693_cleaned.xlsx",
    config_path="config/config.json"
)

print(f"Parts: {problem.num_parts()}")
print(f"Products: {problem.num_products()}")
print(f"Total area: {problem.total_parts_area():.2f} m²")

# Get parts for a product
product_parts = problem.get_parts_by_product("PROD_001")
```

---

## Solution Package

### SheetAssignment

```python
from solution import SheetAssignment
```

Represents the timing of a sheet at a specific station.

#### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `station_name` | str | Station name |
| `machine_index` | int | Which machine (0 to num_machines-1) |
| `start_time` | float | When processing starts |
| `end_time` | float | When processing ends |

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| `duration` | float | Processing duration (end_time - start_time) |

---

### Solution

```python
from solution import Solution
```

Complete solution with sheet assignments and schedule.

#### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `sheets` | List[Sheet] | All sheets with assigned parts |
| `schedule` | Dict[str, List[SheetAssignment]] | Station assignments per sheet |
| `metrics` | Dict[str, float] | Computed metrics |

#### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `get_makespan()` | float | Time when last sheet finishes |
| `get_total_waste()` | float | Sum of waste across all sheets |
| `get_total_used_area()` | float | Sum of used area across all sheets |
| `get_product_completion_time(product_id, problem)` | float | When all parts of a product finish |
| `get_all_product_completion_times(problem)` | Dict[str, float] | Completion times for all products |
| `get_avg_product_completion_time(problem)` | float | Average completion time |
| `num_sheets()` | int | Total number of sheets used |
| `is_valid(problem)` | bool | Check if solution satisfies all constraints |
| `get_sheet_by_id(sheet_id)` | Sheet | Get a sheet by its ID |
| `add_sheet(sheet)` | None | Add a sheet to the solution |
| `add_assignment(sheet_id, assignment)` | None | Add a station assignment |
| `compute_metrics(problem)` | Dict[str, float] | Compute and store all metrics |
| `summary(problem)` | str | Generate summary string |

#### Example

```python
# After solving
solution = solver.solve(problem, evaluator)

print(f"Sheets: {solution.num_sheets()}")
print(f"Makespan: {solution.get_makespan():.2f}")
print(f"Waste: {solution.get_total_waste():.4f} m²")
print(f"Valid: {solution.is_valid(problem)}")

# Get product completion times
times = solution.get_all_product_completion_times(problem)
for product_id, time in times.items():
    print(f"{product_id}: {time:.2f}")
```

---

## Evaluation Package

### Evaluator (Abstract Base Class)

```python
from evaluation import Evaluator
```

Abstract base class for solution evaluators.

#### Abstract Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `evaluate(solution, problem)` | float | Compute fitness value (lower is better) |

---

### WeightedEvaluator

```python
from evaluation import WeightedEvaluator
```

Evaluator using weighted sum of waste and makespan.

```
fitness = alpha * total_waste + beta * makespan
```

#### Constructor

```python
WeightedEvaluator(alpha: float = 1.0, beta: float = 1.0)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `alpha` | float | 1.0 | Weight for total waste |
| `beta` | float | 1.0 | Weight for makespan |

#### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `evaluate(solution, problem)` | float | Compute weighted fitness |
| `get_components(solution, problem)` | dict | Get individual evaluation components |

#### Example

```python
evaluator = WeightedEvaluator(alpha=1.0, beta=0.5)

fitness = evaluator.evaluate(solution, problem)

components = evaluator.get_components(solution, problem)
# {
#     'total_waste': 2.74,
#     'makespan': 18821.0,
#     'weighted_waste': 2.74,
#     'weighted_makespan': 9410.5,
#     'fitness': 9413.24
# }
```

---

## Solvers Package

### Solver (Abstract Base Class)

```python
from solvers import Solver
```

Abstract base class for scheduling algorithms.

#### Abstract Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `solve(problem, evaluator)` | Solution | Run algorithm and return solution |

---

### GreedySolver

```python
from solvers import GreedySolver
```

Greedy solver using First-Fit Decreasing bin packing and FIFO scheduling.

#### Constructor

```python
GreedySolver(sort_by: str = 'area_desc')
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sort_by` | str | 'area_desc' | Sorting strategy for parts |

#### Sorting Strategies

| Strategy | Description |
|----------|-------------|
| `'area_desc'` | Sort by area descending (FFD - recommended) |
| `'area_asc'` | Sort by area ascending |
| `'product'` | Group by product, then by area descending |
| `'none'` | No sorting (original order) |

#### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `solve(problem, evaluator)` | Solution | Solve using greedy approach |

#### Example

```python
solver = GreedySolver(sort_by='area_desc')
solution = solver.solve(problem, evaluator)
```

---

## Output Generator

```python
from output_generator import generate_outputs, OutputGenerator
```

### generate_outputs Function

```python
def generate_outputs(
    solution: Solution,
    problem: Problem,
    data_file: str,
    sheet_width: float = 2.0,
    sheet_height: float = 1.8
) -> Path
```

Generate all output files and images.

| Parameter | Type | Description |
|-----------|------|-------------|
| `solution` | Solution | The solved solution |
| `problem` | Problem | The problem instance |
| `data_file` | str | Path to input data file (for folder naming) |
| `sheet_width` | float | Sheet width in meters |
| `sheet_height` | float | Sheet height in meters |

**Returns**: Path to the output folder

### OutputGenerator Class

For more control over output generation:

```python
generator = OutputGenerator(solution, problem, data_file)

# Individual exports
generator.create_output_folder()
generator.export_sheet_parts_csv()
generator.export_sheet_summary_csv()
generator.export_product_summary_csv()
generator.export_schedule_csv()
generator.export_solution_summary()
generator.generate_all_sheet_images(sheet_width, sheet_height)

# Or all at once
generator.generate_all_outputs(sheet_width, sheet_height)
```

---

## Complete Usage Example

```python
import json
from models import Problem
from solvers import GreedySolver
from evaluation import WeightedEvaluator
from output_generator import generate_outputs

# Load configuration
with open("config/config.json", 'r') as f:
    config = json.load(f)

data_file = config["data_file"]
sheet_x = config.get("sheet_X", 2.0)
sheet_y = config.get("sheet_Y", 1.8)

# Load problem
problem = Problem.load_from_files(data_file, "config/config.json")

# Create evaluator and solver
evaluator = WeightedEvaluator(alpha=1.0, beta=0.5)
solver = GreedySolver(sort_by='area_desc')

# Solve
solution = solver.solve(problem, evaluator)

# Evaluate
fitness = evaluator.evaluate(solution, problem)
print(f"Fitness: {fitness:.4f}")

# Validate
if solution.is_valid(problem):
    print("Solution is valid!")

# Generate outputs
output_folder = generate_outputs(solution, problem, data_file, sheet_x, sheet_y)
print(f"Outputs saved to: {output_folder}")
```
