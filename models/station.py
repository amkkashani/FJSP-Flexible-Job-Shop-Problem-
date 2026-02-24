"""Station model for FJSP."""

from dataclasses import dataclass, field
from typing import List

from .machine import Machine


@dataclass
class Station:
    """
    A processing station with one or more parallel machines.

    Attributes:
        name: Station code: "wa", "wf", "wd", "wo", "wg", "wv", "wx"
        order_index: Position in sequence: wa=0, wf=1, wd=2, wo=3, wg=4, wv=5, wx=6
        workers_per_machine: Number of workers assigned to each machine
        machines: Explicit machine list with name and speed coefficient
        sheet: If True, processes entire sheet as unit. If False, processes individual parts.
    """
    name: str
    order_index: int
    num_machines: int = 1
    workers_per_machine: int = 1
    machines: List[Machine] = field(default_factory=list)
    sheet: bool = True

    def __post_init__(self) -> None:
        self.num_machines = max(1, int(self.num_machines))
        self.workers_per_machine = max(1, int(self.workers_per_machine))

        normalized: List[Machine] = []
        for idx, machine in enumerate(self.machines, start=1):
            if isinstance(machine, Machine):
                normalized.append(machine)
                continue
            if isinstance(machine, dict):
                speed = machine.get("speed")
                if speed is None:
                    speed = machine.get("speed_coefficient")
                if speed is None:
                    speed = machine.get("speedCoefficient", 1.0)
                normalized.append(
                    Machine(
                        name=machine.get("name", f"{self.name}_m{idx}"),
                        speed_coefficient=speed
                    )
                )
                continue
            raise ValueError(
                f"Station '{self.name}' machine at index {idx - 1} must be Machine or dict."
            )

        if not normalized:
            normalized = [
                Machine(name=f"{self.name}_m{i + 1}", speed_coefficient=1.0)
                for i in range(self.num_machines)
            ]

        self.machines = normalized
        self.num_machines = len(self.machines)

    def get_machine_speed(self, machine_index: int) -> float:
        """Get speed multiplier for a machine index (fallback: 1.0)."""
        if 0 <= machine_index < len(self.machines):
            return self.machines[machine_index].speed_coefficient
        return 1.0

    def get_machine_name(self, machine_index: int) -> str:
        """Get machine name for a machine index."""
        if 0 <= machine_index < len(self.machines):
            return self.machines[machine_index].name
        return f"{self.name}_m{machine_index + 1}"

    @property
    def machine_speed_coefficients(self) -> List[float]:
        """Backward-compatible speed list view."""
        return [machine.speed_coefficient for machine in self.machines]

    def __repr__(self) -> str:
        machine_desc = ", ".join(
            f"{m.name}:x{m.speed_coefficient:g}" for m in self.machines
        )
        return (
            f"Station(name={self.name}, order={self.order_index}, machines={self.num_machines}, "
            f"workers_per_machine={self.workers_per_machine}, machine_details=[{machine_desc}], "
            f"sheet={self.sheet})"
        )

    def __lt__(self, other: 'Station') -> bool:
        """Allow sorting stations by order_index."""
        return self.order_index < other.order_index
