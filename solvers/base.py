"""Abstract base class for solvers."""

from abc import ABC, abstractmethod
from typing import List, Optional

from solution.solution import Solution
from models.problem import Problem
from models.remaining import RemainingSection
from evaluation.base import Evaluator


class Solver(ABC):
    """
    Abstract base class for all scheduling algorithms.
    Any algorithm must implement this interface.
    """

    @abstractmethod
    def solve(
        self,
        problem: Problem,
        evaluator: Evaluator,
        remaining_sections: Optional[List[RemainingSection]] = None
    ) -> Solution:
        """
        Run algorithm and return best solution found.

        Args:
            problem: The problem instance to solve
            evaluator: The evaluator to use for fitness calculation
            remaining_sections: Optional list of remaining sections from previous runs

        Returns:
            The best solution found
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
