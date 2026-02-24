"""Abstract base class for solvers."""

from abc import ABC, abstractmethod
from typing import List, Optional

from solution.solution import Solution
from models.problem import Problem
from models.sheet import Sheet
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
        remaining_sheets: Optional[List[Sheet]] = None
    ) -> Solution:
        """
        Run algorithm and return best solution found.

        Args:
            problem: The problem instance to solve
            evaluator: The evaluator to use for fitness calculation
            remaining_sheets: Optional list of remaining sheets from previous runs

        Returns:
            The best solution found
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
