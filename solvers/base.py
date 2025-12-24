"""Abstract base class for solvers."""

from abc import ABC, abstractmethod

from solution.solution import Solution
from models.problem import Problem
from evaluation.base import Evaluator


class Solver(ABC):
    """
    Abstract base class for all scheduling algorithms.
    Any algorithm must implement this interface.
    """

    @abstractmethod
    def solve(self, problem: Problem, evaluator: Evaluator) -> Solution:
        """
        Run algorithm and return best solution found.

        Args:
            problem: The problem instance to solve
            evaluator: The evaluator to use for fitness calculation

        Returns:
            The best solution found
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
