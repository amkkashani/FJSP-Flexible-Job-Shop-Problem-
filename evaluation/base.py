"""Abstract base class for evaluators."""

from abc import ABC, abstractmethod

from solution.solution import Solution
from models.problem import Problem


class Evaluator(ABC):
    """
    Abstract base class for solution evaluators.
    Computes the fitness/objective value of a solution.
    Different evaluators can combine waste and time differently.
    """

    @abstractmethod
    def evaluate(self, solution: Solution, problem: Problem) -> float:
        """
        Evaluate a solution and return its fitness value.

        Args:
            solution: The solution to evaluate
            problem: The problem instance

        Returns:
            Fitness value (lower is better)
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
