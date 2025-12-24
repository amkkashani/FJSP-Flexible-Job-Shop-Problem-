"""Weighted evaluator implementation."""

from .base import Evaluator
from solution.solution import Solution
from models.problem import Problem


class WeightedEvaluator(Evaluator):
    """
    Evaluator that uses weighted sum of waste and makespan.

    fitness = alpha * total_waste + beta * makespan

    Attributes:
        alpha: Weight for waste (default: 1.0)
        beta: Weight for makespan (default: 1.0)
    """

    def __init__(self, alpha: float = 1.0, beta: float = 1.0):
        """
        Initialize the weighted evaluator.

        Args:
            alpha: Weight for total waste
            beta: Weight for makespan
        """
        self.alpha = alpha
        self.beta = beta

    def evaluate(self, solution: Solution, problem: Problem) -> float:
        """
        Evaluate a solution using weighted sum objective.

        Args:
            solution: The solution to evaluate
            problem: The problem instance

        Returns:
            Fitness value (lower is better)
        """
        total_waste = solution.get_total_waste()
        makespan = solution.get_makespan()

        return self.alpha * total_waste + self.beta * makespan

    def get_components(self, solution: Solution, problem: Problem) -> dict:
        """
        Get individual components of the evaluation.

        Args:
            solution: The solution to evaluate
            problem: The problem instance

        Returns:
            Dictionary with waste, makespan, and weighted components
        """
        total_waste = solution.get_total_waste()
        makespan = solution.get_makespan()

        return {
            'total_waste': total_waste,
            'makespan': makespan,
            'weighted_waste': self.alpha * total_waste,
            'weighted_makespan': self.beta * makespan,
            'fitness': self.alpha * total_waste + self.beta * makespan
        }

    def __repr__(self) -> str:
        return f"WeightedEvaluator(alpha={self.alpha}, beta={self.beta})"
