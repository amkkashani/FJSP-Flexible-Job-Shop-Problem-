"""Weighted evaluator implementation."""

from .base import Evaluator
from solution.solution import Solution
from models.problem import Problem


class WeightedEvaluator(Evaluator):
    """
    Evaluator that uses weighted sum of waste, makespan, and product delivery.

    fitness = alpha * total_waste + beta * makespan + gamma * avg_product_completion

    Attributes:
        alpha: Weight for waste (default: 1.0)
        beta: Weight for makespan (default: 1.0)
        gamma: Weight for average product completion time (default: 0.0)
    """

    def __init__(self, alpha: float = 1.0, beta: float = 1.0, gamma: float = 0.0):
        """
        Initialize the weighted evaluator.

        Args:
            alpha: Weight for total waste
            beta: Weight for makespan
            gamma: Weight for average product completion time (product delivery)
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

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
        avg_product_completion = solution.get_avg_product_completion_time(problem)

        return self.alpha * total_waste + self.beta * makespan + self.gamma * avg_product_completion

    def get_components(self, solution: Solution, problem: Problem) -> dict:
        """
        Get individual components of the evaluation.

        Args:
            solution: The solution to evaluate
            problem: The problem instance

        Returns:
            Dictionary with waste, makespan, product delivery, and weighted components
        """
        total_waste = solution.get_total_waste()
        makespan = solution.get_makespan()
        avg_product_completion = solution.get_avg_product_completion_time(problem)

        return {
            'total_waste': total_waste,
            'makespan': makespan,
            'avg_product_completion': avg_product_completion,
            'weighted_waste': self.alpha * total_waste,
            'weighted_makespan': self.beta * makespan,
            'weighted_product_delivery': self.gamma * avg_product_completion,
            'fitness': self.alpha * total_waste + self.beta * makespan + self.gamma * avg_product_completion
        }

    def __repr__(self) -> str:
        return f"WeightedEvaluator(alpha={self.alpha}, beta={self.beta}, gamma={self.gamma})"
