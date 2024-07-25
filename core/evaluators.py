"""
Evaluation module providing basic metrics to run and analyze GLocalX's results.
Two evaluators are provided, DummyEvaluator, which does not optimize performance (stored in base_evaluators),
and MemEvaluator, which stores previously computed measures to speed-up performance.
"""
from abc import abstractmethod

import numpy as np
from scipy.spatial.distance import hamming

from logzero import logger

from base_classes.base_evaluators import Evaluator
from core.rule_glocalx import Rule
from utilities.coverage_utilities import covers, coverage_matrix, binary_fidelity


class MemEvaluator(Evaluator):
    """Memoization-aware Evaluator to avoid evaluating the same measures over the same data."""

    def __init__(self, model_ai):
        """Constructor."""
        self.oracle = model_ai
        self.coverages = dict()
        self.perfect_coverages = dict()
        self.intersecting = dict()
        self.bics = dict()
        self.distances = dict()
        self.binary_fidelities = dict()
        self.coverage_sizes = dict()
        self.scores = dict()

    @abstractmethod
    def covers(self, rule, x):
        """Does @rule cover c?

        Args:
            rule (Rule): The rule.
            x (numpy.array): The record.
        Returns:
            bool: True if this rule covers c, False otherwise.
        """
        return covers(rule, x)

    def coverage(self, rules, x, y=None, ids=None):
        """Compute the coverage of rules over samples.
        Args:
            rules (Union(Rule, list): Rule (or list of rules) whose coverage to compute.
            x (numpy.array): The validation set, dataset of several examples.
            y (numpy.array): The labels, if any. None otherwise. Defaults to None.
            ids (numpy.array): IDS of the given samples, used to speed up evaluation.
        Returns:
            numpy.array: The coverage matrix.
        """
        rules_ = [rules] if not isinstance(rules, list) and not isinstance(rules, set) else rules

        if y is None:
            for rule in rules_:
                # If we did not calculate before (Memoization)
                if rule not in self.coverages:
                    self.coverages[rule] = coverage_matrix(rule, x, y)
            cov = np.array([self.coverages[rule] for rule in rules_])
        else:
            for rule in rules_:
                if rule not in self.perfect_coverages:
                    self.perfect_coverages[rule] = coverage_matrix(rule, x, y)
            cov = np.array([self.perfect_coverages[rule] for rule in rules_])

        cov = cov[:, ids] if ids is not None else cov

        return cov

    def distance(self, A, B, x, ids=None):
        """
        Compute the distance between ruleset `A` and ruleset `B`.
        Using Memoization for remembering distances
        Args:
            A (iterable): Ruleset.
            B (iterable): Ruleset.
            x (numpy.array): Data.
            ids (numpy.array): IDS of the given `x`, used to speed up evaluation.
        Returns:
            (float): The Jaccard distance between the two.
        """
        # If A ruleset already was calculated with respect to B
        if tuple(A) in self.distances and tuple(B) in self.distances[tuple(A)]:
            diff = self.distances[tuple(A)][tuple(B)]
            return diff
        # Or B to A
        if tuple(B) in self.distances and tuple(A) in self.distances[tuple(B)]:
            diff = self.distances[tuple(B)][tuple(A)]
            return diff

        # New distance Compute
        coverage_A, coverage_B = self.coverage(A, x, ids=ids).sum(axis=0), self.coverage(B, x, ids=ids).sum(axis=0)
        diff = hamming(coverage_A, coverage_B)

        # Saving the results
        # If A/B already was in self.distances
        if tuple(A) in self.distances:
            self.distances[tuple(A)][tuple(B)] = diff
        if tuple(B) in self.distances:
            self.distances[tuple(B)][tuple(A)] = diff

        # If it is 1st time for A/B
        if tuple(A) not in self.distances:
            self.distances[tuple(A)] = {tuple(B): diff}
        if tuple(B) not in self.distances:
            self.distances[tuple(B)] = {tuple(A): diff}

        return diff

    def binary_fidelity(self, rule, x, y, ids=None):
        """Evaluate the goodness of rule.
        Args:
            rule (Unit): The unit to evaluate.
            x (numpy.array): The data.
            y (numpy.array): The labels.
            default (int): Default prediction for records not covered by the rule.
            ids (numpy.array): IDS of the given `x`, used to speed up evaluation.
        Returns:
              float: The rule's fidelity_weight
        """
        if y is None:
            y = self.oracle.predict(x).round().squeeze()

        if ids is None:
            if rule not in self.binary_fidelities:
                self.binary_fidelities[rule] = binary_fidelity(rule, x, y, evaluator=self)
            fidelity = self.binary_fidelities[rule]
        else:
            fidelity = binary_fidelity(rule, x, y, self, ids=ids)

        return fidelity

    def binary_fidelity_model(self, rules, x, y, k=1, default=None, ids=None):
        """Evaluate the goodness of the `rules`.
        Args:
            rules (Union(list, set)): The rules to evaluate.
            x (numpy.array): The data.
            y (numpy.array): The labels.
            k (int): Number of rules to use in the Laplacian prediction schema.
            default (int): Default prediction for records not covered by the unit.
            ids (numpy.array): Unique identifiers to tell each element in @c apart.
        Returns:
              float: The rules fidelity_weight.
        """
        if y is None:
            y = self.oracle.predict(x).squeeze().round()

        fidelities = np.array([self.binary_fidelity(rule, x, y) for rule in rules])
        coverage = self.coverage(rules, x)

        if len(rules) == 0:
            predictions = [default] * y.shape[0]
        else:
            rules_consequences = np.array([r.consequence for r in rules])
            # Fast computation for k = 1
            if k == 1:
                weighted_coverage_scores = coverage * fidelities.reshape(-1, 1)  # Coverage matrix weighted by score
                # Best score per row (i.e., record)
                best_rule_per_record_idx = weighted_coverage_scores.argmax(axis=0).squeeze()
                predictions = rules_consequences[best_rule_per_record_idx]
                # Replace predictions of non-covered records w/ default prediction
                predictions[coverage.sum(axis=0) == 0] = default
            # Iterative computation
            else:
                predictions = []
                for record in range(len(x)):
                    record_coverage = np.argwhere(coverage[:, record]).ravel()
                    if len(record_coverage) == 0:
                        prediction = default
                    else:
                        companions_0 = record_coverage[rules_consequences[record_coverage] == 0]
                        companions_1 = record_coverage[rules_consequences[record_coverage] == 1]
                        scores_0 = fidelities[companions_0]
                        scores_1 = fidelities[companions_1]
                        np.argsort_scores_0 = np.flip(np.argsort(fidelities[companions_0])[-k:])
                        np.argsort_scores_1 = np.flip(np.argsort(fidelities[companions_1])[-k:])
                        top_scores_0 = scores_0[np.argsort_scores_0]
                        top_scores_1 = scores_1[np.argsort_scores_1]

                        if len(top_scores_0) == 0 and len(top_scores_1) > 0:
                            prediction = 1
                        elif len(top_scores_1) == 0 and len(top_scores_0) > 0:
                            prediction = 0
                        elif len(top_scores_1) == 0 and len(top_scores_0) == 0:
                            prediction = default
                        else:
                            prediction = 0 if np.mean(top_scores_0) > np.mean(top_scores_1) else 1

                    predictions.append(prediction)
                predictions = np.array(predictions)
        fidelity = 1 - hamming(predictions, y) if len(y) > 0 else 0

        return fidelity

    def bic(self, rules, vl, fidelity_weight=1., complexity_weight=1.):
        """
        Compute the Bayesian Information Criterion for the given `rules` set.
        Args:
            rules (set): Ruleset.
            vl (numpy.array): Validation set.
            fidelity_weight (float): Weight to fidelity_weight (BIC-wise).
            complexity_weight (float): Weight to complexity_weight (BIC-wise).
        Returns:
            float: Model BIC 
        """
        if tuple(rules) in self.bics:
            model_bic = self.bics[tuple(rules)]
        else:
            x, y = vl[:, :-1], vl[:, -1]
            n, m = x.shape
            default = int(y.mean().round())
            log_likelihood = self.binary_fidelity_model(rules, x, y, default=default)

            model_complexity = np.mean([len(r) / m for r in rules])
            model_bic = - (fidelity_weight * log_likelihood - complexity_weight * model_complexity / n)

            logger.debug('Log likelihood: ' + str(log_likelihood) + ' | Complexity: ' + str(model_complexity))

            self.bics[tuple(rules)] = model_bic

        return model_bic

    def forget(self, rules, A=None, B=None):
        """
        Remove rules from this Evaluator's memory. Return the updated evaluator.
        Args:
            rules (iterable): Rules to remove.
            A (set): Rules merged.
            B (set): Rules merged.
        Returns:
            MemEvaluator: This evaluator with no memory of `rules`.

        """
        for rule in rules:
            if rule in self.binary_fidelities:
                del self.binary_fidelities[rule]
            if rule in self.coverages:
                del self.coverages[rule]
            if rule in self.coverage_sizes:
                del self.coverage_sizes[rule]
            if rule in self.perfect_coverages:
                del self.perfect_coverages[rule]
            if rule in self.scores:
                del self.scores[rule]

        if A is not None and B is not None:
            # Delete the whole A, as it has been merged and does not exist anymore
            del self.distances[tuple(A)]
            # Delete the whole B, as it has been merged and does not exist anymore
            del self.distances[tuple(B)]
            # Delete every reference to any of them, as they have been merged and do not exist anymore
            for T in self.distances:
                if tuple(A) in self.distances[T]:
                    del self.distances[T][tuple(A)]
                if tuple(B) in self.distances[T]:
                    del self.distances[T][tuple(B)]

        return self




