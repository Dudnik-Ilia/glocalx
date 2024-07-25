from abc import abstractmethod

import numpy as np
from scipy.spatial.distance import hamming

from utilities.coverage_utilities import binary_fidelity_old, covers, coverage_matrix, coverage_size


class Evaluator:
    """Evaluator interface. Evaluator objects provide coverage and fidelity_weight utilities."""

    @abstractmethod
    def coverage(self, rules, patterns, target=None, ids=None):
        """Compute the coverage of @rules over @patterns.
        Args:
            rules (list) or (Rule):
            patterns (numpy.array): The validation set.
            target (numpy.array): The labels, if any. None otherwise. Defaults to None.
            ids (numpy.array): Unique identifiers to tell each element in @patterns apart.
        Returns:
            numpy.array: The coverage matrix.
        """
        pass

    @abstractmethod
    def coverage_size(self, rule, x, ids=None):
        """Evaluate the cardinality of the coverage of unit on c.

        Args:
            rule (Rule): The rule.
            x (numpy.array): The validation set.
            ids (numpy.array): Unique identifiers to tell each element in @patterns apart.
        Returns:
            int: Number of records of X covered by rule.
        """
        pass

    @abstractmethod
    def binary_fidelity(self, unit, x, y, ids=None, default=np.nan):
        """Evaluate the goodness of unit.
        Args:
            unit (Unit): The unit to evaluate.
            x (numpy.array): The data.
            y (numpy.array): The labels.
            ids (numpy.array): Unique identifiers to tell each element in @patterns apart.
            default (int): Default prediction when no rule covers a record.
        Returns:
              float: The unit's fidelity_weight
        """
        pass

    @abstractmethod
    def binary_fidelity_model(self, units, x, y, k=1, default=None, ids=None):
        """Evaluate the goodness of the `units`.
        Args:
            units (Union(list, set)): The units to evaluate.
            x (numpy.array): The data.
            y (numpy.array): The labels.
            k (int): Number of rules to use in the Laplacian prediction schema.
            default (int): Default prediction for records not covered by the unit.
            ids (numpy.array): Unique identifiers to tell each element in @c apart.
        Returns:
            float: The units fidelity_weight.
        """
        pass

    @abstractmethod
    def covers(self, rule, x):
        """Does @rule cover c?

        Args:
            rule (Rule): The rule.
            x (numpy.array): The record.
        Returns:
            bool: True if this rule covers c, False otherwise.
        """
        pass

    @abstractmethod
    def bic(self, rules, vl, fidelity_weight=1., complexity_weight=1.):
        """
        Compute the Bayesian Information Criterion for the given `rules` set.
        Args:
            rules (set): Ruleset.
            vl (numpy.array): Validation set.
            fidelity_weight (float): Weight to fidelity_weight (BIC-wise).
            complexity_weight (float): Weight to complexity_weight (BIC-wise).
        Returns:
            tuple: Triple (BIC, log likelihood, complexity_weight).
        """
        pass


# Not used
class DummyEvaluator(Evaluator):
    """Dummy evaluator with no memory: every result is computed at each call!"""

    def bic(self, rules, vl, fidelity_weight=1., complexity_weight=1.):
        """
        Compute the Bayesian Information Criterion for the given `rules` set.
        Args:
            rules (set): Ruleset.
            vl (numpy.array): Validation set.
            fidelity_weight (float): Weight to fidelity_weight (BIC-wise).
            complexity_weight (float): Weight to complexity_weight (BIC-wise).
        Returns:
            tuple: Triple (BIC, log likelihood, complexity_weight).
        """
        x, y = vl[:, :-1], vl[:, -1]
        n = x.shape[0]
        default = round(y.mean() + .5)
        log_likelihood = [binary_fidelity_old(rule, x, y, default=default, ids=None) for rule in rules]
        log_likelihood = np.mean(log_likelihood)

        model_complexity = len(rules)
        model_bic = - (fidelity_weight * log_likelihood - complexity_weight * model_complexity / n)

        return model_bic, log_likelihood, model_complexity

    def __init__(self, oracle):
        """Constructor."""
        self.oracle = oracle
        self.coverages = dict()
        self.binary_fidelities = dict()
        self.coverage_sizes = dict()

    def covers(self, rule, x):
        """Does `rule` cover `x`?

        Args:
            rule (Rule): The rule.
            x (numpy.array): The record.
        Returns:
            bool: True if this rule covers c, False otherwise.
        """
        return covers(rule, x)

    def coverage(self, rules, x, y=None, ids=None):
        """Compute the coverage of @rules over @patterns.
        Args:
            rules (Union(Rule, list): Rule (or list of rules) whose coverage to compute.
            x (numpy.array): The validation set.
            y (numpy.array): The labels, if any. None otherwise. Defaults to None.
            ids (numpy.array): Unique identifiers to tell each element in @patterns apart.
        Returns:
            numpy.array: The coverage matrix.
        """
        rules_ = rules if isinstance(rules, list) else [rules]
        coverage_ = coverage_matrix(rules_, x, y, ids=ids)

        return coverage_

    def coverage_size(self, rule, x, ids=None):
        """Evaluate the cardinality of the coverage of unit on c.

        Args:
            rule (Rule): The rule.
            x (numpy.array): The validation set.
            ids (numpy.array): Unique identifiers to tell each element in `x` apart.
        Returns:
            numpy.array: Number of records of X covered by rule.
        """
        return coverage_size(rule, x)

    def binary_fidelity(self, unit, x, y, default=np.nan, ids=None):
        """Evaluate the goodness of unit.
        Args:
            unit (Unit): The unit to evaluate.
            x (numpy.array): The data.
            y (numpy.array): The labels.
            ids (numpy.array): Unique identifiers to tell each element in @patterns apart.
            default (int): Default prediction when no rule covers a record.
        Returns:
              (float): The unit's fidelity_weight
        """
        if self.oracle is not None or y is None:
            y = self.oracle.predict(x).round().squeeze()

        return binary_fidelity_old(unit, x, y, self, default=default, ids=ids)

    def binary_fidelity_model(self, units, x, y, k=1, default=None, ids=None):
        """Evaluate the goodness of unit.
        Args:
            units (Union(list, set): The units to evaluate.
            x (numpy.array): The data.
            y (numpy.array): The labels.
            k (int): Number of rules to use in the Laplacian prediction schema.
            default (int): Default prediction for records not covered by the unit.
            ids (numpy.array): Unique identifiers to tell each element in @patterns apart.
        Returns:
              numpy.array: The units fidelity_weight.
        """
        if self.oracle is not None:
            y = (self.oracle.predict(x).round().squeeze())

        scores = np.array([self.binary_fidelity(rule, x, y, default=default) for rule in units])
        coverage = self.coverage(units, x, y)

        predictions = []
        for record in range(len(x)):
            companions = scores[coverage[:, record]]
            companion_units = units[coverage[:, record]]
            top_companions = np.argsort(companions)[-k:]
            top_units = companion_units[top_companions]
            top_fidelities = companions[top_companions]
            top_fidelities_0 = [top_fidelity for top_fidelity, top_unit in zip(top_fidelities, top_units)
                                if top_unit.consequence == 0]
            top_fidelities_1 = [top_fidelity for top_fidelity, top_unit in zip(top_fidelities, top_units)
                                if top_unit.consequence == 1]

            if len(top_fidelities_0) == 0 and len(top_fidelities_1) > 0:
                prediction = 1
            elif len(top_fidelities_1) == 0 and len(top_fidelities_0) > 0:
                prediction = 0
            elif len(top_fidelities_1) == 0 and len(top_fidelities_0) == 0:
                prediction = default
            else:
                prediction = 0 if np.mean(top_fidelities_0) > np.mean(top_fidelities_1) else 1

            predictions.append(prediction)
        predictions = np.array(predictions)
        fidelity = 1 - hamming(predictions, y) if len(y) > 0 else 0

        return fidelity

