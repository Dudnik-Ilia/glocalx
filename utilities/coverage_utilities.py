import numpy as np
from scipy.spatial.distance import hamming

from rule import Rule, Unit


def covers(rule: Rule, x: np.array):
    """Does `rule` cover sample x?

    Args:
        rule (Rule): The rule.
        x (numpy.np.array): The record.
    Returns:
        bool: True if this rule covers c, False otherwise.
    """
    return all([(x[feature] >= lower) & (x[feature] < upper)] for feature, (lower, upper) in rule)


def binary_fidelity(unit: Unit, x, y, evaluator=None, ids=None, default=np.nan):
    """Evaluate the goodness of unit.
    Args:
        unit (Unit): The unit to evaluate.
        x (numpy.array): The data.
        y (numpy.array): The labels.
        evaluator (Evaluator): Optional evaluator to speed-up computation.
        ids (numpy.array): Unique identifiers to tell each element in @patterns apart.
        default (int): Default prediction for records not covered by the unit.
    Returns:
          float: The unit's fidelity_weight
    """
    coverage = evaluator.coverage(unit, x, ids=ids).flatten()
    unit_predictions = np.array([unit.consequence
                                 for _ in range(x.shape[0] if ids is None else ids.shape[0])]).flatten()
    unit_predictions[~coverage] = default

    fidelity = 1 - hamming(unit_predictions, y[ids] if ids is not None else y) if len(y) > 0 else 0

    return fidelity


def coverage_size(rule, x):
    """Evaluate the cardinality of the coverage of rule on x.

    (Number of records of X covered by rule)
    Args:
        rule (Rule): The rule.
        x (numpy.array): The validation set.

    Returns:
        (int: Number of records of X covered by rule.
    """
    return coverage_matrix([rule], x).sum().item(0)


def check_rule(rule, x, y=None):
    """
    Given rule, check for which samples from x it applies.
    If not y_exists: does not matter what the result of the rule is.
    """
    y_exists = y is not None

    if not y_exists:
        premises = [
            [(x[:, feature] > lower) & (x[:, feature] <= upper)]
            for feature, (lower, upper) in rule
        ]
    else:
        # Additionally check the result of the rule
        premises = [
            (x[:, feature] > lower) & (x[:, feature] <= upper)
            & (y == rule.consequence)
            for feature, (lower, upper) in rule
        ]

    premises = np.logical_and.reduce(premises)
    premises = premises.squeeze()
    # Take indexes of where True
    premises = np.argwhere(premises).squeeze()

    return premises


def coverage_matrix(rules, x, y=None, ids=None):
    """Compute the coverage of rule(s) over data 'x'.
    Args:
        rules (Union(list, Rule)): List of rules (or single Rule) whose coverage to compute.
        x (numpy.array): The validation set.
        y (numpy.array): The labels, if any. None otherwise. Defaults to None.
        ids (numpy.array): Unique identifiers to tell each element in `x` apart.
    Returns:
        numpy.array: The coverage matrix.
    """

    if isinstance(rules, list):
        coverage_matrix_ = np.full((len(rules), len(x)), False)
        hit_columns = [check_rule(rule, x, y) for rule in rules]

        for k, hits in zip(range(len(x)), hit_columns):
            coverage_matrix_[k, hits] = True
    else:
        coverage_matrix_ = np.full((len(x)), False)
        hit_columns = [check_rule(rules, x, y)]
        coverage_matrix_[hit_columns] = True

    coverage_matrix_ = coverage_matrix_[:, ids] if ids is not None else coverage_matrix_

    return coverage_matrix_

