########################
# Framework validation #
########################
import numpy as np

from evaluators import MemEvaluator
from models import Rule
from utilities.coverage_utilities import coverage_matrix


# Not used
def validate(glocalx, oracle, vl, m=None, alpha=0.5, is_percentile=False):
    """Validate the given `glocalx` instance on the given `tr` dataset.
    Arguments:
        glocalx (Union(GLocalX, list)): GLocalX object or list of rules.
        oracle (Predictor): Oracle to validate against.
        vl (numpy.array): Validation set.
        m (int): Initial number of rules, if known, None otherwise. Defaults to None.
        alpha (Union(float, int, iterable)): Pruning hyperparameter, rules with score
                                            less than `alpha` are removed from the ruleset
                                            used to perform the validation. The score can be
                                            - rule fidelity (`alpha` float, `is_percentile`=False)
                                            - rule fidelity percentile (`alpha` float, `is_percentile`=True)
                                            - number of rules (`alpha` integer)
                                            Same applies if an iterable is provided.
                                            Defaults to '0.5'.
        is_percentile (bool): Whether the provided `alpha` is a percentile or not. Defaults to False.
    Returns:
        dict: Dictionary of validation measures.
    """

    def len_reduction(ruleset_a, ruleset_b):
        return ruleset_a / ruleset_b

    def coverage_pct(rules, x):
        coverage = coverage_matrix(rules, x)
        coverage_percentage = (coverage.sum(axis=0) > 0).sum() / x.shape[0]

        return coverage_percentage

    if oracle is None:
        x = vl[:, :-1]
        y = vl[:, -1]
        evaluator = MemEvaluator(model_ai=None)
    else:
        evaluator = MemEvaluator(model_ai=oracle)
        x = vl[:, :-1]
        y = oracle.predict(x).round().squeeze()
    majority_label = int(y.mean().round())

    if isinstance(alpha, float) or isinstance(alpha, int):
        alphas = [alpha]
    else:
        alphas = alpha

    results = {}
    for alpha in alphas:
        if isinstance(glocalx, list) or isinstance(glocalx, set):
            rules = glocalx
        else:
            if oracle is None:
                evaluator = MemEvaluator(model_ai=None)
            rules = glocalx.get_fine_boundary_alpha(alpha=alpha, data=np.hstack((x, y.reshape(-1, 1))),
                                                    evaluator=evaluator, is_percentile=is_percentile)
        rules = [r for r in rules if len(r) > 0 and isinstance(r, Rule)]

        if len(rules) == 0:
            results[alpha] = {
                'alpha': alpha,
                'fidelity': np.nan,
                'fidelity_weight': np.nan,
                'coverage': np.nan,
                'mean_length': np.nan,
                'std_length': np.nan,
                'rule_reduction': np.nan,
                'len_reduction': np.nan,
                'mean_prediction': np.nan,
                'std_prediction': np.nan,
                'size': 0
            }
            continue

        evaluator = MemEvaluator(model_ai=oracle)
        validation = dict()
        validation['alpha'] = alpha
        validation['size'] = len(rules)
        validation['fidelity'] = evaluator.binary_fidelity_model(rules, x=x, y=y, default=majority_label, k=1)
        validation['coverage'] = coverage_pct(rules, x)
        validation['mean_length'] = np.mean([len(r) for r in rules])
        validation['std_length'] = np.std([len(r) for r in rules])
        validation['rule_reduction'] = 1 - len(rules) / m if m is not None else np.nan
        validation['len_reduction'] = len_reduction(validation['mean_length'], m) if m is not None else np.nan

        # Predictions
        validation['mean_prediction'] = np.mean([r.consequence for r in rules])
        validation['std_prediction'] = np.std([r.consequence for r in rules])

        results[alpha] = validation

    return results
