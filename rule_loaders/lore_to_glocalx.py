import json
from numpy import inf
from core.rule_glocalx import Rule
import pickle

def lore_to_glocalx(lore_rules_file:str, info_file:str) -> list:
    """Load LORE rules and convert it to GlocalX rules
    Args:
        lore_rules_file (str): Path to the Pickle file with LORE rules.
        info_file (str): Path to the info file containing the rules' metadata.
    Returns:
        (list): List of `Rule` objects.
    """
    with open(lore_rules_file, 'rb') as lore_rules, open(info_file, 'r') as info_log:
        loaded_rules = pickle.load(lore_rules)
        infos = json.load(info_log)

    assert 'class_values' in infos and 'feature_names' in infos, "Info file for loading LORE is not correct"
    class_values = infos['class_values']
    feature_names = infos['feature_names']

    loaded_rules = [r for r in loaded_rules if len(r) > 0]
    output_rules = []
    
    for lora_rule in loaded_rules:
        consequence = class_values.index(lora_rule.cons)
        premises = lora_rule.premises
        features = [feature_names.index(premise.att) for premise in premises]
        ops = [premise.op for premise in premises]
        values = [premise.thr for premise in premises]
        values_per_feature = {feature: [val for f, val in zip(features, values) if f == feature]
                              for feature in features}
        ops_per_feature = {feature: [op for f, op in zip(features, ops) if f == feature]
                           for feature in features}

        output_premises = {}
        for f in features:
            values, operators = values_per_feature[f], ops_per_feature[f]
            # 1 value, either <= or >
            if len(values) == 1:
                if operators[0] == '<=':
                    output_premises[f] = (-inf, values[0])
                else:
                    output_premises[f] = (values[0], +inf)
            # 2 values, < x <=
            else:
                output_premises[f] = (min(values), max(values))

        transformed_rule = Rule(premises=output_premises, consequence=consequence, names=feature_names)
        output_rules.append(transformed_rule)

    output_rules = list(set(output_rules))

    return output_rules