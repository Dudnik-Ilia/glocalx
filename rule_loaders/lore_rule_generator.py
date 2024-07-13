import warnings
import random
import json
import pickle
from tqdm import tqdm

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score

from lore_explainer.util import neuclidean
from lore_explainer.datamanager import prepare_adult_dataset, prepare_dataset
from lore_explainer.lorem import LOREM

"""
This script creates LORE rules and stores'em in pickle file in ./data/lore_rules/
Also info file - in ./data/info_files/
Also model (if was not provided to this script) is trained and saved at ./data/models/
Args:
    data_filename: str, file name for data (csv) in ./data
    bb: str, black box model name file (will be trained if None)
    num_samples: int, number of samples to process to get rules == num of rules
Result:
    Saved data_info json file (needed for conversion to glocalx format)
    Saved lore_rules pickle files
    Saved trained model (if not given)
"""

data_filename = "adult"     # CSV name in ./data for which to generate LORE rules
bb = None                   # Balck box model, str, name file
num_samples = 30            # Select num of records to explain
# Also additional settings below for LORE

################################DATA+MODEL################################

# warning suppression
with warnings.catch_warnings():
    warnings.simplefilter(action='ignore', category=FutureWarning)
    
    # Prepare data
    if data_filename == "adult":
        df, class_name = prepare_adult_dataset(f'data/{data_filename}.csv')
    else:
        raise NotImplementedError
    df, feature_names, class_values, numeric_columns, df_orig, real_feature_names, features_map = prepare_dataset(
        df, class_name)

test_size = 0.30
random_state = 42

X_train, X_test, Y_train, Y_test = train_test_split(df[feature_names].values, df[class_name].values, 
                                                    test_size=test_size,
                                                    random_state=random_state, 
                                                    stratify=df[class_name].values)

_, X_test_orig, _, _ = train_test_split(df_orig[real_feature_names].values, df_orig[class_name].values, 
                            test_size=test_size,
                            random_state=random_state, 
                            stratify=df[class_name].values)

# Load model
if bb is None:
    # Train a black box classifier from scratch
    bb = RandomForestClassifier(n_estimators=100, random_state=random_state)
    model_name = "RandomForestClassifier"
    bb.fit(X_train, Y_train)
    # save model
    with open(f"data/models/{model_name}_for_{data_filename}.pkl", "wb") as model_file:
        pickle.dump(bb, model_file)
elif isinstance(bb, str):
    # If given, load it
    with open(f"data/models/{bb}.pkl", 'rb') as model_file:
        bb = pickle.load(model_file)
else:
    raise ValueError("provided model is not recognized")

Y_pred = bb.predict(X_test)

print('Accuracy %.3f' % accuracy_score(Y_test, Y_pred))
print('F1-measure %.3f' % f1_score(Y_test, Y_pred))

##################################LORE####################################

indices = random.sample(range(len(X_test)), num_samples)
x_to_exp = X_test[indices]

# LORE initialize
lore_explainer = LOREM(X_test_orig, bb.predict,
                        feature_names, class_name, class_values, numeric_columns, features_map,
                        neigh_type='geneticp',
                        categorical_use_prob=True,
                        continuous_fun_estimation=False,
                        size=1000, ocr=0.1, random_state=random_state,
                        ngen=10, bb_predict_proba=bb.predict_proba, 
                        verbose=False)

rules = []

# warning suppression
with warnings.catch_warnings():
    warnings.simplefilter(action='ignore', category=RuntimeWarning)

    for i in tqdm(range(num_samples)):
        exp = lore_explainer.explain_instance(x_to_exp[i], samples=50, use_weights=True, metric=neuclidean)
        rules.append(exp.rule)

# Create rules file
with open(f"data/lore_rules/lore_rules_{data_filename}_{num_samples}.pkl", "wb") as file:
    pickle.dump(rules, file)

# Create info file (for conversion to glocalx rules later)
info_dict = {'feature_names': feature_names, "class_values": class_values}
with open(f"data/info_files/{data_filename}_info.json", "w") as file:
    json.dump(info_dict, file, indent=4)
