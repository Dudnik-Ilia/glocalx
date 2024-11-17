import numpy as np
import logzero
import pickle

from glocalx import GLocalX, shut_up_tensorflow
from rule_loaders.lore_to_glocalx import lore_to_glocalx

from lore_explainer.datamanager import prepare_adult_dataset, prepare_dataset

# Set log profile: INFO for normal logging, DEBUG for verbosity
logzero.loglevel(logzero.logging.DEBUG)
shut_up_tensorflow()

# Load black box: optional! Use black_box = None to use the dataset labels
model_file = "RandomForestClassifier_for_adult"
# Load the model
with open(f"data/models/{model_file}.pkl", 'rb') as model_file:
    black_box = pickle.load(model_file)
# Load data and header

# data = np.genfromtxt('data/dummy/dummy_dataset.csv', delimiter=',', names=True)
# features_names = data.dtype.names

data_filename = 'adult'
# Prepare data
if data_filename == "adult":
    df, class_name = prepare_adult_dataset(f'data/{data_filename}.csv')
else:
    raise NotImplementedError
df, feature_names, *_ = prepare_dataset(
    df, class_name)
data = df.to_numpy()
data = [list(sample) for sample in data]
data = np.array(data)

# Load local explanations
lore_rules_file = "lore_rules_adult_30"
info_file = "adult_info"
glocal_rules = lore_to_glocalx(f"data/lore_rules/{lore_rules_file}.pkl", f"data/info_files/{info_file}.json")
print(glocal_rules)
# Create a GLocalX instance for `black_box`
glocalx = GLocalX(model_ai=black_box)
# Fit the model, use batch_size=128 for larger datasets
glocalx.fit(glocal_rules, data, batch_size=128,)

# Retrieve global explanations by fidelity
alpha = 0.5
# global_explanations = glocalx.get_fine_boundary_alpha(alpha, data)
# Retrieve global explanations by fidelity percentile
alpha = 0.95
# global_explanations = glocalx.get_fine_boundary_alpha(alpha, data)
# Retrieve exactly `alpha` global explanations, `alpha/2` per class
alpha = 10
global_explanations = glocalx.get_fine_boundary_alpha(alpha)
print(len(global_explanations))
