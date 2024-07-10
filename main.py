from tensorflow.keras.models import load_model
import numpy as np
import logzero

from glocalx import GLocalX, shut_up_tensorflow
from models import Rule

# Set log profile: INFO for normal logging, DEBUG for verbosity
logzero.loglevel(logzero.logging.INFO)
shut_up_tensorflow()

# Load black box: optional! Use black_box = None to use the dataset labels
black_box = load_model('data/dummy/dummy_model.h5')
# Load data and header
data = np.genfromtxt('data/dummy/dummy_dataset.csv', delimiter=',', names=True)
features_names = data.dtype.names

tr_set = [list(sample) for sample in data]
tr_set = np.array(tr_set)

# Load local explanations
local_explanations = Rule.from_json('data/dummy/dummy_rules.json', names=features_names)

# Create a GLocalX instance for `black_box`
glocalx = GLocalX(model_ai=black_box)
# Fit the model, use batch_size=128 for larger datasets
glocalx = glocalx.fit(local_explanations, tr_set, batch_size=2, name='black_box_explanations')

# Retrieve global explanations by fidelity
alpha = 0.5
global_explanations = glocalx.get_fine_boundary_alpha(alpha, tr_set)
# Retrieve global explanations by fidelity percentile
alpha = 95
global_explanations = glocalx.get_fine_boundary_alpha(alpha, tr_set, is_percentile=True)
# Retrieve exactly `alpha` global explanations, `alpha/2` per class
alpha = 10
global_explanations = glocalx.get_fine_boundary_alpha(alpha, tr_set)