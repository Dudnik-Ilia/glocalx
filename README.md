# GLocalX - Global through Local Explainability

Explanations come in two forms: local, explaining a single model prediction, and global, explaining all model predictions. The Local to Global (L2G) problem consists in bridging these two family of explanations. Simply put, we generate global explanations by merging local ones.

## The algorithm

Local and global explanations are provided in the form of decision rules:

```
age < 40, income > 50000, status = married, job = CEO ⇒ grant loan application
```

This rule describes the rationale followed given by an unexplainable model to grant the loan application to an individual younger than 40 years-old, with an income above 50000$, married and currently working as a CEO.

---

## Setup

```bash
git clone https://github.com/msetzu/glocalx/
cd glocalx
```

Dependencies are listed in `requirements.txt`, a virtual environment is advised:

```bash
mkvirtualenv glocalx # optional but reccomended
pip3 install -r requirements.txt
```

## Running the code

### Python interface

```python
from tensorflow.keras.models import load_model
from numpy import genfromtxt, float as np_float
import logzero

from glocalx import GLocalX, shut_up_tensorflow
from models import Rule

# Set log profile: INFO for normal logging, DEBUG for verbosity
logzero.loglevel(logzero.logging.INFO)
shut_up_tensorflow()

# Load black box: optional! Use black_box = None to use the dataset labels
black_box = load_model('data/dummy/dummy_model.h5')
# Load data and header
data = genfromtxt('data/dummy/dummy_dataset.csv', delimiter=',', names=True)
features_names = data.dtype.names
tr_set = data.view(np_float).reshape(data.shape + (-1,))

# Load local explanations
local_explanations = Rule.from_json('data/dummy/dummy_rules.json', names=features_names)

# Create a GLocalX instance for `black_box`
glocalx = GLocalX(oracle=black_box, name='black_box_explanations')
# Fit the model
glocalx = glocalx.fit(local_explanations, tr_set)

# Retrieve global explanations
alpha = 0.5
global_explanations = glocalx.rules(alpha, tr_set)
```



### Command line interface

You can invoke `GLocalX` from the api interface in `api.py`:

```bash
> python3 api.py --help
Usage: api.py [OPTIONS] RULES TR

Options:
  -o, --oracle PATH
  --names TEXT                   Features names.
  -cbs, --callbacks FLOAT        Callback step, either int or float. Defaults
                                 to 0.1

  -m, --name TEXT                Name of the log files.
  --generate TEXT                Number of records to generate, if given.
                                 Defaults to None.

  -i, --intersect TEXT           Whether to use coverage intersection
                                 ('coverage') or polyhedra intersection
                                 ('polyhedra'). Defaults to 'coverage'.

  -f, --fidelity_weight FLOAT    Fidelity weight. Defaults to 1.
  -c, --complexity_weight FLOAT  Complexity weight. Defaults to 1.
  -a, --alpha FLOAT              Pruning factor. Defaults to 0.5
  -b, --batch INTEGER            Batch size. Set to -1 for full batch.
                                 Defaults to 128.

  -u, --undersample FLOAT        Undersample size, to use a percentage of the
                                 rules. Defaults to 1.0 (No undersample).

  --high_concordance             Use to use high concordance.
  --strong_cut                   Use to use the strong cut.
  --global_direction             Use to use the global search direction.
  -d, --debug INTEGER            Debug level.
  --help                         Show this message and exit.

```

A minimal run simply requires a set of local input rules, a training set and (optionally) a black box:

```bash
python3.8 apy.py my_rules.json training_set.csv my_black_box.h5 \
		--name my_black_box_explanations
```

If you are interested, the folder `data/dummy/` contains a dummy example.

The remaining hyperparameters are optional:

- `--names $names`  Comma-separated features names
- `--callbacks $callbacks` Callback step, either int or float. Callbacks are invoked every `--callbacks` iterations of the algorithm
- `--name $name` Name of the log files and output. The script dumps here all the additional info as it executes
- `--generate $size` Number of synthetic records to generate, if you don't wish to use the provided training set
- `--intersect $strategy` Intersection strategy: either `coverage` or `polyhedra`. Defaults to `coverage`
- `--fidelity_weight $fidelity` Fidelity weight to reward accurate yet complex models. Defaults to 1.
- `--complexity_weight $complexity` Complexity weight to reward simple yet less accurate models. Defaults to 1.
- `--alpha $alpha` Pruning factor. Defaults to 0.5
- `--batch $batch` Batch size. Set to -1 for full batch. Defaults to 128.
- `--undersample $pct` Undersample size, to use a percentage of the input rules. Defaults to 1.0 (all rules)
- `--high_concordance` to use a more stringent `join`
- `--strong_cut` to use a more stringent `cut`
- `--global_direction` Use to evaluate merges on the whole validation set
- `--debug` Debug level: the higher, the less messages shown

## Validation

`GLocalX` outputs a set of rules (list of `models.Rule`) stored in a `$names.rules.glocalx.alpha=$alpha.json` file.

The `evaluators.validate` function provides a simple interface for validation. If you wish to extend it, you can directly extend either the `evaluators.DummyEvaluator` or `evaluators.MemEvaluator` class.

---

## Run on your own dataset

GLocalX has a strict format on input data. It accepts tabular datasets and binary classification tasks. You can find a dummy example for each of these formats in `/data/dummy/`.

#### Rules [`/data/dummy/dummy_rules.json`]

Local rules are to be stored in a `JSON` format:

```json
[
    {"22": [30.0, 91.9], "23": [-Infinity, 553.3], "label": 0},
    ...
]
```

Each rule in the list is a dictionary with an arbitrary (greater than 2) premises. The rule prediction ({0, 1}) is stored in the key `label`. Premises on features are stored according to their ordering and bounds: in the above, `"22": [-Infinity, 91.9]` indicates the premise "feature number 22 has value between 30.0 and 91.9".

#### Black boxes [`/data/dummy/dummy_model.h5`]

Black boxes (if used) are to be stored in a `hdf5` format if given through command line. If given programmatically instead, it suffices that they implement the `Predictor` interface:

```python
class Predictor:
    @abstractmethod
    def predict(self, x):
        pass
```

when called to predict `numpy.ndarray:x` the predictor shall return its predictions in a `numpy.ndarray` of integers.

#### Training data[`/data/dummy/dummy_dataset.csv`]

Training data is to be stored in a csv, comma-separated format with features names as header. The classification labels should have feature name `y`.

---

## Docs

You can find the software documentation in the `/html/` folder and a powerpoint presentation on GLocalX can be found [here](https://docs.google.com/presentation/d/12Nv2MRlvpQfwk9A8TeN6QQwnVUKgE00V-ZWGS2FV5p8/edit?usp=sharing).

The work is currently under submission, we'll update with a citable reference and online documentation as soon as it is published.



---

## Useful functions & Overrides

#### Callbacks

The `fit()` function provides a `callbacks` parameter to add any callbacks you desire to be invoked every `callbacks_step` iterations. The callback should implement the `callbacks.Callback` interface. You can find the set of parameters available to the callback in `glocalx.GLocalX.fit()`.

#### Extending the `merge` function

To override the merge function, simply extend the `glocalx.GLocalX` object and override the `merge` function with the following signature:

```python
merge(self, A:set, B:set, x:numpy.ndarray, y:numpy.ndarray, ids:numpy.ndarray)
```

where `A` and `B` are the sets of `models.Rule` you are merging, `x` is the training data, `y` are the training labels and `ids` are the batch ids. The ids are used by the `MemEvaluator` to store pre-computed results.

#### Extending the `distance` function

The `distance` between explanations is computed by the `evaluators.Evaluator`  objects. To override it, override either the `evaluators.DummyEvaluator` or `evaluators.MemEvaluator` object with the following signature:

```python
distance(self, A:set, B:set, x:numpy.ndarray, ids:numpy.ndarray)
```

where `A`, `B` are the two (sets of) explanation(s), `x` is the training data and `ids` are the ids for the current batch.