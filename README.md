# GLocalX - Global through Local Explainability

Explanations come in two forms: local, explaining a single model prediction, and global, explaining all model predictions. The Local to Global (L2G) problem consists in bridging these two family of explanations. Simply put, we generate global explanations by merging local ones.

## The algorithm

Local and global explanations are provided in the form of decision rules:

```
age < 40, income > 50000, status = married, job = CEO ⇒ grant loan application
```

This rule describes the rationale followed given by an unexplainable model to grant the loan application to an individual younger than 40 years-old, with an income above 50000$, married and currently working as a CEO.

## Running the code

### Python interface

Check the `main.py` for usage example.

### Lore rules creation using LORE-ext library

Script `./rule_loaders/lore_rule_generator.py` provides a functionality to create LORE rules. You can adjust the settings for rules calculation inside.

You need to provide with: 
- data (and preprocess it before it, check the ipynb for example)
- trained model (black box), otherwise it will be trained from the data

Then rules are stored in `./data/lore_rules`, which then are used for GlocalX (e.g. main.py)

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

  --strict_join                  Use to use high concordance.
  --strict_cut                   Use to use the strong cut.
  --global_direction             Use to use the global search direction.
  -d, --debug INTEGER            Debug level.
  --help                         Show this message and exit.

```

A minimal run simply requires a set of local input rules, a data set and (optionally) a black box:

```shell script
python3.8 apy.py data/dummy/dummy_rules.json data/dummy/dummy_dataset.csv --oracle data/dummy/dummy_model.h5 --name dummy --batch 2
```

If you are interested, the folder `data/dummy/` contains a dummy example.
You can run it with
```shell script
python3.8 apy.py my_rules.json training_set.csv --oracle my_black_box.h5 \
		--name my_black_box_explanations
```

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
- `--strict_join` to use a more stringent `join`
- `--strict_cut` to use a more stringent `cut`
- `--global_direction` Use to evaluate merges on the whole validation set
- `--debug` Debug level: the higher, the less messages shown

## Run on your own dataset

GLocalX has a strict format on input data. It accepts tabular datasets and binary classification tasks. You can find a dummy example for each of these formats in `/data/dummy/`.

## Local rules

We provide an integration with [Lore](https://github.com/riccotti/LORE) rules through the `loaders` module.
To convert Lore rules to GLocalX rules, use the provided `loaders.lore.lore_to_glocalx(json_file, info_file)` function. `json_file` should be the path to a json with Lore's rules, and `info_file` should be the path to a JSON dictionary holding the class values (key `class_names`) and a list of the features' names (key `feature_names`).
You can find dummy examples of the info file in [data/loaders/adult_info.json](https://github.com/msetzu/glocalx/blob/main/data/loaders/adult_info.json).


### Rules [`/data/dummy/dummy_rules.json`]

Local rules are to be stored in a `JSON` format:

```json
[
    {"22": [30.0, 91.9], "23": [-Infinity, 553.3], "label": 0},
    ...
]
```

Each rule in the list is a dictionary with premises. The rule prediction stored in `label`. Premises on features are stored according to their ordering and bounds: in the above, `"22": [-Infinity, 91.9]` indicates the premise "feature number 22 has value between 30.0 and 91.9".

### Black boxes [`/data/dummy/dummy_model.h5`]

Black boxes (if used) are to be stored in a `hdf5` format if given through command line. If given programmatically instead, it suffices that they implement the `Predictor` interface:

```python
class Predictor:
    @abstractmethod
    def predict(self, x):
        pass
```

when called to predict `numpy.ndarray:x` the predictor shall return its predictions in a `numpy.ndarray` of integers.

### Training data[`/data/dummy/dummy_dataset.csv`]

Training data is to be stored in a csv, comma-separated format with features names as header. The classification labels should have feature name `y`.

---
## Serialization and deserialization

You can dump to disk and load `GLocalX` instances and their output with the `Rule` object and the `serialization` module:

```python
from models import Rule
import serialization

rules_only_json = 'input_rules.json'
run_file = 'my_run.glocalx.json'

# Load input rules
rules = Rule.from_json(rules_only_json)

# Load GLocalX output
glocalx_output = serialization.load_run(run_file)

# Load a GLocalX instance from a set of rules, regardless of whether they come from an actual run or not!
# From a GLocalX run...
glocalx = serialization.load_glocalx(run_file, is_glocalx_run=True)
# From a 
glocalx = serialization.load_glocalx(rules_only_json, is_glocalx_run=False)
