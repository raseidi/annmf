# Approximate Nearest Neighbor Meta-Features 

Implementation of the ann-based meta-features extractor, described in [this paper](https://www.sciencedirect.com/science/article/abs/pii/S0306437922001016).

---

## Usage

### Installing

Via conda: 

```bash
conda env create -f env.yml 
```

or via pip:
```bash
pip install requirements.txt
```

### Usage

Run the code: 
```bash
python3 extract.py --dataset-path <DATASET_PATH> --nr-inst <NUMBER_OF_INSTANCES>
```

Parameters: 

```
--dataset-path: directory for you dataset (csv format)
--nr-inst: number of instances to be sampled from the original dataset (in case the dataset is too large)
```