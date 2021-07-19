# paccmann_kinase_binding_residues
[![build](https://github.com/PaccMann/paccmann_kinase_binding_residues/workflows/build/badge.svg)](https://github.com/PaccMann/paccmann_kinase_binding_residues/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


Drug-protein affinity prediction - A comparison of active site residues and full protein sequences:

![Summary GIF](https://github.com/PaccMann/paccmann_kinase_binding_residues/blob/master/assets/summary.gif "Summary GIF")


## Description
This repository facilitates the reproduction of the experiments conducted in the paper [`Active site sequence representation of human kinases outperforms full sequence for affinity prediction and inhibitor generation: 3D effects in a 1D model`](https://doi.org/10.33774/chemrxiv-2021-np7xj). We provide scripts to:
1. Train and evaluate the BimodalMCA for drug-protein affinity prediction
2. Evaluate the bimodal KNN affinity predictor either in a CV setting or on a plain train/test script
3. Optimize a SMILES- or SELFIES-based molecular generative model to produce molecules with high binding affinities for a protein of interest (affinity is predicted with the KNN model).

### Data
The preprocessed BindingDB data (CV and test data for ligand split and kinase split, data used for pretraining and affinity optimization) can be accessed on this [Box link](https://ibm.biz/active_site_data).


## Installation

The core functionality of this repo is provided in the `pkbr` package which can be installed (in editable mode) by `pip install -e .`.
In order to execute the example scripts, we recommend setting up a conda environment:


```console
conda env create -f conda.yml
conda activate pkbr
```
Afterwards you can execute the scripts to run the KNN, the BiMCA or the affinity optimization.

## Code examples
All examples assume that you downloaded the data from [this Box link](ibm.biz/active_site_data) and stored it in a folder `data` in the root of this repo.

### Running the KNN affinity predictor in a cross validation
```sh
python3 scripts/knn_cv.py -d data/ligand_split -tr train.csv -te validation.csv \
-f 10 -lp data/ligands.smi -kp data/human_kinases_active_site.smi -r my_cv_results
```

### Running KNN in single train/test split
```sh
python3 scripts/knn_test.py -t data/ligand_split/fold_0/validation.csv -test data/ligand_split/fold_0/test.csv \
-tlp data/ligands.smi -telp data/ligands.smi -kp data/human_kinases_active_site.smi -r my_results.csv
```

### Training the BiMCA model
```sh
python3 scripts/bimca_train.py \
	data/ligand_split/fold_0/train.csv data/ligand_split/fold_0/validation.csv data/ligand_split/fold_0/test.csv \
	human-kinase-alignment data/human_kinases_active_site.smi data/ligands.smi data/smiles_vocab.json \
	models config/active_site.json -n my_as_model
```

### Evaluating the BiMCA model
```sh
python3 scripts/bimca_test.py \
data/ligand_split/fold_0/validation.csv human-kinase-alignment data/human_kinases_sequence.smi data/ligands.smi \
path_to_your_trained_model

```

### Affinity optimization with SMILES generator
To execute this part you need to utilize the pretrained SMILES/SELFIES VAE stored under `data/models`
```sh
python scripts/gp_generation_smiles_knn.py \
    data/models/smiles_vae data/affinity_optimization/bindingdb_all_kinase_active_site.csv \
    data/affinity_optimization/example_active_site.smi \
    smiles_active_site_generator/ -s 42 -t 0.85 -r 10 -n 50 -s 42 -i 40 -c 80
```

### Affinity optimization with SELFIES generator
```sh
python scripts/gp_generation_selfies_knn.py \
    data/models/selfies_vae data/affinity_optimization/bindingdb_all_kinase_sequence.csv \
    data/affinity_optimization/example_sequence.smi \
    selfies_sequence_generator/ -s 42 -t 0.85 -r 10 -n 50 -s 42 -i 40 -c 80
```


## Citation
If you use this repo in your projects, please temporarily cite the following:

```bib
@article{born2021active,
  title={Active site sequence representation of human kinases outperforms full sequence for affinity prediction and inhibitor generation: 3D effects in a 1D model},
  author={Born, Jannis and Huynh, Tien and Stroobants, Astrid and Cornell, Wendy and Manica, Matteo},
  publisher={ChemRxiv},
  doi={10.33774/chemrxiv-2021-np7xj},
  year={2021}
}
```
