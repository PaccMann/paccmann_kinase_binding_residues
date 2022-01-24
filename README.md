# Active sites outperform full proteins for modeling human kinases
[![Python package](https://github.com/PaccMann/paccmann_kinase_binding_residues/actions/workflows/python-package.yml/badge.svg)](https://github.com/PaccMann/paccmann_kinase_binding_residues/actions/workflows/python-package.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Summary
This repository contains data & code for the JCIM paper: [Active Site Sequence Representations of Human Kinases Outperform Full Sequence Representations for Affinity Prediction and Inhibitor Generation: 3D Effects in a 1D Model](https://pubs.acs.org/doi/10.1021/acs.jcim.1c00889). We study the impact of different protein sequence representations for modeling human kinases. We find that using **active site residues yields superior performance to using full protein sequences for predicting binding affinity**. We also study the difference of active site vs. full sequence on de-novo design tasks. We generate kinase inhibitors directly from protein sequences with our previously developed hybrid-VAE (PaccMann<sup>RL</sup>) but find no major differences between both kinase representations.


## News
- January 2022: We are proud to be featured on [**the JCIM cover**](https://pubs.acs.org/toc/jcisd8/62/2) with _an AI created artwork_! 👉  <img align="right" width="25%" height="25%" src="https://github.com/PaccMann/paccmann_kinase_binding_residues/blob/master/assets/cover.jpg">
- December 2021: Our work has been published in the ACS [**Journal of Chemical Information & Modeling**](https://pubs.acs.org/doi/10.1021/acs.jcim.1c00889). 
- December 2021: The part about binding affinity prediction was presented at the [NeurIPS 2021 workshop on *Machine Learning for Structural Biology*](https://www.mlsb.io) and the [ELLIS Machine Learning for Molecule Discovery workshop](https://moleculediscovery.github.io/workshop2021/) alongside NeurIPS 2021.
- November 2021: Our work has **won** the [🥇 #IOPP best poster award🥇](https://ioppublishing.org/twitter-conference/) in the category *Biomedical engineering* 
- July 2021: A preliminary version of our work was presented at the [Twitter #IOPPposter conference](https://ioppublishing.org/twitter-conference/) (see GIF below ⬇️))

![Summary GIF](https://github.com/PaccMann/paccmann_kinase_binding_residues/blob/master/assets/summary.gif "Summary GIF")


## Description
This repository facilitates the reproduction of the experiments conducted in the JCIM paper [`Active Site Sequence Representations of Human Kinases Outperform Full Sequence Representations for Affinity Prediction and Inhibitor Generation: 3D Effects in a 1D Model`](https://pubs.acs.org/doi/10.1021/acs.jcim.1c00889). We provide scripts to:
1. Train and evaluate the BimodalMCA for drug-protein affinity prediction
2. Evaluate the bimodal KNN affinity predictor either in a CV setting or on a plain train/test script
3. Optimize a SMILES- or SELFIES-based molecular generative model to produce molecules with high binding affinities for a protein of interest (affinity is predicted with the KNN model).

### Data
The preprocessed BindingDB data (CV and test data for ligand split and kinase split, data used for pretraining and affinity optimization) can be accessed on this [Box link](https://ibm.biz/active_site_data). We also release the aligned active site sequences (29 residues) for all kinases. If you use the data, please [cite](#citation) our work.


## Installation

The core functionality of this repo is provided in the `pkbr` package which can be installed (in editable mode) by `pip install -e .`.
In order to execute the example scripts, we recommend setting up a conda environment:


```console
conda env create -f conda.yml
conda activate pkbr
```
Afterwards you can execute the scripts to run the KNN, the BiMCA or the affinity optimization.

## Code examples
All examples assume that you downloaded the data from [this Box link](https://ibm.biz/active_site_data) and stored it in a folder `data` in the root of this repo.

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
If you use this repo or our data in your projects, please cite the following:

```bib
@article{born2022active,
	author = {Born, Jannis and Huynh, Tien and Stroobants, Astrid and Cornell, Wendy D. and Manica, Matteo},
	title = {Active Site Sequence Representations of Human Kinases Outperform Full Sequence Representations for Affinity Prediction and Inhibitor Generation: 3D Effects in a 1D Model},
	journal = {Journal of Chemical Information and Modeling},
	volume = {62},
	number = {2},
	pages = {240-257},
	year = {2022},
	doi = {10.1021/acs.jcim.1c00889},
	note ={PMID: 34905358},
	URL = {https://doi.org/10.1021/acs.jcim.1c00889}
}
```
