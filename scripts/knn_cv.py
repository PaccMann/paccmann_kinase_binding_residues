#!/usr/bin/env python3
"""
Evaluate lazy KNN baseline predictor in a Cross-Validation setting.
Assumes that ligand_fp and kinase_fp contain all molecules and kinases that
are used in training and testng dataset.

"""
import argparse
import json
import logging
import os
import sys
from typing import List

import numpy as np
import pandas as pd
from pkbr.knn import knn
from pytoda.files import read_smi
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error

# setup logging
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

parser = argparse.ArgumentParser()
parser.add_argument(
    "-d",
    "--data_path",
    type=str,
    help="Path to the folder where the fold-specific data is stored",
)
parser.add_argument(
    "-tr",
    "--train_name",
    type=str,
    help="Name of train files stored _inside_ the fold-specific folders."
    "Needs to have column names as specified in column_names",
)
parser.add_argument(
    "-te",
    "--test_name",
    type=str,
    help="Name of test files stored _inside_ the fold-specific folders"
    "Needs to have column names as specified in column_names",
)
parser.add_argument(
    "-f",
    "--num_folds",
    type=int,
    help="Number of folds. Folders should be named fold_0, fold_1 etc.  ",
)
parser.add_argument(
    "-lp", "--ligand_fp", type=str, help="Path to the ligand data (.smi)"
)
parser.add_argument(
    "-kp", "--kinase_fp", type=str, help="Path to the kinase data (.smi)"
)
parser.add_argument(
    "-r", "--results_path", type=str, help="Path to folder to store results"
)
parser.add_argument(
    "-k",
    "--k",
    type=int,
    help=(
        "K for the KNN classification. Note that classification reports are generated"
        " for all odd x: 1<= x <= k"
    ),
    default=25,
)
parser.add_argument(
    "-c",
    "--column_names",
    type=List[str],
    help="Columns in train/test files. Order: Name of ligand, name of protein, label",
    default=["ligand_name", "uniprot_accession", "pIC50"],
)


def main(
    # fmt: off
    data_path, train_name, test_name, num_folds, ligand_fp, kinase_fp, results_path, k, 
    column_names
):
    # fmt: on
    logger = logging.getLogger("knn_prediction")
    logging.getLogger("matplotlib.font_manager").disabled = True

    rename_dict = {
        column_names[0]: "ligand_name",
        column_names[1]: "sequence_id",
        column_names[2]: "label",
    }

    ligands = read_smi(ligand_fp, names=["data"])
    kinases = read_smi(kinase_fp, names=["data"])

    # Create right amount of empty lists
    all_results = np.empty((np.ceil(k / 2).astype(int), 0)).tolist()

    for fold in range(num_folds):
        train_data = pd.read_csv(
            os.path.join(data_path, f"fold_{fold}", train_name), index_col=0
        )[column_names]
        test_data = pd.read_csv(
            os.path.join(data_path, f"fold_{fold}", test_name), index_col=0
        )[column_names]
        train_data.rename(columns=rename_dict, inplace=True)
        test_data.rename(columns=rename_dict, inplace=True)

        try:
            test_data["SMILES"] = ligands.loc[test_data["ligand_name"]].values
            test_data["sequence"] = kinases.loc[test_data["sequence_id"]].values
            train_data["SMILES"] = ligands.loc[train_data["ligand_name"]].values
            train_data["sequence"] = kinases.loc[train_data["sequence_id"]].values
        except KeyError:
            raise KeyError(
                "Either colum names in the dataframes are incorrect or"
                "some SMILES or AAs were not found in the .smi files."
            )

        logger.info(f"Fold {fold}: Data loaded")

        # Classify data
        predictions, knn_labels = knn(
            train_data,
            test_data,
            k=k,
            return_knn_labels=True,
            verbose=False,
            approx=True,
            seq_nns=15,
            result_path=os.path.join(results_path, f"knn_all_fold_{fold}.csv"),
        )
        logger.info(f"Fold {fold}: Predictions done")

        labels = list(test_data["label"].values)

        for idx, _k in enumerate(range(k, 0, -2)):

            # Overwrite predictions to match _k instead of k
            predictions = [np.mean(sample_knns[:_k]) for sample_knns in knn_labels]

            # Compute metrics
            rmse = np.sqrt(mean_squared_error(labels, predictions))
            pearson = pearsonr(labels, predictions)[0]
            spearman = spearmanr(labels, predictions)[0]

            # Create and save json
            results = {"RMSE": rmse, "Pearson": pearson, "Spearman": spearman}

            # Write data
            os.makedirs(os.path.join(results_path, f"k{_k}"), exist_ok=True)
            with open(
                os.path.join(results_path, f"k{_k}", f"fold_{fold}_report.json"),
                "w",
            ) as f:
                json.dump(results, f)
            all_results[idx].append(results)

            # Save predictions
            pd.DataFrame({"labels": labels, "predictions": predictions}).to_csv(
                os.path.join(results_path, f"k{_k}", f"fold_{fold}_results.csv")
            )
        logger.info(f"Fold {fold}: Reports generated and saved.")

    # Generate reports across folds
    for idx, _k in enumerate(range(k, 0, -2)):
        df = pd.DataFrame(all_results[idx])
        df.index = range(num_folds)
        df.loc["mean"] = df.mean()
        df.loc["std"] = df.std()
        df.to_csv(os.path.join(results_path, f"knn_{_k}_cv_results.csv"))

    logger.info("Done, shutting down.")


if __name__ == "__main__":
    args = parser.parse_args()
    main(
        args.data_path,
        args.train_name,
        args.test_name,
        args.num_folds,
        args.ligand_fp,
        args.kinase_fp,
        args.results_path,
        args.k,
        args.column_names,
    )
