#!/usr/bin/env python3
"""
Predict with lazy KNN baseline predictor.

"""
import argparse
import logging
import sys
from typing import List

import pandas as pd
from pkbr.knn import knn
from pytoda.files import read_smi

# setup logging
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

parser = argparse.ArgumentParser()
parser.add_argument(
    "-t", "--train_path", type=str, help="Path to the training samples (.csv)"
)
parser.add_argument(
    "-test", "--test_path", type=str, help="Path to the testing samples (.csv)"
)
parser.add_argument(
    "-tlp", "--train_ligand_fp", type=str, help="Path to the train ligand data (.smi)"
)
parser.add_argument(
    "-telp", "--test_ligand_path", type=str, help="Path to the test ligand data (.smi)"
)
parser.add_argument(
    "-kp", "--kinase_fp", type=str, help="Path to the kinase data (.smi)"
)
parser.add_argument(
    "-r", "--results_path", type=str, help="Path to file to store results (.csv)"
)
parser.add_argument(
    "-k",
    "--k",
    type=int,
    help=(
        "K for the KNN classification. Note that classification reports are generated"
        " for all odd x: 1<= x <= k"
    ),
    default=13,
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
    train_path, test_path, train_ligand_fp, test_ligand_path, kinase_fp, results_path,
    k, column_names
):
    # fmt: on
    logger = logging.getLogger("knn_prediction")
    logging.getLogger("matplotlib.font_manager").disabled = True
    rename_dict = {
        column_names[0]: "ligand_name",
        column_names[1]: "sequence_id",
        column_names[2]: "label",
    }

    train_ligands = read_smi(train_ligand_fp, names=["data"])
    test_ligands = read_smi(test_ligand_path)["SMILES"]
    kinases = read_smi(kinase_fp, names=["data"])

    train_data = pd.read_csv(train_path, index_col=0)[column_names]
    train_data.rename(columns=rename_dict, inplace=True)

    test_data = pd.read_csv(test_path, index_col=None)
    test_data.rename(columns=rename_dict, inplace=True)
    test_data = test_data[test_data.sequence_id.isin(kinases.index)]

    try:
        train_data["SMILES"] = train_ligands.loc[train_data["ligand_name"]].values
        train_data["sequence"] = kinases.loc[train_data["sequence_id"]].values
        test_data["SMILES"] = test_ligands.loc[test_data["ligand_name"]].values
        test_data["sequence"] = kinases.loc[test_data["sequence_id"]].values
    except KeyError:
        raise KeyError("Some SMILES or AAs were not found in the .smi files.")

    seq_nn_dict = None

    # Classify data
    print("Data parsed. Calling KNN Method")
    predictions, knn_labels = knn(
        train_data,
        test_data,
        k=k,
        return_knn_labels=True,
        approx=True,
        verbose=False,
        seq_nns=15,
        result_path=results_path,
        seq_nn_dict=seq_nn_dict,  # This will compute the 15 nearest neighbors of each
        # seq on the fly. Might be time consuming, consider feeding the dict directly.
    )
    logger.info("Done, shutting down.")


if __name__ == "__main__":
    args = parser.parse_args()
    main(
        args.train_path,
        args.test_path,
        args.train_ligand_fp,
        args.test_ligand_path,
        args.kinase_fp,
        args.results_path,
        args.k,
        args.column_names,
    )
