import os
import warnings
from time import time
from typing import Dict, Optional

import numpy as np
import pandas as pd
from Levenshtein import distance as levenshtein_distance
from pytoda.smiles.transforms import Canonicalization
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error
from tqdm import tqdm


def knn(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    k: int = 1,
    approx: bool = False,
    seq_nns: int = 10,
    seq_nn_dict: Optional[Dict] = None,
    return_knn_labels: bool = False,
    verbose: bool = False,
    result_path: str = None,
):
    """
    KNN model for CPI affinity prediction, intended for usage in CV setting or simple
    train-test split.

    Applies KNN regression using as similarity the length-normalized Levensthein
    distance between protein sequences and FP similarity of ligands.
    NOTE: Can be used out of the box for binary classification.
    NOTE: Turning the `approx` flag on will yield an approximate KNN making one
    assumption: If the protein sequence and/or the ligand is present in the training
    data, then the search will be restricted to samples where this this ligand, this
    sequence or one of the seq_nns nearest neighbors of the sequence is part of.

    Predictions conceptually correspond to the predict_proba method of
    sklearn.neighbors.KNeighborsClassifier.

    Args:
        train_df (pd.DataFrame): DF with training samples in rows. Columns are:
            'ligand_name', 'SMILES', 'sequence_id', 'sequence' and 'label'.
        test_df (pd.DataFrame): DF with testing samples in rows. Columns are:
            'ligand_name', 'SMILES', 'sequence_id', 'sequence'.
        k (int, optional): Hyperparameter for KNN classification. Defaults to 1.
        approx (bool, optional): Whether the approximate KNN is used instead.
            Defaults to False. If switched on, the NN search is restricted to a subset
            of training data where either:
                (1) ligand is identical to the ligand of interest OR
                (2) protein is identical to the protein of interest OR
                (3) protein is one of the `seq_nns` nearest neighbors of POI.
            Recommended especially for large datasets to accelerate inference.
        seq_nns (int, optional): Ignored unless `approx` is True. Defines how many of
            the closest sequence neighbors are considered for search. Defaults to 10.
            Using a value of 0 implies that (3) collapses & search is restricted to
            samples where (1) or (2) holds (not recommended).
        seq_nn_dict (Dict, optional): Ignored unless `approx` is True. If closest
            sequence neighbors were computed beforehand, they can be passed with this
            dict. This can be useful for datasets where protein sequence similarity
            computation is expensive (long sequences or many proteins).
        return_knn_labels (bool, optional): If set, the labels of the K nearest
            neighbors are also returned.
        verbose (bool, optional): Verbosity level. Defaults to False.
        result_path (str, optional): Where to save results. Should be a *filepath* name,
            ending on `.csv` where the csv with the final predictions will be stored.
    """
    column_names = ["SMILES", "ligand_name", "sequence", "sequence_id", "label"]
    assert isinstance(train_df, pd.DataFrame)
    assert isinstance(test_df, pd.DataFrame)

    assert all([train_df.columns.__contains__(a) for a in column_names]), "Missing cols"
    assert all([test_df.columns.__contains__(a) for a in column_names]), "Missing cols"

    if approx and seq_nns == 0:
        warnings.warn(
            "Approximate KNN was indicated but seq_nns is set to 0. In this case search"
            " is restricted to samples where either protein or ligand is identical."
        )

    # Compute FPs of training data:
    print("Computing fingerprints, removing invalid SMILES...")
    train_fp_dict = get_fp_dict(train_df)
    test_fp_dict = get_fp_dict(test_df)

    train_df = train_df[train_df.ligand_name.isin(train_fp_dict.keys())]
    test_df = test_df[test_df.ligand_name.isin(test_fp_dict.keys())]

    # Will store computed distances to avoid re-computation
    tani_dict = {}
    lev_dict = {}

    # Compute NN-Dict for sequences
    if approx and seq_nn_dict is None:
        seq_nn_dict = {}

        seq_test_df = test_df.drop_duplicates(subset="sequence_id")
        train_seqs = train_df.drop_duplicates(subset="sequence_id")["sequence"].values
        train_seq_ids = train_df.drop_duplicates(subset="sequence_id")[
            "sequence_id"
        ].values

        if seq_nns > 0:
            print("Finding neighbors of proteins...")
            print(f"Proteins for train {len(train_seqs)}, test: {len(seq_test_df)}")
            for seq, seq_id in tqdm(
                zip(seq_test_df["sequence"], seq_test_df["sequence_id"])
            ):
                ls = len(seq)
                distances = np.array(
                    [levenshtein_distance(seq, seq_t) / ls for seq_t in train_seqs]
                )
                # Holds the sequence IDs for the ${seq_nns} nearest protein neighbors
                seq_nn_dict[seq_id] = list(
                    train_seq_ids[np.argsort(distances)[:seq_nns]]
                )
        elif approx:
            for seq_id in seq_test_df["sequence_id"].unique():
                seq_nn_dict[seq_id] = list()

    print(f"Prediction starts. {len(test_df)} samples will now be processed.")

    predictions, knn_labels, smiles, seqs, smiles_id, seq_ids = [], [], [], [], [], []
    nn_smiles, nn_seqs, labels, nn_ligand_name, nn_label = [], [], [], [], []

    ti = time()
    for idx_loc, test_sample in tqdm(test_df.iterrows()):

        idx = test_df.index.get_loc(idx_loc)
        if verbose and idx % 10 == 0:
            print(f"Idx {idx}/{len(test_df)}", time() - ti)
            ti = time()

        seq = test_sample["sequence"]
        ls = len(seq)
        fp = test_fp_dict[test_sample["ligand_name"]]
        ln = test_sample["ligand_name"]

        if ln not in tani_dict.keys():
            tani_dict[ln] = {}

        def get_mol_dist(train_ligand_name):
            if train_ligand_name in tani_dict[ln].keys():
                return tani_dict[ln][train_ligand_name]
            else:
                d = flipper(
                    DataStructs.FingerprintSimilarity(
                        fp, train_fp_dict[train_ligand_name]
                    )
                )
                tani_dict[ln][train_ligand_name] = d
                return d

        sn = test_sample["sequence_id"]
        if sn not in lev_dict.keys():
            lev_dict[sn] = {}

        def get_seq_dist(sid, sequence):
            if sid in lev_dict[sn].keys():
                return lev_dict[sn][sid]
            else:
                d = levenshtein_distance(seq, sequence) / max(ls, len(sequence))
                lev_dict[sn][sid] = d
                return d

        # Restrict search to samples with same ligand, same seq or 5 NNs of the sequence
        if approx:
            tdf = train_df[
                (train_df["ligand_name"] == ln)
                | (train_df["sequence_id"] == sn)
                | train_df["sequence_id"].isin(seq_nn_dict[sn])
            ]
        else:
            tdf = train_df

        train_sids = list(tdf["sequence_id"])
        train_seqs = list(tdf["sequence"])
        train_ln = list(tdf["ligand_name"])

        mol_dists = np.array(list(map(get_mol_dist, train_ln)))
        seq_dists = np.array(list(map(get_seq_dist, train_sids, train_seqs)))

        knns = np.argsort(np.array(mol_dists) + np.array(seq_dists))[:k]
        _knn_labels = np.array(tdf["label"].values)[knns]
        nn_smiles.append(np.array(tdf["SMILES"])[knns[0]])
        nn_seqs.append(np.array(tdf["sequence"])[knns[0]])
        nn_ligand_name.append(np.array(tdf["ligand_name"])[knns[0]])
        nn_label.append(np.array(tdf["label"])[knns[0]])
        predictions.append(np.mean(_knn_labels))
        knn_labels.append(_knn_labels)
        smiles.append(test_sample["SMILES"])
        smiles_id.append(ln)
        seqs.append(seq)
        seq_ids.append(test_sample["sequence_id"])
        labels.append(test_sample["label"])

        if result_path is not None and idx % 100 == 0 and idx > 0 and verbose:

            df = pd.DataFrame(knn_labels)
            df.insert(0, "Labels", labels)
            df.insert(0, "NN-Labels", nn_label)
            df.insert(0, "NN-SMILES", nn_smiles)
            df.insert(0, "NN-Ligand-Name", nn_ligand_name)
            df.insert(0, "NN-Seqs", nn_seqs)
            df.insert(0, "SMILES", smiles)
            df.insert(0, "sequence", seqs)
            df.insert(0, "ligand_name", smiles_id)
            df.insert(0, "sequence_id", seq_ids)
            df.to_csv(os.path.join(os.path.dirname(result_path), f"knn_{idx}.csv"))

    rmse = np.sqrt(mean_squared_error(labels, nn_label))
    pearson = pearsonr(labels, nn_label)[0]
    spearman = spearmanr(labels, nn_label)[0]

    print(f"NN --> RMSE = {rmse}, Pearson = {pearson}, Spearman = {spearman}")

    if result_path is not None:
        df = pd.DataFrame(knn_labels)
        df.insert(0, "Labels", labels)
        df.insert(0, "NN-Labels", nn_label)
        df.insert(0, "NN-SMILES", nn_smiles)
        df.insert(0, "NN-Ligand-Name", nn_ligand_name)
        df.insert(0, "NN-Seqs", nn_seqs)
        df.insert(0, "SMILES", smiles)
        df.insert(0, "sequence", seqs)
        df.insert(0, "ligand_name", smiles_id)
        df.insert(0, "sequence_id", seq_ids)
        df.to_csv(result_path)

    return (predictions, knn_labels) if return_knn_labels else predictions


class KnnPredictor:
    """
    Callable KNN Predictor (intended for usage during affinity optimization)
    """

    def __init__(
        self,
        train_df: pd.DataFrame,
        seq_nns: int = -1,
        radius: int = 2,
        bits: int = 2048,
    ):
        """
        Args:
            train_df (pd.DataFrame): Pandas df with columns 'SMILES', 'ligand_name',
                'sequence', 'sequence_id' and 'label'.
            seq_nns (int):  Number of nearest neighbor of the target sequence to which
                the search is restricted. Defaults to -1, meaning all source sequences
                are used as in the KNN formulation. Using a positive value implies to
                use the approximate KNN. Increasing this value will yield better
                approximations, but inference will be slower (since more neighbors are
                searched). The approximate KNN is especially beneficial for large
                datasets and repeated queries for the same protein.
            radius (int, optional): Radius for MorganFP. Defaults to 2 (i.e., ECFP4).
            bits (int, optional): Number of bits for FP. Defaults to 2048.
        """

        assert isinstance(train_df, pd.DataFrame)
        column_names = ["SMILES", "ligand_name", "sequence", "sequence_id", "label"]
        assert all(
            [train_df.columns.__contains__(a) for a in column_names]
        ), f"At least one of the required columns {column_names} is missing."

        self.train_df = train_df
        self.radius = radius
        self.bits = bits
        self.canonicalize = Canonicalization()
        self.seq_nns = seq_nns
        self.setup()

    def setup(self):
        """
        Sets up class for accelerated inference. E.g., computes the fingerprints of
            all SMILES.
        """

        # Compute FPs of training data:
        print("Computing fingerprints. Make sure all SMILES are valid...")
        self.train_fp_dict = get_fp_dict(
            self.train_df, radius=self.radius, bits=self.bits
        )
        self.train_df = self.train_df[
            self.train_df.ligand_name.isin(self.train_fp_dict.keys())
        ]

        # Setup dictionaries which will store distances
        self.tani_dict = {}
        self.lev_dict = {}
        self.seq_nn_dict = {}

        self.train_sids = list(self.train_df["sequence_id"])
        self.train_seqs = list(self.train_df["sequence"])
        self.train_ln = list(self.train_df["ligand_name"])
        self.labels = self.train_df["label"].values

    def __call__(
        self, smiles: str, protein: str, protein_accession: str, k: float = 13
    ) -> float:
        """
        Perform a KNN prediction

        Args:
            smiles (str): SMILES of the moolecule of interest
            protein (str): AA sequence of the protein of interest (might be active site
                residues only).
            protein_accession (str): Protein identifier (e.g., UniProt ID).
            k (float, optional): k for the KNN algorithm. Defaults to 13.

        Returns:
            float: The affinity prediction.
        """

        ls = len(protein)
        smi = self.canonicalize(smiles)
        fp = AllChem.GetMorganFingerprintAsBitVect(
            Chem.MolFromSmiles(smiles), radius=self.radius, nBits=self.bits
        )

        if smi not in self.tani_dict.keys():
            self.tani_dict[smi] = {}

        def get_mol_dist(train_ligand_name: str) -> float:
            """Get Tanimoto-based distance of a training molecule with ligand of interest.

            Args:
                train_ligand_name (str): Name of training ligand.

            Returns:
                float: Distance based on Tanimoto similarity to ligand of interest.
            """
            if train_ligand_name in self.tani_dict[smi].keys():
                return self.tani_dict[smi][train_ligand_name]
            else:
                d = flipper(
                    DataStructs.FingerprintSimilarity(
                        fp, self.train_fp_dict[train_ligand_name]
                    )
                )
                self.tani_dict[smi][train_ligand_name] = d
                return d

        if protein_accession not in self.lev_dict.keys():
            self.lev_dict[protein_accession] = {}

        if self.seq_nns > 0:
            tdf = self.get_sub_df(protein, protein_accession, smi)
        else:
            tdf = self.train_df

        def get_seq_dist(seq_id: str, sequence: str) -> float:
            """Get normalized Levenshtein distance of a training protein to the protein
                of interest.

            Args:
                seq_id (str): Protein identifier of training protein.
                sequence (str): AA sequence of training protein.

            Returns:
                float: Length-normalized Levenshtein distance.
            """
            if seq_id in self.lev_dict[protein_accession].keys():
                return self.lev_dict[protein_accession][seq_id]
            else:
                d = levenshtein_distance(protein, sequence) / max(ls, len(sequence))
                self.lev_dict[protein_accession][seq_id] = d
                return d

        mol_dists = np.array(list(map(get_mol_dist, list(tdf["ligand_name"]))))
        seq_dists = np.array(
            list(map(get_seq_dist, list(tdf["sequence_id"]), list(tdf["sequence"])))
        )

        knns = np.argsort(np.array(mol_dists) + np.array(seq_dists))[:k]
        dists = tdf["label"].values[knns]
        return dists.mean()

    def get_sub_df(
        self, protein: str, protein_accession: str, smi: str
    ) -> pd.DataFrame:
        """Retrieve subset of training data where either:
            (1) ligand is identical to the ligand of interest
            (2) protein is identical to the protein of interest
            (3) proteiin is one of the `seq_nns` nearest neighbors of POI.

        Args:
            protein (str): AA sequence of the protein of interest.
            protein_accession (str): Identifier of the protein of interest.
            smi (str): SMILES of the ligand of interest.

        Returns:
            pd.DataFrame: Subset of original training data.
        """

        seq_train_df = self.train_df.drop_duplicates(subset="sequence_id")
        ls = len(protein)

        if protein_accession not in self.seq_nn_dict.keys():
            distances = np.array(
                [
                    levenshtein_distance(protein, seq_test) / ls
                    for seq_test in seq_train_df["sequence"]
                ]
            )
            # Holds the sequence IDs for the ${seq_nns} nearest neighbors in proteins.
            self.seq_nn_dict[protein_accession] = list(
                seq_train_df["sequence_id"].values[
                    np.argsort(distances)[: self.seq_nns]
                ]
            )

        return self.train_df[
            (self.train_df["ligand_name"] == smi)
            | (self.train_df["sequence_id"] == protein_accession)
            | self.train_df["sequence_id"].isin(self.seq_nn_dict[protein_accession])
        ]


"""
Some KNN helper functions
"""


def get_fp_dict(df: pd.DataFrame, radius: int = 2, bits: int = 2048) -> Dict:
    """
    Args:
        df (pd.DataFrame): Dataframe with columns SMILES and ligand_name.
        radius (int, optional): Radius for MorganFP. Defaults to 2 (i.e., ECFP4).
        bits (int, optional): Number of bits for FP. Defaults to 2048.

    Returns:
        Dict: Dictionary containing ligand names as keys and FP vectors as values.
    """
    ligand_names = np.array(
        df.drop_duplicates(subset=["SMILES", "ligand_name"])["ligand_name"].values
    )
    smis = df.drop_duplicates(subset=["SMILES", "ligand_name"])["SMILES"].values
    fps = []
    invalid_idx = []
    for idx, smi in enumerate(smis):
        try:
            fps.append(
                AllChem.GetMorganFingerprintAsBitVect(
                    Chem.MolFromSmiles(smi), radius=radius, nBits=bits
                )
            )
        except Exception:
            invalid_idx.append(idx)
    ligand_names = np.delete(ligand_names, invalid_idx)
    if len(invalid_idx) > 0:
        print(
            f"FP computed. Removed {len(invalid_idx)} invalid mols (total {len(fps)})."
        )
    return dict(zip(ligand_names, fps))


def flipper(x: float) -> float:
    """Convert Tanimoto similarity into a distance

    Args:
        x (float): Tanimoto similarity

    Returns:
        float: Distance
    """
    return x * -1 + 1
