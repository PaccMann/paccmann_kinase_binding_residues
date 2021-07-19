"""Data processing utilities."""
import re
import sys
from typing import List

import pandas as pd

active_site_split = re.compile(r",\s+")
is_number = re.compile(r"([<>])?\s*([\d+|\d+\.\d+]+)")


def parse_active_site_data_line(line: str) -> List[str]:
    """
    Parse active site data line.

    Args:
        line (str): a line from the active site data file.

    Returns:
        List[str]: a list containing identifiers and the sequence.
    """
    identifiers, sequence = re.split(r",\s+", line.strip(">\n"))
    return identifiers.split() + [sequence]


def read_active_site_data(filepath: str) -> pd.DataFrame:
    """
    Read active site data.

    Args:
        filepath (str): acitve site data filepath.

    Returns:
        pd.DataFrame: a data frame containing active site data.
    """
    with open(filepath) as fp:
        data = [parse_active_site_data_line(line) for line in fp]
    # get data frame
    df = pd.DataFrame(
        data, columns=["identifier", "uniprot", "kinase", "accession", "sequence"]
    ).set_index("identifier")
    # filter out the ones with lower sequence range
    df["sequence_ending"] = [int(identifier.split("-")[-1]) for identifier in df.index]
    to_keep = (
        df.groupby(["kinase"])["sequence_ending"].transform(max)
        == df["sequence_ending"]
    )
    return df[to_keep].drop(["sequence_ending"], axis=1)


def str2float(string_value: str) -> float:
    """
    Convert a string to a float.

    Args:
        string_value (str): string value representing a float.

    Returns:
        float: the converted value.
    """
    qualifier, number = is_number.search(string_value).groups()
    number = float(number)
    number += sys.float_info.min * (1 if qualifier == ">" else -1) if qualifier else 0.0
    return number


def read_binding_db_data(binding_db_filepath: str) -> pd.DataFrame:
    """
    Read BindingDB data with a focus on IC50.

    Args:
        binding_db_filepath (str): path to the BindingDB dump.

    Returns:
        pd.DataFrame: the processed data from BindingDB.
    """
    data = pd.read_csv(
        binding_db_filepath,
        sep="\t",
        error_bad_lines=False,
        warn_bad_lines=True,
        engine="python",
    )
    data = data[~data["IC50 (nM)"].isna()]
    data["ic50_nM_numerical"] = [str2float(value) for value in data["IC50 (nM)"]]
    sequence_strings = pd.Series(
        [
            isinstance(sequence, str)
            for sequence in data["BindingDB Target Chain  Sequence"]
        ]
    )
    smiles_strings = pd.Series(
        [isinstance(smiles, str) for smiles in data["Ligand SMILES"]]
    )
    single_chain = (
        data["Number of Protein Chains in Target (>1 implies a multichain complex)"]
        <= 1
    )
    filtered = data[sequence_strings & smiles_strings & single_chain].dropna(
        subset=["Ligand SMILES", "BindingDB Target Chain  Sequence"], axis=0
    )
    processed_data = filtered[
        [
            "Ligand SMILES",
            "BindingDB Target Chain  Sequence",
            "BindingDB Ligand Name",
            "Target Name Assigned by Curator or DataSource",
            "Target Source Organism According to Curator or DataSource",
            "UniProt (SwissProt) Primary ID of Target Chain",
            "ic50_nM_numerical",
        ]
    ]
    processed_data.columns = [
        "smiles",
        "sequence",
        "ligand_name",
        "sequence_name",
        "organism",
        "uniprot_accession",
        "ic50_nM",
    ]
    return processed_data
