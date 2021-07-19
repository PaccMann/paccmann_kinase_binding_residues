"""PDB file utils."""
import pandas as pd
from pytoda.proteins.processing import IUPAC_CODES

iupac_codes_extension = {
    residue.upper(): aa_letter for residue, aa_letter in IUPAC_CODES.items()
}
IUPAC_CODES.update(iupac_codes_extension)
IUPAC_CHARACTER_SET = set(IUPAC_CODES.values())


def read_aa_sequence_attention(
    sequence_attention_filepath: str, column: str
) -> pd.DataFrame:
    """Read AA sequence attention.

    Args:
        sequence_attention_filepath: sequence attention file.
        column: column containing attention values.

    Returns:
        the AAs mapped to the attention values.
    """
    return pd.read_csv(sequence_attention_filepath)[["sequence", column]].reindex()
