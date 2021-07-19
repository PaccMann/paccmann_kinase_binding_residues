"""Testing affinity models."""
import argparse
import json
import os
from collections import OrderedDict

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from loguru import logger
from paccmann_predictor.models.bimodal_mca import BimodalMCA
from paccmann_predictor.utils.utils import get_device
from pytoda.datasets.drug_affinity_dataset import DrugAffinityDataset
from pytoda.smiles.smiles_language import SMILESTokenizer
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error

parser = argparse.ArgumentParser()
parser.add_argument(
    "test_affinity_filepath", type=str, help="path to the drug affinity data for test."
)
parser.add_argument(
    "protein_vocabulary",
    type=str,
    help="protein vocabulary used.",
    choices=["iupac", "human-kinase-alignment"],
)
parser.add_argument("protein_filepath", type=str, help="path to the protein sequences.")
parser.add_argument("smi_filepath", type=str, help="path to the SMILES data.")
parser.add_argument(
    "checkpoint_filepath", type=str, help="path to model root or full checkpoint."
)
parser.add_argument(
    "-s", "--seed", type=int, help="seed for reproducibility.", default=42
)


class AffinityModel(pl.LightningModule):
    def __init__(self, params: dict) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = BimodalMCA(params)

    def forward(self, smiles: torch.Tensor, proteins: torch.Tensor) -> torch.Tensor:
        return self.model.forward(smiles, proteins)[0]

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        smiles, proteins, y = batch
        y_hat = self(smiles, proteins)
        return self.model.loss(y_hat, y)

    def validation_step(self, batch: tuple, batch_idx: int) -> None:
        smiles, proteins, y = batch
        y_hat = self(smiles, proteins)
        loss = self.model.loss(y_hat, y)
        self.log("valid_loss", loss)

    def test_step(self, batch: tuple, batch_idx: int) -> None:
        smiles, proteins, y = batch
        y_hat = self(smiles, proteins)
        loss = self.model.loss(y_hat, y)
        self.log("test_loss", loss)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(
            self.model.parameters(), lr=self.hparams["params"]["lr"]
        )


def run_testing(
    test_affinity_filepath: str,
    protein_vocabulary: str,
    protein_filepath: str,
    smi_filepath: str,
    checkpoint_filepath: str,
    seed: int,
) -> None:
    """Run testing."""
    pl.seed_everything(seed=seed)

    model_dir = os.path.dirname(checkpoint_filepath)

    data_name = test_affinity_filepath.split("/")[-1].split(".")[0]
    # read configuration
    with open(os.path.join(model_dir, "model_params.json")) as fp:
        params = json.load(fp)
    # setting device
    device = get_device()
    logger.info("using device {}".format(device))

    smiles_language = SMILESTokenizer.from_pretrained(
        os.path.join(model_dir, "smiles_language")
    )
    smiles_language.set_smiles_transforms(
        augment=False,
        # If we augment at training time we should canonicalize at test time
        canonical=params.get(
            "test_smiles_canonical", params.get("augment_smiles", False)
        ),
        kekulize=params.get("smiles_kekulize", False),
        all_bonds_explicit=params.get("smiles_bonds_explicit", False),
        all_hs_explicit=params.get("smiles_all_hs_explicit", False),
        remove_bonddir=params.get("smiles_remove_bonddir", False),
        remove_chirality=params.get("smiles_remove_chirality", False),
        selfies=params.get("selfies", False),
        sanitize=params.get("sanitize", False),
    )
    # preparing datasets
    column_names = ["ligand_name", "uniprot_accession", "pIC50"]
    dataset_arguments = OrderedDict(
        [
            ("column_names", column_names),
            ("smi_filepath", smi_filepath),
            ("protein_filepath", protein_filepath),
            ("protein_amino_acid_dict", protein_vocabulary),
            ("protein_padding", params.get("protein_padding", True)),
            ("protein_padding_length", params.get("receptor_padding_length", None)),
            ("protein_add_start_and_stop", params.get("protein_add_start_stop", True)),
            ("protein_augment_by_revert", False),
            ("device", device),
            ("drug_affinity_dtype", torch.float),
            ("backend", "eager"),
            ("iterate_dataset", params.get("iterate_dataset", False)),
        ]
    )
    logger.info("dataset arguments: {}".format(dataset_arguments))
    logger.info("preparing test dataset")
    test_dataset = DrugAffinityDataset(
        drug_affinity_filepath=test_affinity_filepath,
        smiles_language=smiles_language,
        **dataset_arguments,
    )
    logger.info("test loader")
    bs = params["batch_size"]
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=bs,
        shuffle=False,
        drop_last=False,
        num_workers=params.get("num_workers", 0),
    )

    logger.info("restore the model")
    if checkpoint_filepath.endswith(".ckpt"):
        ckpt_names = [checkpoint_filepath]
    else:
        # Only root to model dir was provided, create list of checkpoints
        ckpt_names = [
            os.path.join(checkpoint_filepath, f"val_rmse-v{v}.ckpt")
            for v in range(4, -1, -1)
        ]
        ckpt_names.extend([os.path.join(checkpoint_filepath, "val_rmse.ckpt")])

    for ckpt_filename in ckpt_names:
        if os.path.isfile(ckpt_filename):
            model = AffinityModel.load_from_checkpoint(ckpt_filename).to(device)
            logger.info(f"Found and restored existing model {ckpt_filename}")
            break
    try:
        model
    except NameError:
        raise NameError("Could not restore model")

    logger.info("testing the model")

    ligand_names = test_dataset.drug_affinity_df[column_names[0]].values
    sequence_ids = test_dataset.drug_affinity_df[column_names[1]].values
    df_labels = test_dataset.drug_affinity_df[column_names[2]].values

    model.eval()
    labels, predictions, loss_item = [], [], 0
    ligand_attention, protein_attention = [], []
    for ind, (smiles, proteins, y) in enumerate(test_loader):
        if ind > 0 and ind % 10 == 0:
            logger.info(f"Batch {ind}/{len(test_loader)}")

        # This verifies that no augmentation occurs, shuffle is False and that the
        # order of the dataloder is identical to the dataset
        assert all(smiles[0, :] == test_dataset[ind * bs][0])
        assert all(proteins[0, :] == test_dataset[ind * bs][1])
        assert all(y[0, :] == test_dataset[ind * bs][2])

        y_hat, pred_dict = model.model(smiles, proteins)
        loss = model.model.loss(y_hat, y.to(device))
        loss_item += loss.item()

        labels.extend(y.cpu().detach().squeeze().tolist())
        predictions.extend(y_hat.cpu().detach().squeeze().tolist())
        ligand_attention.extend(
            pred_dict["ligand_attention"].cpu().detach().squeeze().numpy()
        )
        protein_attention.extend(
            pred_dict["receptor_attention"].cpu().detach().squeeze().numpy()
        )

    assert np.allclose(np.array(labels), np.array(df_labels), atol=1e-5)

    protein_attention = np.array(protein_attention).astype("float32")
    ligand_attention = np.array(ligand_attention).astype("float32")

    # Compute metrics
    loss = loss_item / len(labels)

    rmse = np.sqrt(mean_squared_error(labels, predictions))
    pearson = pearsonr(labels, predictions)[0]
    spearman = spearmanr(labels, predictions)[0]

    print(f"RMSE = {rmse}, Pearson = {pearson}, Spearman = {spearman}, Loss = {loss}")

    # Save data
    results = {"RMSE": rmse, "Pearson": pearson, "Spearman": spearman, "Loss": loss}
    with open(
        os.path.join(model_dir, f"{ckpt_filename}_{data_name}_data_report.json"), "w"
    ) as f:
        json.dump(results, f)
    pd.DataFrame(
        {
            "sequence_id": sequence_ids,
            "ligand_name": ligand_names,
            "predictions": predictions,
            "labels": labels,
        }
    ).to_csv(
        os.path.join(model_dir, f"{ckpt_filename}_{data_name}_data_predictions.csv")
    )

    np.save(
        os.path.join(model_dir, f"{ckpt_filename}_{data_name}_protein_attention"),
        protein_attention,
    )
    np.save(
        os.path.join(model_dir, f"{ckpt_filename}_{data_name}_ligand_attention"),
        ligand_attention,
    )


if __name__ == "__main__":
    # parse arguments
    args = parser.parse_args()
    # run the training
    run_testing(
        args.test_affinity_filepath,
        args.protein_vocabulary,
        args.protein_filepath,
        args.smi_filepath,
        args.checkpoint_filepath,
        args.seed,
    )
