"""Training affinity models."""
import argparse
import json
import logging
import os
import sys
from collections import OrderedDict
from copy import deepcopy
from typing import Optional

import numpy as np
import pytorch_lightning as pl
import torch
from loguru import logger
from paccmann_predictor.models.bimodal_mca import BimodalMCA
from pytoda.datasets.drug_affinity_dataset import DrugAffinityDataset
from pytoda.smiles.smiles_language import SMILESTokenizer
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error

# setup logging
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

parser = argparse.ArgumentParser()
parser.add_argument(
    "train_affinity_filepath",
    type=str,
    help="path to the drug affinity data for train.",
)
parser.add_argument(
    "dev_affinity_filepath", type=str, help="path to the drug affinity data for dev."
)
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
    "smiles_language_filepath", type=str, help="path to the SMILES language."
)
parser.add_argument(
    "model_path", type=str, help="directory where the model will be stored."
)

parser.add_argument("params_filepath", type=str, help="path to the parameter file.")
parser.add_argument(
    "-n",
    "--training_name",
    type=str,
    help="name for the training.",
    default="affinity_training",
)
parser.add_argument(
    "-s", "--seed", type=int, help="seed for reproducibility.", default=42
)
parser.add_argument(
    "-f",
    "--finetune_path",
    type=str,
    help="Optional path to a checkpoint for finetuning",
    default="",
)


class AffinityModel(pl.LightningModule):
    def __init__(self, params: dict) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = BimodalMCA(params)

        self.num_params = sum(p.numel() for p in self.model.parameters())

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
        y = y.cpu().detach().squeeze().tolist()
        y_hat = y_hat.cpu().detach().squeeze().tolist()
        rmse = np.sqrt(mean_squared_error(y_hat, y))
        pearson = pearsonr(y_hat, y)[0]
        spearman = spearmanr(y_hat, y)[0]
        self.log("val_loss", loss)
        self.log("val_rmse", rmse)
        self.log("val_pearson", pearson)
        self.log("val_spearman", spearman)

    def test_step(self, batch: tuple, batch_idx: int) -> None:
        smiles, proteins, y = batch
        y_hat = self(smiles, proteins)
        loss = self.model.loss(y_hat, y)

        y = y.cpu().detach().squeeze().tolist()
        y_hat = y_hat.cpu().detach().squeeze().tolist()
        rmse = np.sqrt(mean_squared_error(y_hat, y))
        pearson = pearsonr(y_hat, y)[0]
        spearman = spearmanr(y_hat, y)[0]
        self.log("test_loss", loss)
        self.log("test_rmse", rmse)
        self.log("test_pearson", pearson)
        self.log("test_spearman", spearman)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(
            self.model.parameters(), lr=self.hparams["params"]["lr"]
        )


def run_training(
    train_affinity_filepath: str,
    dev_affinity_filepath: str,
    test_affinity_filepath: str,
    protein_vocabulary: str,
    protein_filepath: str,
    smi_filepath: str,
    smiles_language_filepath: str,
    model_path: str,
    params_filepath: str,
    training_name: str,
    seed: int,
    finetune_path: Optional[str],
) -> None:
    """Run training."""
    pl.seed_everything(seed=seed)
    # read configuration
    with open(params_filepath) as fp:
        params = json.load(fp)
    # prepare model folder
    model_dir = os.path.join(model_path, training_name)
    os.makedirs(model_dir, exist_ok=True)
    logger.info("set up model folder at {}".format(model_dir))
    smiles_language = SMILESTokenizer(
        vocab_file=smiles_language_filepath,
        padding_length=params.get("ligand_padding_length", None),
        randomize=None,
        add_start_and_stop=params.get("ligand_start_stop_token", True),
        padding=params.get("ligand_padding", True),
        augment=params.get("augment_smiles", False),
        canonical=params.get("smiles_canonical", False),
        kekulize=params.get("smiles_kekulize", False),
        all_bonds_explicit=params.get("smiles_bonds_explicit", False),
        all_hs_explicit=params.get("smiles_all_hs_explicit", False),
        remove_bonddir=params.get("smiles_remove_bonddir", False),
        remove_chirality=params.get("smiles_remove_chirality", False),
        selfies=params.get("selfies", False),
        sanitize=params.get("sanitize", False),
    )
    # preparing datasets
    dataset_shared_arguments = OrderedDict(
        [
            ("column_names", ["ligand_name", "uniprot_accession", "pIC50"]),
            ("smi_filepath", smi_filepath),
            ("protein_filepath", protein_filepath),
            ("protein_amino_acid_dict", protein_vocabulary),
            ("protein_padding", params.get("protein_padding", True)),
            ("protein_padding_length", params.get("receptor_padding_length", None)),
            ("protein_add_start_and_stop", params.get("protein_add_start_stop", True)),
            ("protein_augment_by_revert", params.get("protein_augment", False)),
            ("drug_affinity_dtype", torch.float),
            ("backend", "eager"),
            ("iterate_dataset", params.get("iterate_dataset", False)),
        ]
    )  # yapf: disable
    logger.info("shared dataset arguments: {}".format(dataset_shared_arguments))
    logger.info("preparing training dataset")
    train_augment = params.get('protein_sequence_augment', {})
    # If training augmentation is used, we have to trigger the pipeline also at
    # test time to ensure that full sequences can be used even if the `protein_filepath`
    # has only active sites (and `discard_lowercase` is False).
    test_augment = (
        {}
        if train_augment == {}
        else {
            'discard_lowercase': train_augment.get('discard_lowercase', True),
        }
    )

    train_dataset = DrugAffinityDataset(
        drug_affinity_filepath=train_affinity_filepath,
        smiles_language=smiles_language,
        protein_sequence_augment=train_augment,
        **dataset_shared_arguments,
    )

    # Augmentation is always off at test time
    dataset_shared_arguments.pop("protein_augment_by_revert")
    dataset_shared_arguments["protein_augment_by_revert"] = False
    eval_smiles_language = deepcopy(smiles_language)
    eval_smiles_language.set_smiles_transforms(
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

    logger.info("preparing dev dataset")
    dev_dataset = DrugAffinityDataset(
        drug_affinity_filepath=dev_affinity_filepath,
        smiles_language=eval_smiles_language,
        protein_sequence_augment=test_augment,
        **dataset_shared_arguments,
    )
    logger.info("preparing test dataset")
    test_dataset = DrugAffinityDataset(
        drug_affinity_filepath=test_affinity_filepath,
        smiles_language=eval_smiles_language,
        protein_sequence_augment=test_augment,
        **dataset_shared_arguments,
    )
    logger.info("updating language-related parameters")
    params.update(
        {
            "ligand_vocabulary_size": train_dataset.smiles_dataset.smiles_language.number_of_tokens,
            "receptor_vocabulary_size": train_dataset.protein_sequence_dataset.protein_language.number_of_tokens,
        }
    )
    logger.info("storing languages")
    os.makedirs(os.path.join(model_dir, "smiles_language"), exist_ok=True)
    train_dataset.smiles_dataset.smiles_language.save_pretrained(
        os.path.join(model_dir, "smiles_language")
    )
    logger.info("storing parameters")
    with open(os.path.join(model_dir, "model_params.json"), "w") as fp:
        json.dump(params, fp, indent=4)
    logger.info("train loader")
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=params["batch_size"],
        shuffle=True,
        drop_last=True,
        num_workers=params.get("num_workers", 0),
    )
    logger.info("dev loader")
    dev_loader = torch.utils.data.DataLoader(
        dataset=dev_dataset,
        batch_size=params["batch_size"],
        shuffle=False,
        drop_last=True,
        num_workers=params.get("num_workers", 0),
    )
    logger.info("test loader")
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=params["batch_size"],
        shuffle=False,
        drop_last=True,
        num_workers=params.get("num_workers", 0),
    )
    model = AffinityModel(params)
    logger.info(f"Instantiate model. Learning rate is {model.hparams['params']['lr']}")

    if finetune_path:
        if os.path.isfile(finetune_path):
            try:
                model = AffinityModel.load_from_checkpoint(finetune_path)
                logger.info(f"Restored pretrained model {finetune_path}")
            except Exception:
                raise KeyError(f"Could not restore model from {finetune_path}")
        else:
            raise FileNotFoundError(f"Did not find model at {finetune_path}")
            logger.info(f"Did not find provided checkpoint {finetune_path}")
    else:
        ckpt_filenames = params.get(
            "model_ckpt_name",
            [
                "val_rmse-v2.ckpt",
                "val_rmse-v1.ckpt",
                "val_rmse-v0.ckpt",
                "val_rmse.ckpt",
            ],
        )
        for ckpt_filename in ckpt_filenames:
            ckpt_path = os.path.join(model_dir, ckpt_filename)
            if os.path.isfile(ckpt_path):
                model = AffinityModel.load_from_checkpoint(ckpt_path)
                logger.info(f"Found and restored existing model {ckpt_path}")
                break

    val_loss_callback = pl.callbacks.ModelCheckpoint(
        dirpath=model_dir,
        filename="val_loss",
        mode="min",
        monitor="val_loss",
        verbose=True,
        period=1,
        save_last=True,
        save_top_k=1,
    )
    val_rmse_callback = pl.callbacks.ModelCheckpoint(
        dirpath=model_dir,
        filename="val_rmse",
        mode="min",
        monitor="val_rmse",
        verbose=True,
        period=1,
        save_top_k=1,
    )
    val_pearson_callback = pl.callbacks.ModelCheckpoint(
        dirpath=model_dir,
        filename="val_pearson",
        mode="max",
        monitor="val_pearson",
        verbose=True,
        period=1,
        save_top_k=1,
    )
    logger.info(f"# Params = {model.num_params}")

    logger.info("instantiate trainer")
    logger.info(f"Auto LR finder {params.get('auto_lr_find', False)}")

    trainer = pl.Trainer(
        max_epochs=params["epochs"],
        checkpoint_callback=True,
        callbacks=[val_rmse_callback, val_loss_callback, val_pearson_callback],
        check_val_every_n_epoch=params.get("save_model", 2),
        default_root_dir=model_dir,
        gpus=(1 if torch.cuda.is_available() else 0),
        progress_bar_refresh_rate=0,
        auto_lr_find=params.get("auto_lr_find", False),
    )
    logger.info("run training")
    trainer.fit(model, train_loader, dev_loader)
    logger.info("testing the model")
    trainer.test(test_dataloaders=test_loader, ckpt_path=None)


if __name__ == "__main__":
    # parse arguments
    args = parser.parse_args()
    # run the training
    run_training(
        args.train_affinity_filepath,
        args.dev_affinity_filepath,
        args.test_affinity_filepath,
        args.protein_vocabulary,
        args.protein_filepath,
        args.smi_filepath,
        args.smiles_language_filepath,
        args.model_path,
        args.params_filepath,
        args.training_name,
        args.seed,
        args.finetune_path,
    )
