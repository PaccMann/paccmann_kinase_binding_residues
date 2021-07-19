"""Generative modeling module."""
from typing import Any, List

import numpy as np
import pytorch_lightning as pl
import torch
from loguru import logger
from paccmann_chemistry.utils import get_device
from paccmann_gp.minimization_function import DecoderBasedMinimization
from paccmann_gp.smiles_generator import SmilesGenerator
from paccmann_predictor.models.bimodal_mca import BimodalMCA
from pytoda.transforms import LeftPadding, ToTensor
from rdkit import Chem

device = get_device()


class PlainSmilesGenerator(SmilesGenerator):
    """Smiles Generator"""

    def generate_smiles(self, latent_point: Any, to_smiles: bool = True) -> List[Any]:
        """
        Generate SMILES from latent latent_point.

        Args:
            latent_point (Any): the input latent point as tensor with shape
                `[1,batch_size,latent_dim]`.
            to_smiles (bool): boolean to specify if output should be SMILES (True) or
                numerical sequence (False). Defaults to True.

        Returns:
            List of molecules represented as SMILES or tokens.
        """
        molecules_numerical = self.model.generate(
            latent_point,
            prime_input=torch.LongTensor([self.model.smiles_language.start_index]).to(
                device
            ),
            end_token=torch.LongTensor([self.model.smiles_language.stop_index]).to(
                device
            ),
            search=self.search,
        )

        # convert numerical output to smiles
        if to_smiles:
            smiles = [
                self.model.smiles_language.token_indexes_to_smiles(
                    molecule_numerical.tolist()
                )
                for molecule_numerical in iter(molecules_numerical)
            ]
            molecules = []
            for a_smiles in smiles:
                try:
                    molecules.append(Chem.MolFromSmiles(a_smiles, sanitize=True))
                except Exception:
                    molecules.append(None)

            smiles = [
                smiles[index]
                for index in range(len(molecules))
                if not (molecules[index] is None)
            ]
            return smiles if smiles else self.generate_smiles(latent_point)
        else:
            return molecules_numerical


class CustomAffinityMinimization(DecoderBasedMinimization):
    """Minimization function for target affinity."""

    def __init__(
        self,
        smiles_decoder: SmilesGenerator,
        batch_size: int,
        affinity_predictor: BimodalMCA,
        protein: str,
    ) -> None:
        """
        Initialize an affinity minimization function.

        Args:
            smiles_decoder (SmilesGenerator): a SMILES generator.
            batch_size (int): size of the batch for evaluation.
            affinity_predictor (BimodalMCA): an affinity predictor.
            protein (str): string description of a protein compatible with the generator
                and the predictor.
        """
        super(CustomAffinityMinimization, self).__init__(smiles_decoder)

        self.batch = batch_size

        self.predictor = affinity_predictor
        self.to_tensor = ToTensor(device)

        self.protein = protein

        # protein to tensor
        self.pad_protein_predictor = LeftPadding(
            self.predictor.protein_padding_length,
            self.predictor.protein_language.padding_index,
        )

        self.protein_numeric = torch.unsqueeze(
            self.to_tensor(
                self.pad_protein_predictor(
                    self.predictor.protein_language.sequence_to_token_indexes(
                        self.protein
                    )
                )
            ),
            0,
        )

        self.pad_smiles_predictor = LeftPadding(
            self.predictor.smiles_padding_length,
            self.predictor.smiles_language.padding_index,
        )

    def evaluate(self, point: Any) -> float:
        """
        Evaluate a point.

        Args:
            point (Any): point to evaluate.

        Returns:
            (float): evaluation for the given point.
        """
        latent_point = torch.tensor([[point]]).to(device)
        batch_latent = latent_point.repeat(1, self.batch, 1)
        smiles = self.generator.generate_smiles(batch_latent)
        # smiles to tensor for affinity prediction
        smiles_tensor = torch.cat(
            [
                torch.unsqueeze(
                    self.to_tensor(
                        self.pad_smiles_predictor(
                            self.predictor.smiles_language.smiles_to_token_indexes(
                                smile
                            ).tolist()
                        )
                    ),
                    0,
                )
                for smile in smiles
            ],
            dim=0,
        )
        if len(smiles) < 3:
            smiles_tensor = smiles_tensor.repeat(2, 1)
        protein_tensor = self.protein_numeric.repeat(len(smiles_tensor), 1)
        # affinity predicition
        try:
            with torch.no_grad():
                affinity_prediction, _ = self.predictor(smiles_tensor, protein_tensor)
            scaled_predictions = [
                np.clip(raw_prediction, 0.0, 9.0) / 9.0  # 0 millimolar - 9 nano molar
                for raw_prediction in torch.squeeze(
                    affinity_prediction.cpu(), 1
                ).numpy()
            ]
            return 1 - (sum(scaled_predictions) / len(smiles))
        except Exception:
            logger.warning("Affinity calculation failed")
            return 1


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
