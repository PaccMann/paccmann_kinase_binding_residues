"""Generate SMILES from latent code with BO."""
import argparse
import json
import os
import pickle
from time import time

import pandas as pd
import torch
from loguru import logger
from paccmann_chemistry.models.vae import StackGRUDecoder, StackGRUEncoder, TeacherVAE
from paccmann_chemistry.utils import disable_rdkit_logging, get_device
from paccmann_chemistry.utils.search import SamplingSearch
from paccmann_generator.drug_evaluators.sas import SAS
from paccmann_gp.callable_minimization import CallableMinimization
from paccmann_gp.gp_optimizer import GPOptimizer
from pkbr.generative import PlainSmilesGenerator
from pkbr.knn import KnnPredictor
from pytoda.smiles.smiles_language import SMILESLanguage
from rdkit import Chem
from rdkit.Chem.Descriptors import qed

# parser
parser = argparse.ArgumentParser(description="SVAE SMILE generation with BO.")
parser.add_argument(
    "svae_path", type=str, help="Path to the trained model (SMILES VAE)"
)
parser.add_argument(
    "affinity_data_path", type=str, help="Path to the affinity data for lazy KNN"
)
parser.add_argument(
    "protein_filepath",
    type=str,
    help="Path to .smi file containing protein representations.",
)
parser.add_argument(
    "output_prefix", type=str, help="Prefix for the optimization (path/to/prefix)."
)
parser.add_argument(
    "-r",
    "--number_of_rounds",
    type=int,
    default=5,
    help="Number of optimization rounds.",
)
parser.add_argument(
    "-n",
    "--number_of_points_per_round",
    type=int,
    default=30,
    help="Number of points to generate per round.",
)
parser.add_argument(
    "-i",
    "--number_of_initial_points",
    type=int,
    default=40,
    help="Number of initial points.",
)
parser.add_argument(
    "-c", "--number_of_calls", type=int, default=50, help="Number of calls."
)
parser.add_argument(
    "-b", "--batch_size", type=int, default=64, help="Number of points in a batch."
)
parser.add_argument("-s", "--seed", type=int, default=1234, help="Random seed.")
parser.add_argument(
    "-t",
    "--temperature",
    type=float,
    default=1.0,
    help="Temperature for molecule generation",
)
parser.add_argument(
    "-j", "--number_of_jobs", type=int, default=1, help="Cores for GP optimization"
)


def main(parser_namespace):
    disable_rdkit_logging()
    affinity_data_path = parser_namespace.affinity_data_path
    svae_path = parser_namespace.svae_path
    protein_filepath = parser_namespace.protein_filepath
    output_prefix = parser_namespace.output_prefix
    number_of_rounds = parser_namespace.number_of_rounds
    number_of_points_per_round = parser_namespace.number_of_points_per_round
    number_of_initial_points = parser_namespace.number_of_initial_points
    number_of_calls = parser_namespace.number_of_calls
    batch_size = parser_namespace.batch_size
    number_of_jobs = parser_namespace.number_of_jobs
    seed = parser_namespace.seed
    temperature = parser_namespace.temperature

    os.makedirs(os.path.dirname(output_prefix), exist_ok=True)

    log_filepath = f"{output_prefix}.log"
    latent_point_prefix = f"{output_prefix}_latent_point"
    output_filepath_prefix = f"{output_prefix}_results"

    logger.add(log_filepath, rotation="10 MB")

    logger.info("loading protein data")
    protein_df = pd.read_csv(
        protein_filepath, header=None, index_col=1, sep="\t", names=["target_protein"]
    )

    logger.info(f"loading generative model from {svae_path}")
    svae_weights_filepath = os.path.join(svae_path, "weights.pt")
    svae_parameters = dict()
    with open(os.path.join(svae_path, "params.json"), "r") as f:
        svae_parameters.update(json.load(f))
    smiles_language = SMILESLanguage.load(
        os.path.join(svae_path, "smiles_language.pkl")
    )
    setattr(
        smiles_language,
        "special_indexes",
        {
            smiles_language.padding_index: smiles_language.padding_token,
            smiles_language.unknown_index: smiles_language.unknown_token,
            smiles_language.start_index: smiles_language.start_token,
            smiles_language.stop_index: smiles_language.stop_token,
        },
    )
    gru_encoder = StackGRUEncoder(svae_parameters)
    gru_decoder = StackGRUDecoder(svae_parameters)
    gru_vae = TeacherVAE(gru_encoder, gru_decoder)
    gru_vae.load_state_dict(
        torch.load(svae_weights_filepath, map_location=get_device())
    )
    gru_vae._associate_language(smiles_language)
    gru_vae.eval()
    smiles_generator = PlainSmilesGenerator(
        gru_vae, search=SamplingSearch(temperature=temperature)
    )
    logger.info(f"Set up SMILES generator with temperature {temperature}")

    logger.info(f"Set up KNN predictor from {affinity_data_path}")

    train_df = pd.read_csv(affinity_data_path)
    knn = KnnPredictor(train_df, seq_nns=10)

    for protein_identifier, protein_series in protein_df.iterrows():
        target_protein = protein_series["target_protein"]
        logger.info(
            f"processing protein {protein_identifier}, sequence: {target_protein}"
        )

        logger.info("setting up the optimizer")
        target_minimization_function = CallableMinimization(
            smiles_generator,
            knn,
            batch_size,
            mode="max",
            evaluator_kwargs={
                "protein": target_protein,
                "protein_accession": protein_identifier,
            },
        )
        # multi-objective example
        # from paccmann_gp.sa_minimization import SAMinimization
        # from paccmann_gp.qed_minimization import QEDMinimization
        # from paccmann_gp.combined_minimization import CombinedMinimization
        # qed_function = QEDMinimization(smiles_generator, batch_size)
        # sa_function = SAMinimization(smiles_generator, batch_size)
        # combined_minimization = CombinedMinimization(
        #     [target_minimization_function, qed_function, sa_function],
        #     1,
        #     [1.0, 0.25, 0.25],
        # )
        # optimizer = GPOptimizer(combined_minimization.evaluate)
        optimizer = GPOptimizer(target_minimization_function.evaluate)

        parameters = dict(
            dimensions=[(-5.0, 5.0)] * gru_vae.decoder.latent_dim,
            acq_func="EI",
            acq_optimizer="lbfgs",
            n_calls=number_of_calls,
            n_initial_points=number_of_initial_points,
            initial_point_generator="random",
            random_state=seed,
            n_jobs=number_of_jobs,
        )
        logger.info(f"optimisation parameters: {parameters}")

        def evaluate_point(latent_point):

            smile_set = set()
            logger.info(f"sample {number_of_points_per_round} mols from optimal point")
            t = time()
            while len(smile_set) < number_of_points_per_round and time() - t < 100:
                smiles = smiles_generator.generate_smiles(
                    latent_point.repeat(1, batch_size, 1)
                )
                smile_set.update(set(smiles))
            smile_set = list(smile_set)

            logger.info("compute properties")
            affinities = []
            sas = SAS()
            sa_scores = []
            qed_scores = []
            for index, a_smiles in enumerate(smile_set):
                try:
                    affinities.append(
                        knn(
                            a_smiles,
                            protein=target_protein,
                            protein_accession=protein_identifier,
                        )
                    )
                except Exception:
                    logger.warning("could not evaluate affinity")
                    affinities.append(0.0)

                try:
                    sa_scores.append(sas(a_smiles))
                except Exception:
                    logger.warning("could not evaluate SA")
                    sa_scores.append(10.0)

                try:
                    qed_scores.append(qed(Chem.MolFromSmiles(a_smiles)))
                except Exception:
                    logger.warning("could not evaluate QED")
                    qed_scores.append(0.0)

            output_filepath = f"{output_filepath_prefix}_protein={protein_identifier}_round={round+1}.csv"
            logger.info(f"saving results to {output_filepath}")

            with open(output_filepath, "w") as f:
                f.write(f"point,affinity,QED,SA,smiles{os.linesep}")
                for generated_point_index in range(number_of_points_per_round):
                    f.write(
                        f"{generated_point_index + 1},{affinities[generated_point_index]},{qed_scores[generated_point_index]}"
                        f",{sa_scores[generated_point_index]},{smile_set[generated_point_index]}{os.linesep}"
                    )

        logger.info("Evaluating random point")
        latent_point = torch.randn(size=(1, 1, gru_vae.decoder.latent_dim))
        round = -1
        evaluate_point(latent_point)

        # optimisation
        for round in range(number_of_rounds):
            logger.info(f"starting round: {round+1}")
            try:

                results = optimizer.optimize(parameters)
                latent_point = torch.tensor([[results.x]])
                latent_point_filepath = f"{latent_point_prefix}_protein={protein_identifier}_round={round+1}.pkl"
                logger.info(f"saving optimal latent point to {latent_point_filepath}")
                with open(latent_point_filepath, "wb") as f:
                    pickle.dump(latent_point, f, protocol=2)

                evaluate_point(latent_point)

            except Exception:
                logger.error(
                    f"optimization error in round={round+1} for {protein_identifier}"
                )


if __name__ == "__main__":
    main(parser.parse_args())
