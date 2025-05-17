import argparse
from datetime import datetime
from glob import glob
import os
import re
from typing import List, Dict, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import model_adata_setup as model_setup
import yaml
from scipy.stats import median_abs_deviation 

from embedded_rp_poisson_model import EmbeddedRegPotPoissonRegressor 
import evaluate_regions
import deviance_residuals as dev_res
import in_silico_deletion
import make_embeddings


def load_config(path="config.yaml"):
    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    # Validate types
    assert isinstance(raw["model"]["num_pro_types"], int), "num_pro_types must be an int"
    assert isinstance(raw["model"]["num_enh_types"], int), "num_enh_types must be an int"
    assert isinstance(raw["model"]["num_genes"], int), "num_genes must be an int"
    assert isinstance(raw["model"]["num_models"], int), "num_models must be an int"
    assert isinstance(raw["model"]["num_classes"], int), "num_classes must be an int"
    assert isinstance(raw["model"]["max_cres_per_gene"], int), "max_cres_per_gene must be an int"

    assert isinstance(raw["model"]["cre_decay_distance"], int), "cre_decay_distance must be an int"
    assert isinstance(raw["model"]["cre_max_distance"], int), "cre_max_distance must be an int"

    assert isinstance(raw["features"]["use_signal"], bool), "use_signal must be a bool"
    # assert isinstance(raw["features"]["use_rp"], bool), "use_rp must be a bool"
    #assert isinstance(raw["features"]["use_embeddings"], bool), "use_embeddings must be a bool"

    if "use_local_embeddings" in raw["features"]:
        assert isinstance(raw["features"]["use_local_embeddings"], bool), "use_embeddings must be a bool"

    assert isinstance(raw["files"]["gene_ref_file"], str), "gene_ref_file must be a string"
    assert isinstance(raw["files"]["gene_alias_file"], str), "gene_alias_file must be a string"
    assert isinstance(raw["files"]["atac_file"], str), "atac_file must be a string"
    assert isinstance(raw["files"]["gex_file"], str), "gex_file must be a string"
    assert isinstance(raw["files"]["embed_file"], (str, type(None))), "embed_file must be a string or None"

    assert isinstance(raw["experiment"]["name"], str), "experiment.name must be a string"

    assert isinstance(raw["training"]["epochs"], int), "epochs must be int"

    config = {
        "num_pro_types": raw["model"]["num_pro_types"],
        "num_enh_types": raw["model"]["num_enh_types"],
        "num_genes": raw["model"]["num_genes"],
        "num_models": raw["model"]["num_models"],
        "num_classes": raw["model"]["num_classes"],
        "max_cres_per_gene": raw["model"]["max_cres_per_gene"],

        "cre_decay_distance": raw["model"]["cre_decay_distance"],
        "cre_max_distance": raw["model"]["cre_max_distance"],

        "use_signal": raw["features"]["use_signal"],
        #"use_rp": raw["features"]["use_rp"],
        #"use_embeddings": raw["features"]["use_embeddings"],
        # default to true -- use local embeddings
        "use_local_embeddings": raw["features"].get("use_local_embeddings", True),

        "gene_ref_file": raw["files"]["gene_ref_file"],
        "gene_alias_file": raw["files"]["gene_alias_file"], 
        "atac_file": raw["files"]["atac_file"],
        "gex_file": raw["files"]["gex_file"],
        "embed_file": raw["files"]["embed_file"],

        "experiment_name": raw["experiment"]["name"],

        "epochs": raw["training"]["epochs"]
    }

    # set use_embeddings to True if either local embeddings or external is True
    config["use_embeddings"] = config["use_local_embeddings"] or config["embed_file"]
    # convert a potential None to False
    config["use_embeddings"] = True if config["use_embeddings"] else False

    return config


def compose_experiment_name(
    experiment_name: str,
    use_local_embeddings: bool,
    use_external_embeddings: bool, 
    use_signal: bool ) -> str:
    name_a = "local-embed" if use_local_embeddings else "no-local-embed"
    name_b = "external-embed" if use_external_embeddings else "no-external-embed"
    name_c = "signal" if use_signal else "no-signal"
    return f"{experiment_name}_{name_a}_{name_b}_{name_c}"


def find_common_prefix(strings):
    """Find the longest common prefix among a list of strings."""
    if not strings:
        return ''
    prefix = strings[0]
    for s in strings[1:]:
        # zip pairs up characters, take while they match
        prefix = ''.join(c1 for c1, c2 in zip(prefix, s) if c1 == c2)
        if not prefix:
            break
    return prefix


def get_first_token(s):
    """Extract the first meaningful token from a remainder string."""
    # Split on underscores or hyphens (common in filenames)
    tokens = re.split(r'[_\-]', s)
    return tokens[0] if tokens else ''


def assign_name_category(names):
    """
    Assign categories based on the first differing token after common prefix.
    Args:
        model_names (List[str]): List of model filenames.
    Returns:
        List[str]: List of indices matching input order.
    """
    common_prefix = find_common_prefix(names)
    first_tokens = [get_first_token(name[len(common_prefix):]) for name in names]
    unique_tokens = sorted(set(first_tokens))
    token_to_index = {token: i for i, token in enumerate(unique_tokens)}
    return [token_to_index[token] for token in first_tokens]


def evaluate_model_fit(model_paths: List[str], config_paths: List[str], verbose=True):
    """
    Load models, compute predictions and residuals, and plot MAD residual comparison.

    Parameters:
        model_paths (List[str]): List of model checkpoint paths.
        config_paths (List[str]): List of config.yaml paths, one for each model.
        verbose (bool): Print status messages.
    """
    if len(model_paths) != len(config_paths):
        raise ValueError("model_paths and config_paths must have the same length.")

    models = []
    model_labels = []
    all_results = []

    for model_path, config_path in zip(model_paths, config_paths):
        if verbose:
            print('='*50)
            print(f"Model: {model_path}")
            print(f"Config: {config_path}")
            print('='*50)

        config = load_config(path=config_path)

        if not os.path.exists(model_path):
            print(f"Model path does not exist: {model_path}")

        checkpoint = torch.load(model_path, map_location="cpu")
        model = EmbeddedRegPotPoissonRegressor(**checkpoint["params"])
        model.load_state_dict(checkpoint["state_dict"])

        if verbose:
            print("==== checkpoint ====")
            print(checkpoint["params"])
            print("====================")

        data_files = checkpoint["data_files"]
        # TODO check data_keys = {"atac_file", "gex_file", "embed_file", "max_cres_per_gene", "num_genes"}
        data_keys = {"atac_file", "gex_file", "embed_file",
            "gene_alias_file", "gene_ref_file",  
            "max_cres_per_gene", "num_genes", "use_local_embeddings"}

        data_kwargs = {k: v for k, v in config.items() if k in data_keys}
        data_kwargs.update(data_files)

        gene_reg_data = model_setup.model_setup(**data_kwargs)
        gene_reg_data.populate_gene_subset_matrices(gene_set_name='validate')

        if verbose:
            print("Gene regulation data loaded.")
            print("Shape of CRE tensor:", gene_reg_data.cre_tensor.shape)

        model.eval()
        models.append(model)

        print( gene_reg_data.atac_adata.obs )
        cell_types = gene_reg_data.atac_adata.obs['cell_type']

        results = dev_res.compute_predictions_and_residuals([model], gene_reg_data)[0]  # Only one model at a time
        all_results.append(results)

        label = checkpoint["params"].get("label", os.path.splitext(os.path.basename(model_path))[0])
        model_labels.append(label)
        if verbose:
            print(f"Loaded model from {path} with label: {label}")

    if not models:
        raise ValueError("No valid models loaded.")

    # Compute predictions & residuals for each model
    # Plot MAD deviance residual comparison
    model_categories = assign_name_category(model_labels)
    dev_res.plot_mad_deviance_by_cell_state_rows(all_results, model_labels=model_labels, cell_types=cell_types, 
        model_categories=model_categories, name="mad_comparison")
    dev_res.plot_deviance_residual_histograms_by_cell_state(all_results, model_labels=model_labels,name="dev_res_histograms")


def evaluate_cre_activity_and_write(model_path, config_path, method='cre_activity', plot_diagnostics=False):  

    config = load_config(path=config_path)
    if not os.path.exists(model_path):
        raise "path not found"
    checkpoint = torch.load(model_path, map_location="cpu")

    model = EmbeddedRegPotPoissonRegressor(**checkpoint["params"])
    model.load_state_dict(checkpoint["state_dict"])

    if plot_diagnostics:
        model.plot_cre_heatmaps( cell_state_labels=None, pro_type_labels=None, enh_type_labels=None)

    data_files = checkpoint["data_files"]
    data_keys = {"atac_file", "gex_file", "embed_file", 
        "gene_alias_file", "gene_ref_file",  
        "max_cres_per_gene", "num_genes", "use_local_embeddings"}

    data_kwargs = {k: v for k, v in config.items() if k in data_keys}
    data_kwargs.update(data_files)

    gene_reg_data = model_setup.model_setup(**data_kwargs)
    gene_reg_data.populate_gene_subset_matrices(gene_set_name='all')

    _,name = os.path.split(model_path)
    name, ext = os.path.splitext(name)
 
    if method == 'cre_activity':
        evaluate_regions.evaluate_cre_activity_and_write( model=model, gene_reg_data=gene_reg_data, model_name=name )  
    elif method == 'in_silico_deletion':
        print( 'In silico deletion analysis is not recommended in this application.')
        in_silico_deletion.in_silico_deletion_analysis(model, gene_reg_data)


def fit_model(config_file="config.yaml", output='.',  verbose=False):
    """Main training function for Poisson Regressor."""

    config = load_config(config_file)

    experiment_name = compose_experiment_name(
        experiment_name = config["experiment_name"],
        use_local_embeddings = config["use_local_embeddings"],
        use_external_embeddings = (config["embed_file"] is not None),
        use_signal = config["use_signal"]
    )

    data_keys = {"atac_file", "gex_file", "embed_file", 
        "gene_alias_file", "gene_ref_file",  
        "max_cres_per_gene", "num_genes", "use_local_embeddings"}

    data_kwargs = {k: v for k, v in config.items() if k in data_keys}

    # NOTE: external_embeddings and local_embeddings are handled in setup
    # they are treated in the same way in the model
    gene_reg_data = model_setup.model_setup(**data_kwargs)

    num_cell_states = gene_reg_data.cre_all.shape[0]
    #max_num_genes = gene_reg_data.cre_tensor.shape[1]
    max_num_genes = config['num_genes']

    #max_cres_per_gene = gene_reg_data.cre_tensor.shape[2]
    max_cres_per_gene = config['max_cres_per_gene']
    if gene_reg_data.cre_embeddings_ref is None:
        embedding_dim = 0
    else:
        embedding_dim = gene_reg_data.cre_embeddings_ref.shape[1]

    models = []
    loss_record = {}

    model_keys = {
        "num_genes", "max_cres_per_gene", "num_cell_states",
        "num_pro_types", "num_enh_types", "num_classes",
        "cre_decay_distance", "cre_max_distance",
        "use_embeddings", "use_baseline", "use_signal"  # TODO check use_embeddings
    }

    model_kwargs = {k: v for k, v in config.items() if k in model_keys}
    model_kwargs["embedding_dim"] = embedding_dim
    model_kwargs["num_cell_states"] = num_cell_states

    # solve multiple times to avoid poor local minima
    for _iter in range( config["num_models"] ):
 
        gene_reg_data.populate_gene_subset_matrices(gene_set_name='train')
        model_kwargs["num_genes"] = gene_reg_data.num_genes
        # Initialize model with embeddings
        model = EmbeddedRegPotPoissonRegressor(**model_kwargs)
        print( "model instantiated" )

        model.train_model( gene_reg_data, num_epochs= config['epochs'])
 
        # switch to cross-validation gene set and train gene baseline params
        gene_reg_data.populate_gene_subset_matrices(gene_set_name='validate')
        #gene_reg_data.summarize_cre_tensors()
        model.initialize_gene_parameters( num_genes=gene_reg_data.num_genes )
        model.freeze_non_gene_parameters()

        # tunes only cell type and gene bias params -- so fewer epochs are needed
        model.train_model( gene_reg_data, num_epochs=config['epochs'] )

        model.eval()
        loss_value = model.evaluate_loss(gene_reg_data)

        if verbose:
            print(model.summarize_model_state())
            print("log_delta (mean):", model.log_delta.mean().item()) 
            print(f"Validation Loss: {loss_value}")

        data_keys = {"atac_file", "gex_file", "embed_file"}
        data_files = {k: v for k, v in config.items() if k in data_keys}

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join( output, f"model_{experiment_name}_{timestamp}_validate.pth" )
        model.save_model( path=model_path, data_files=data_files )
        models.append(model)
        loss_record[model_path] = loss_value

    # Find the model with the smallest loss
    best_model_path = min(loss_record, key=loss_record.get)

    # Read in this best model
    if not os.path.exists(best_model_path):
        raise "Best model not found"

    # load best model
    checkpoint = torch.load(best_model_path)
    model = EmbeddedRegPotPoissonRegressor(**checkpoint["params"])
    result = model.load_state_dict(checkpoint["state_dict"])

    # switch to all genes and train gene baseline params
    gene_reg_data.populate_gene_subset_matrices(gene_set_name='all')
    print( "updating params for all genes num genes", gene_reg_data.num_genes )
    model.initialize_gene_parameters( num_genes=gene_reg_data.num_genes )
    model.freeze_non_gene_parameters()
    model.train_model( gene_reg_data, num_epochs=config['epochs'] )
    model.eval()

    # save model with all genes
    data_files = checkpoint["data_files"]
    if verbose:
        print( "Best model:", best_model_path )
    #name,ext = best_model_path.split('.')
    name, ext = os.path.splitext(best_model_path)

    name = name.replace('validate','all')
    model_path = f"{name}{ext}"
    model.save_model( path=model_path, data_files=data_files )

    # Delete the other models
    for path in loss_record:
        if path != best_model_path and os.path.exists(path):
            os.remove(path)
            if verbose:
                print(f"Deleted model: {path}")


def expand_model_paths(paths):
    """Expand any folder paths into .pth model files."""
    expanded = []
    for path in paths:
        if os.path.isdir(path):
            expanded.extend(sorted(glob(os.path.join(path, "*.pth"))))
        else:
            expanded.append(path)
    return expanded


def main():
    parser = argparse.ArgumentParser(description="scEpiSparX is a suite of tools for cis-regulatory analysis of single cell data.\
         The method can accept scRNA-seq optionally in combination with matched-barcode scATAC-seq data (eg. 10X Multiome or SHARE-seq).\
         Genomic regions are represented as epigenetic embeddings, either CistromeSparX embeddings from Cistrome DB epiegenetic data or \
         embeddings from scATAC-seq data, which can be generated with the \"make_embeddings\" option.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Option 1: Fit a model using a config file
    fit_parser = subparsers.add_parser("fit_model", help="Fit cis-regulatory model from config")
    fit_parser.add_argument("config", type=str, help="Path to config file")
    fit_parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    fit_parser.add_argument("--output", type=str, default='.', help="Output path")

    # Option 2: Evaluate CRE activity for a single model
    eval_parser = subparsers.add_parser("find_cres", help="Evaluate CRE activity for a single model and write to BED files (one file per cell type).")
    eval_parser.add_argument("model_path", type=str, help="Path to a pytorch model file")
    eval_parser.add_argument("--config", type=str, required=True, help="Path to a config file")
    eval_parser.add_argument("--diagnostics", action="store_true", help="Plot diagnostics.")

    # Option 3: Evaluate and compare multiple models in terms of model fit
    compare_parser = subparsers.add_parser("compare_models", help="Evaluate model fit for multiple models")
    compare_parser.add_argument("model_paths", nargs="+", type=str, help="List of model paths or folders")
    compare_parser.add_argument("--configs", nargs="+", type=str, required=True, help="List of config paths, matching model_paths")
    compare_parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    # Option 4: Make embeddings from scATAC-seq data.
    embed_parser = subparsers.add_parser("make_embeddings", help="Make embeddings from scATAC-seq data and save in AnnData file.")
    embed_parser.add_argument("input_file", type=str, help="Path to the input AnnData (.h5ad) file.")
    embed_parser.add_argument("--output_file", "-o", type=str, default = "", help="Path to save the processed AnnData (.h5ad) file.")
    embed_parser.add_argument("--embeddings", "-e", type=str, default="PPMI", help="Embedding type PPMI or LSI")
    embed_parser.add_argument("--n_components", "-n", type=int, default=16, help="Embedding dimension (default: 16).")
    embed_parser.add_argument("--no_binarize", action="store_true", help="Disable binarization of the data.")
    embed_parser.add_argument("--min_cells", type=int, default=10,    help="Filter peaks.")
    embed_parser.add_argument("--min_reads", type=int, default=100,   help="Filter cells.")
    embed_parser.add_argument("--min_promoter_fraction", type=int, default=0,   help="Filter cells based on promoter reads.")
    embed_parser.add_argument("--max_mito_fraction", type=int, default=1,   help="Filter cells based on mitochondrial reads.")
    embed_parser.add_argument("--verbose", action="store_true", help="Print information to help with debugging.")

    args = parser.parse_args()

    if args.command == "fit_model":
        fit_model(config_file=args.config, output=args.output, verbose=args.verbose)

    elif args.command == "find_cres":
        evaluate_cre_activity_and_write( args.model_path, args.config, method='cre_activity', plot_diagnostics=args.diagnostics)

    elif args.command == "compare_models":
        model_paths = expand_model_paths(args.model_paths)
        if not model_paths:
            raise ValueError("No .pth files found in provided paths.")
        evaluate_model_fit(args.model_paths, args.configs, verbose=args.verbose)

    elif args.command == "make_embeddings":
        make_embeddings.main(args)


if __name__ == "__main__":
    main()

