import pandas as pd
import numpy as np

def write_simulated_data(sim_res, file_prefix="simulated"):
    """
    Writes simulated gene expression, CRE activity, CRE accessibility, 
    CRE type, CRE embeddings, and gene-CRE distance data to CSV files.
    
    Args:
        sim_res (dict): Dictionary containing simulation results.
        file_prefix (str): Prefix for output filenames.
    """
    num_states = len(sim_res["gene_expression"])
    num_genes = len(sim_res["gene_expression"][0])  
    num_cres = len(sim_res["cre_activity"][0])  
    num_cre_types = sim_res["cre_type"].shape[1]
    dim_cre_embeddings = sim_res["cre_embeddings"].shape[1]

    gene_indices = [f"Gene_{i}" for i in range(num_genes)]
    cre_indices = [f"CRE_{i}" for i in range(num_cres)]

    # Save gene expression
    gene_expr_df = pd.DataFrame(
        {f"State_{i}": sim_res["gene_expression_int_poisson"][i] for i in range(num_states)},
        index=gene_indices
    )
    gene_expr_df.to_csv(f"{file_prefix}_gene_expression.csv")

    # Save CRE activity
    cre_activity_df = pd.DataFrame(
        {f"State_{i}": sim_res["cre_activity_int_beta"][i] for i in range(num_states)},
        index=cre_indices
    )
    cre_activity_df.to_csv(f"{file_prefix}_cre_activity.csv")

    # Save CRE accessibility
    cre_access_df = pd.DataFrame(
        {f"State_{i}": sim_res["cre_accessibility"][i] for i in range(num_states)},
        index=cre_indices
    )
    cre_access_df.to_csv(f"{file_prefix}_cre_accessibility.csv")

    # Save CRE type matrix
    cre_type_df = pd.DataFrame(
        {f"Type_{i}": sim_res["cre_type"][:, i] for i in range(num_cre_types)},
        index=cre_indices
    )
    cre_type_df.to_csv(f"{file_prefix}_cre_type.csv")

    # Save CRE embeddings
    cre_embeddings_df = pd.DataFrame(
        sim_res["cre_embeddings"],
        index=cre_indices,
        columns=[f"Embed_dim_{i}" for i in range(dim_cre_embeddings)]
    )
    cre_embeddings_df.to_csv(f"{file_prefix}_cre_embeddings.csv")

    # Save Gene-CRE distances
    cre_gene_dist_df = pd.DataFrame([
        {"Gene": gene, "CRE": cre, "Distance": dist}
        for (gene, cre), dist in sim_res["cre_gene_distances"].items()
    ])
    cre_gene_dist_df.to_csv(f"{file_prefix}_gene_cre_distances.csv", index=False)


def read_simulated_data(file_prefix="simulated"):
    """
    Reads simulated gene expression, CRE activity, CRE accessibility, 
    CRE type, CRE embeddings, and gene-CRE distance data from CSV files.

    Args:
        file_prefix (str): Prefix for input filenames.

    Returns:
        dict: A dictionary containing the loaded data.
    """
    # Read gene expression
    gene_expr_df = pd.read_csv(f"{file_prefix}_gene_expression.csv", index_col=0)

    # Read CRE activity
    cre_activity_df = pd.read_csv(f"{file_prefix}_cre_activity.csv", index_col=0)

    # Read CRE accessibility
    cre_accessibility_df = pd.read_csv(f"{file_prefix}_cre_accessibility.csv", index_col=0)

    # Read CRE type matrix
    cre_type_df = pd.read_csv(f"{file_prefix}_cre_type.csv", index_col=0)

    # Read CRE embeddings
    cre_embeddings_df = pd.read_csv(f"{file_prefix}_cre_embeddings.csv", index_col=0)
    cre_embeddings = cre_embeddings_df.to_numpy()  # Convert to NumPy array for compatibility

    # Read Gene-CRE distances
    cre_gene_dist_df = pd.read_csv(f"{file_prefix}_gene_cre_distances.csv")

    # Extract states dynamically
    states = list(gene_expr_df.columns)

    # Organize data into a dictionary
    data = {
        "gene_expression": {state: gene_expr_df[state].to_dict() for state in states},
        "cre_activity": {state: cre_activity_df[state].to_dict() for state in states},
        "cre_accessibility": {state: cre_accessibility_df[state].to_dict() for state in states},
        "cre_type": cre_type_df.to_numpy(),  # Convert back to NumPy array
        "cre_embeddings": cre_embeddings,  # Store as NumPy array
        "gene_cre_distances": cre_gene_dist_df.to_dict(orient="records")
    }

    return data

