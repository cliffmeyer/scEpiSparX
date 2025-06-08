import torch
import numpy as np
import embedding_utils
#import model_embed
import matplotlib.pyplot as plt
import model_adata_setup as model_setup
import os


def evaluate_cre_influence( cre_embeddings, embedding_to_type, cre_act, atac_signal=None ):
    """
    Evaluate the regulatory influence of CREs based solely on their embeddings.

    Args:
        - cre_embeddings: CRE embeddings (dim_embedding, max_num_cres)
        - embedding_to_type: weight matrix for embedding to type mapping (dim_embedding, num_cre_types)
        - cre_act: activity matrix for CREs (num_cre_types, num_cell_states)

    Returns:
        - Predicted regulatory influence for each CRE (num_cell_states)
    """
    # Ensure deltas are strictly positive
    eps = 1e-6
    
    # Map embeddings into CRE type space
    _cre_type = torch.einsum("em,te->tm", cre_embeddings, embedding_to_type).softmax(dim=0)

    # Compute CRE activity per type (cell_types x max_cres)
    _cre_act = torch.einsum("tm,tc->cm", _cre_type, cre_act)

    # Apply ATAC signal if available (assumed shape: (num_cres,))
    if atac_signal is not None:
        _cre_act *= atac_signal

    return _cre_act


def parse_region_name(name):
    """
    Convert region names to (chrom, start, end) tuples, handling different delimiters.
    """
    if ":" in name and "-" in name:  # Format: chr1:100-200
        chrom, coords = name.split(":")
        start, end = map(int, coords.split("-"))
    elif "-" in name:  # Format: chr1-100-200
        parts = name.split("-")
        chrom, start, end = parts[0], int(parts[1]), int(parts[2])
    else:
        raise ValueError(f"Unexpected region format: {name}")
    
    return (chrom, start, end)


def print_influence_to_bed( reg_influence, regions = [], cell_types = [], n_top=1000, name = '' ):
    # Sort and save top 1000 influential regions
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)

    # TODO move this to another function
    for k,cell_state in enumerate(cell_types):

        sorted_indices = np.argsort(-reg_influence[k])  # Sort descending
        if n_top:
            top_indices = sorted_indices[:n_top]
        else:
            top_indices = sorted_indices
        top_regions = [regions[i] for i in top_indices]
        top_values = reg_influence[k, top_indices]

        bed_file = os.path.join(output_dir, f"{name}_{cell_state}.bed")
        with open(bed_file, "w") as f:
            peak_idx = 0
            for region, value in zip(top_regions, top_values):
                chrom, start, end = region
                region_name = f'peak_{peak_idx}'
                f.write(f"{chrom}\t{int(start)}\t{int(end)}\t{region_name}\t{value:.4f}\n")
                peak_idx += 1

        print(f"Saved top 1000 influential regions for cell state {cell_state} to {bed_file}")

    return reg_influence



def print_influence_to_bedgraph(reg_influence, regions=[], cell_types=[], name=''):
    """
    Save all influential regions as BedGraph files, one per cell type.
    
    Parameters:
    - reg_influence: np.ndarray of shape (num_cell_types, num_regions)
    - regions: list of (chrom, start, end)
    - cell_types: list of cell type names
    - name: suffix for output file names
    """
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)

    for k, cell_state in enumerate(cell_types):
        values = reg_influence[k]
        bedgraph_file = os.path.join(output_dir, f"{name}_{cell_state}.bedgraph")
        with open(bedgraph_file, "w") as f:
            for region, value in zip(regions, values):
                chrom, start, end = region
                f.write(f"{chrom}\t{int(start)}\t{int(end)}\t{value:.4f}\n")
        print(f"Saved BedGraph for cell state '{cell_state}' to {bedgraph_file}")

    return reg_influence



def print_influence_to_table(reg_influence, regions=[], cell_types=[], name=''):
    """
    Prints infuence of CRE of regions in all cell types.
    Outputs tab delimited file with columns chr, start, end, cell-type-0, cell-type-1, ...
    """
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, f"reg_influence_table_{name}.tsv")
    with open(output_file, "w") as f:
        # Write header
        header = ["chr", "start", "end"] + list(cell_types)
        f.write("\t".join(header) + "\n")

        # Write each row
        for idx, (chrom, start, end) in enumerate(regions):
            values = [f"{reg_influence[k, idx]:.4f}" for k in range(len(cell_types))]
            row = [chrom, str(int(start)), str(int(end))] + values
            f.write("\t".join(row) + "\n")


def print_adata_info(adata):
    # Check the structure of atac_adata
    print(adata)
 
    # Check available attributes
    print("Observations (obs):", adata.obs.columns)
    print("Variables (var):", adata.var.columns)
    print("Layers:", adata.layers.keys())
    print("Variable metadata (varm):", adata.varm.keys())
    print("Observations metadata (obsm):", adata.obsm.keys())
 
    # Inspect 'embeddings' in varm
    if "embeddings" in adata.varm:
        print("Shape of embeddings:", adata.varm["embeddings"].shape)
        print("Sample of embeddings:", adata.varm["embeddings"][:5])
 
    # Inspect atac_data.X
    print("Shape of atac_data.X:", adata.X.shape)
    print("Sample values:", adata.X[:5, :5])

    cell_types = [ elem.replace(' ','_') for elem in adata.obs["cell_type"] ]
    cell_types = [ elem.replace('/','_') for elem in cell_types ]
    print("Cell types:", cell_types)


def plot_regulatory_influence_histograms(reg_influence, cell_types, trim_percent=1):
    """
    Plot separate histograms for each cell state in reg_influence.
    
    Parameters:
    - reg_influence: np.ndarray of shape (num_cell_states, num_regions)
    - cell_types: list of cell type names (should match the number of rows in reg_influence)
    """
    num_cell_states = reg_influence.shape[0]

    # Plot each cell state's histogram in a separate subplot
    for k in range(num_cell_states):
        # Trim the top and bottom `trim_percent` of the data
        lower_bound = np.percentile(reg_influence[k], trim_percent)
        upper_bound = np.percentile(reg_influence[k], 100 - trim_percent)
        
        trimmed_data = reg_influence[k][(reg_influence[k] >= lower_bound) & (reg_influence[k] <= upper_bound)]
        
        # Create the histogram for the trimmed data
        #plt.subplot( int(k / 3 + 1) , num_cell_states, int(k % 3 + 1) )  # 1 row, num_cell_states columns
        plt.figure(figsize=(8, 6))
        plt.hist(trimmed_data, bins=100, color='skyblue', edgecolor='black')
        plt.title(f'Cell State: {cell_types[k]}')
        plt.xlabel('Regulatory Influence Score')
        plt.ylabel('Frequency')
        plt.grid(True)
        #plt.tight_layout()  # Adjust layout to avoid overlap
        plt.show()


def compute_cre_influence_for_all_regions(embedding_weights=None, activities=None,
                                          batch_size=1000, atac_adata=None, use_atac_signal=True):
    """
    Compute the regulatory influence for all regions using batched embedding loading.

    Args:
        - embedding_weights: weight matrix for embedding to type mapping
        - activities: activity matrix for CREs (this is a parameter from the model)
        - batch_size: number of regions to process in a single batch
        - atac_adata: AnnData object containing ATAC-seq embeddings
        - use_atac_signal: flag to use ATAC-seq signal

    Returns:
        - Regulatory influence for all regions (num_cell_states, num_regions)
    """

    # Initialize an array to store regulatory influence
    num_cell_states = activities.shape[1]
    regions = model_setup.get_cre_tuples(atac_adata, bin_size=0)
    reg_influence = np.zeros((num_cell_states, len(regions)))

    region_tuple_to_index = {
        parse_region_name(name): i for i, name in enumerate(atac_adata.var_names)
    }

    # Process embeddings in batches
    for batch_start in range(0, len(regions), batch_size):
        batch_end = min(batch_start + batch_size, len(regions))
        batch_regions = regions[batch_start:batch_end]

        # Convert (chrom, start, end) tuples to indices for atac_adata
        batch_indices = [region_tuple_to_index.get(region, None) for region in batch_regions]
        batch_indices = [idx for idx in batch_indices if idx is not None]  # Remove None values

        if not batch_indices:  # Skip batch if no valid indices
            continue

        # Load embeddings and ATAC-seq signal using indices
        embeddings = atac_adata.varm["embeddings"][batch_indices]
        cre_embeddings = torch.tensor(embeddings, dtype=torch.float32).T  # Ensure correct shape

        if use_atac_signal:
            atac_signal = atac_adata.X[:, batch_indices]  # Assuming atac_data.X is (num_cells, num_regions)
            atac_signal = torch.tensor(atac_signal, dtype=torch.float32)
        else:
            atac_signal = None

        # Compute influence for the batch
        reg_influence[:, batch_start:batch_end] = evaluate_cre_influence(
            cre_embeddings, embedding_weights, activities, atac_signal
        ).detach().numpy()

    return reg_influence, regions


def evaluate_cre_activity_and_write( *, model=None, gene_reg_data=None, n_top_regions=1000, model_name='test', 
    output_enhancer=True, output_promoter=False, output_negative=False, output_bedgraph=True ):  
    """
    Args:
        - model: 
        - gene_reg_data: 
        - n_top_regions: number of regions to output to BED file
        - model_name: name to include in output file name
        - output_enhancer (bool): output enhancer CREs (default True)
        - output_promoter (bool): output promoter CREs (default False)
        - output_negative (bool): output negative CREs (default False)
        - output_bedgraph (bool): output values as BEDgraph (default True)
    """

    cell_types = gene_reg_data.atac_adata.obs_names
    cell_types = [ elem.replace(' ','_') for elem in cell_types ]
    cell_types = [ elem.replace('/','_') for elem in cell_types ]
 
    # enhancers
    if output_enhancer:
        reg_influence, regions = compute_cre_influence_for_all_regions( 
            embedding_weights = model.embedding_to_enh_type.weight, 
            activities = model.enh_act,
            atac_adata = gene_reg_data.atac_adata,
            use_atac_signal = model.use_signal
        )

        plot_regulatory_influence_histograms(reg_influence, cell_types)

        print_influence_to_bed( reg_influence, regions = regions, cell_types = cell_types, n_top = n_top_regions, name = f'{model_name}_enhancer_positive' )

        if output_bedgraph: 
            print_influence_to_bedgraph( reg_influence, regions = regions, cell_types = cell_types, name = f'{model_name}_enhancer' )

        if output_negative:
            print_influence_to_bed( reg_influence, regions = regions, cell_types = cell_types, n_top = -n_top_regions, name = f'{model_name}_enhancer_negative' )

    # promoters
    if output_promoter:
        reg_influence, regions = compute_cre_influence_for_all_regions( 
            embedding_weights = model.embedding_to_pro_type.weight, 
            activities = model.pro_act,
            atac_adata = gene_reg_data.atac_adata,
            use_atac_signal = model.use_signal
        )
     
        plot_regulatory_influence_histograms(reg_influence, cell_types)
     
        print_influence_to_bed( reg_influence, regions = regions, cell_types = cell_types, n_top = n_top_regions, name = f'{model_name}_promoter_positive' )

        if output_bedgraph: 
            print_influence_to_bedgraph( reg_influence, regions = regions, cell_types = cell_types, name = f'{model_name}_promoter' )

        if output_negative:
            print_influence_to_bed( reg_influence, regions = regions, cell_types = cell_types, n_top = -n_top_regions, name = f'{model_name}_promoter_negative' )

