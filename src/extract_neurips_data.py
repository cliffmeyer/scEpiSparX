import numpy as np
import pandas as pd
import scanpy as sc
import generate_embeddings as embed

def list_cell_types_and_atac_regions(adata):
    # List unique cell types
    cell_types = adata.obs['cell_type'].unique().tolist()
    print("Cell Types:", cell_types)
    
    # List 10 ATAC-seq regions
    atac_regions = adata.var_names[:10].tolist()
    print("10 ATAC-seq regions:", atac_regions)
    
    # Extract genomic coordinates
    coords = adata.var.index.to_series().str.extract(r'(?P<chrom>chr[\dXYM]+)[:-](?P<start>\d+)[-](?P<end>\d+)')
    print("Extracted Genomic Coordinates:", coords.head(10))
    
    return cell_types, atac_regions, coords


def extract_and_save_data(adata, cell_types, sample_names, feature_types, 
        n_components = 16, min_cells = 10,
        gex_outfile='', atac_outfile=''):
    """
    Extracts and aggregates data from an AnnData object based on specified cell types and sample names.

    Args:
        adata (AnnData): Input AnnData object containing multiome data.
        cell_types (list of str): List of cell types to extract.
        sample_names (list of str): List of sample names to filter.
        feature_types (list of str): List of feature types to keep (e.g., 'GEX', 'ATAC').
        n_components (int): Dimension of region embeddings 

    Outputs:
        Aggregated read counts per cell type, saved as new AnnData objects.
    """

    # Filter for specific cell types and sample names
    if sample_names:
        subset_obs = adata.obs[
            adata.obs['cell_type'].isin(cell_types) & adata.obs['Samplename'].isin(sample_names)
        ]
    else:
        subset_obs = adata.obs[
            adata.obs['cell_type'].isin(cell_types)
        ]

    adata_subset = adata[subset_obs.index, :]

    # Separate GEX and ATAC into two AnnData objects
    gex_vars = adata_subset.var[adata_subset.var['feature_types'] == 'GEX'].index
    atac_vars = adata_subset.var[adata_subset.var['feature_types'] == 'ATAC'].index

    adata_gex = adata_subset[:, gex_vars]
    adata_atac = adata_subset[:, atac_vars]
    adata_gex.X = adata_gex.layers['counts']
    adata_atac.X = adata_atac.layers['counts']

    sc.pp.filter_genes(adata_atac, min_cells=min_cells)
    print(f"Remaining features after filtering: {adata_atac.shape[1]}")

    adata_atac.varm["embeddings"] = embed.generate_pssm_svd_embeddings(adata_atac, n_components=n_components)
    embeddings = adata_atac.varm['embeddings']
    has_nans = np.isnan(embeddings).any()
    print("Embeddings contain NaNs:" , has_nans)

    # TODO: Sum up reads for all cells by cell type using scanpy's aggregation
    adata_gex_agg = sc.get.aggregate(adata_gex, by='cell_type', func='sum')
    adata_atac_agg = sc.get.aggregate(adata_atac, by='cell_type', func='sum')
    print( 'gex.X', adata_gex_agg.X )
    print( 'layers', adata_gex.layers.keys() )

    adata_gex_agg.X = adata_gex_agg.layers['sum']
    adata_atac_agg.X = adata_atac_agg.layers['sum']

    # Check the data type of the embeddings
    embeddings = adata_atac.varm["embeddings"]

    if isinstance(embeddings, pd.DataFrame):
        # If it's a DataFrame, use .loc to align with var_names
        adata_atac_agg.varm["embeddings"] = embeddings.loc[adata_atac_agg.var_names]
    elif isinstance(embeddings, np.ndarray):
        # If it's a NumPy array, slice based on the number of aggregated features
        adata_atac_agg.varm["embeddings"] = embeddings[: len(adata_atac_agg.var_names)]
    else:
        raise TypeError(f"Unexpected data type for embeddings: {type(embeddings)}")

    # Save the aggregated AnnData objects
    adata_gex_agg.write( gex_outfile )
    adata_atac_agg.write( atac_outfile )

    print(f"Saved aggregated GEX data to {gex_outfile}")
    print(f"Saved aggregated ATAC data to {atac_outfile}")


def main():

    adata_path = "GSE194122_openproblems_neurips2021_multiome_BMMC_processed.h5ad"
    adata = sc.read_h5ad(adata_path)
    samplenames = ['site1_donor1_multiome']
    #samplenames = []

    list_cell_types_and_atac_regions(adata)
    assert(0)
 

    tcells = ['CD8+ T', 'Lymph prog', 'CD4+ T naive', 'CD4+ T activated', 'NK', 'ILC'] 
    monocytes = ['HSC', 'G/M prog', 'cDC2', 'pDC', 'CD16+ Mono', 'CD14+ Mono'] 
    erythrocytes = ['MK/E prog', 'Proerythroblast', 'Erythroblast', 'Normoblast'] 
    bcells = ['Lymph prog', 'Naive CD20+ B', 'Transitional B', 'Plasma cell']

    gex_file = 'adata_gex_aggregated_eryth_mono.h5ad'
    atac_file = 'adata_atac_aggregated_eryth_mono_ppmi.h5ad'
    extract_and_save_data(adata, erythrocytes + monocytes, samplenames,
        ['GEX', 'ATAC'], gex_outfile = gex_file, atac_outfile = atac_file )


if __name__ == '__main__':
    main()

