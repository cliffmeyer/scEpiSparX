import os
import requests
import gzip
import shutil

import numpy as np
import pandas as pd
import scanpy as sc
from tqdm import tqdm

import make_embeddings as embed


def list_cell_types_and_atac_regions(adata,verbose=False):
    # List unique cell types
    cell_types = adata.obs['cell_type'].unique().tolist()
    
    # List 10 ATAC-seq regions
    atac_regions = adata.var_names[:10].tolist()
    
    # Extract genomic coordinates
    coords = adata.var.index.to_series().str.extract(r'(?P<chrom>chr[\dXYM]+)[:-](?P<start>\d+)[-](?P<end>\d+)')

    if verbose:
        print("Cell Types:", cell_types)
        print("10 ATAC-seq regions:", atac_regions)
        print("Extracted Genomic Coordinates:", coords.head(10))

    return cell_types, atac_regions, coords


def extract_adata(adata, cell_types, sample_names, feature_types, 
        n_components = 16, min_cells = 10):
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

    print( adata.obs.keys() )

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
    return adata_gex, adata_atac


def aggregate_cell_reads( adata ):
    # Sum up reads for all cells by cell type using scanpy's aggregation
    adata_agg = sc.get.aggregate(adata, by='cell_type', func='sum')
    adata_agg.X = adata_agg.layers['sum']

    if "embeddings" in adata.varm.keys():
        # Check the data type of the embeddings
        embeddings = adata.varm["embeddings"]
        has_nans = np.isnan(embeddings).any()
        print("Embeddings contain NaNs:" , has_nans)
    else:
        embeddings = None

    if isinstance(embeddings, type(None)):
        pass
    elif isinstance(embeddings, pd.DataFrame):
        # If it's a DataFrame, use .loc to align with var_names
        adata_agg.varm["embeddings"] = embeddings.loc[adata_agg.var_names]
    elif isinstance(embeddings, np.ndarray):
        # If it's a NumPy array, slice based on the number of aggregated features
        adata_agg.varm["embeddings"] = embeddings[: len(adata_agg.var_names)]
    else:
        raise TypeError(f"Unexpected data type for embeddings: {type(embeddings)}")

    return adata_agg


def make_embeddings( adata_atac, n_components=16 ):  
    adata_atac.varm["embeddings"] = embed.generate_pssm_svd_embeddings(adata_atac, n_components=n_components)
    return adata_atac


def write_adata( adata_agg, outfile ):
    # Save the aggregated AnnData objects
    adata_agg.write( outfile )
    print(f"Saved aggregated data to {outfile}")


def download_dataset_from_GEO(url,output_path):
    if os.path.exists(output_path):
        print(f"{output_path} already exists. Skipping download.")
        return

    print("Downloading dataset...")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total_size = int(r.headers.get('Content-Length', 0))
        with open(output_path, 'wb') as f, tqdm(
            desc=output_path,
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024
        ) as bar:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))
    return


def unzip_gz_file(input_gz_path, output_path=None):
    if output_path is None:
        output_path = input_gz_path[:-3]  # remove .gz

    if os.path.exists(output_path):
        print(f"{output_path} already exists. Skipping unzip.")
        return output_path

    print(f"Unzipping {input_gz_path}...")
    with gzip.open(input_gz_path, 'rb') as f_in:
        with open(output_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    print(f"Unzipped to {output_path}.")
    return output_path


def main(
    adata_infile='',
    atac_outfile='',
    gex_outfile='',
    sample_names='',
    cell_types='',
    dim_embedding = 16,
    min_cells = 10
    ):

    # read multimodal AnnData file
    adata = sc.read_h5ad(adata_infile)
    list_cell_types_and_atac_regions(adata)

    # read multimodal AnnData object and separate into separate scRNA-seq and scATAC-seq objects
    feature_types = ['GEX', 'ATAC']
    adata_gex, adata_atac = extract_adata(
        adata, cell_types, sample_names, feature_types, 
        n_components = dim_embedding, 
        min_cells = min_cells )

    # make epigenetic embeddings
    adata_atac = make_embeddings( adata_atac, n_components=dim_embedding )

    # sum reads for each cell type
    adata_atac_agg = aggregate_cell_reads( adata_atac )
    adata_gex_agg = aggregate_cell_reads( adata_gex )

    # write AnnData files
    write_adata(adata_atac_agg, atac_outfile)
    write_adata(adata_gex_agg, gex_outfile)


if __name__ == '__main__':
    """
    Download, extract, generate embeddings and summarize data from use with scEpiSparX.
    Test data is 10x Genomics scRNA-seq + ATAC-seq multiome data described in:
    Luecken, Malte D., et al. \"A sandbox for prediction and integration of DNA, RNA, and proteins in single cells.\"
    Thirty-fifth conference on neural information processing systems datasets and benchmarks track (Round 2). 2021.
    """

    url = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE194nnn/GSE194122/suppl/GSE194122_openproblems_neurips2021_multiome_BMMC_processed.h5ad.gz"
    gz_path = "data/GSE194122_multiome_BMMC_processed.h5ad.gz"

    sample_names = ['site1_donor1_multiome']

    tcells = ['CD8+ T', 'Lymph prog', 'CD4+ T naive', 'CD4+ T activated', 'NK', 'ILC'] 
    monocytes = ['HSC', 'G/M prog', 'cDC2', 'pDC', 'CD16+ Mono', 'CD14+ Mono'] 
    erythrocytes = ['MK/E prog', 'Proerythroblast', 'Erythroblast', 'Normoblast'] 
    bcells = ['Lymph prog', 'Naive CD20+ B', 'Transitional B', 'Plasma cell']

    # cell_types = erythrocytes + monocytes
    # gex_file = 'data/_adata_gex_aggregated_eryth_mono.h5ad'
    # atac_file = 'data/_adata_atac_aggregated_eryth_mono_ppmi.h5ad'

    cell_types = tcells + bcells
    gex_file = 'data/_adata_gex_aggregated_tcell_bcell.h5ad'
    atac_file = 'data/_adata_atac_aggregated_tcell_bcell_ppmi.h5ad'

    download_dataset_from_GEO(url,gz_path)
    h5ad_path = unzip_gz_file(gz_path)

    main(
        adata_infile=h5ad_path,
        atac_outfile=atac_file,
        gex_outfile=gex_file,
        sample_names=sample_names,
        cell_types=cell_types
    )

