import argparse
import numpy as np
import scanpy as sc
from scipy.sparse import random
from scipy.sparse import dok_matrix, csr_matrix
from scipy.sparse import diags
from sklearn.decomposition import TruncatedSVD
import warnings

def multiply_by_rows(matrix, row_coefs):
    normalizer = dok_matrix((len(row_coefs), len(row_coefs)))
    normalizer.setdiag(row_coefs)
    return normalizer.tocsr().dot(matrix)

def multiply_by_columns(matrix, col_coefs):
    normalizer = dok_matrix((len(col_coefs), len(col_coefs)))
    normalizer.setdiag(col_coefs)
    return matrix.dot(normalizer.tocsr())

def calc_pointwise_mutual_info(counts, exponent=1, neg_val=1, region_axis=1):
    """
    Calculates positive PMI
    args:
        - counts: sparse array csr format
        - exponent: param
        - region_axis: counts matrix is regions x cells if 0, otherwise cells x regions
    returns:
        pmi array in csr format
    """
    eps = 1e-5
    counts = counts.astype(float)
    if region_axis == 1:
        counts = counts.T    

    # TODO check if counts is words x contexts or contexts x words
    # ie regions x cells or cells x regions
    sum_w = np.array(counts.sum(axis=1))[:, 0] + eps  # sum over columns
    sum_c = np.array(counts.sum(axis=0))[0, :] + eps  # sum over rows

    if exponent != 1:
        sum_c = sum_c ** exponent

    sum_total = sum_c.sum()
    sum_w = np.reciprocal(sum_w)
    sum_c = np.reciprocal(sum_c)

    print( 'sum total:', sum_total )

    pmi = csr_matrix(counts)
    pmi = multiply_by_rows(pmi, sum_w)
    pmi = multiply_by_columns(pmi, sum_c)
    pmi = pmi * sum_total

    # take log, eliminate negative values
    data = pmi.data
    mask = data > 0 
    log_data = np.zeros_like(data)
    log_data[mask] = np.log(data[mask])
    # set negative values to zero
    mask = log_data < 0
    log_data[mask] = 0
    pmi.data = log_data
    # remove zeros from sparse matrix
    pmi.eliminate_zeros() 
    return pmi


def generate_pssm_svd_embeddings(regions, n_components=64, n_iter=70, random_state=42, alpha=0.5, verbose=False):
    """
    args:
        - regions: anndata object
        - n_components: number of singular value decomposition components 
        - n_iter: truncated svd parameter
        - random_state: seed

    NOTE: regions loaded in col format
        rows are samples, cols are regions
        col format is fast for col access
    """

    mutual_info_matrix = calc_pointwise_mutual_info( regions.X, exponent=1)

    # Perform Truncated SVD
    svd = TruncatedSVD(n_components=n_components, n_iter=n_iter, random_state=random_state)
    svd.fit(mutual_info_matrix)

    # Print shapes of resulting matrices
    if verbose:
        print("Input matrix shape (MI):", mutual_info_matrix.shape)
        print("Left singular vectors (U):", svd.transform(mutual_info_matrix).shape)
        print("Singular values (s):", svd.singular_values_.shape)
        print("Right singular vectors (Vt):", svd.components_.shape)

    # embeddings
    # left singular vectors
    U = svd.transform(mutual_info_matrix)

    V = svd.components_
    S = diags(svd.singular_values_)
    S_dense = S.toarray()
    # Take the power of the diagonal elements of S
    S_alpha = np.power(np.diag(S_dense),alpha)
    S = np.diag(S_alpha)

    # cell embeddings are the matrix product of V.T and S
    cell_embeddings = V.T @ S

    # Normalize embeddings
    cell_embedding_norms = np.linalg.norm(cell_embeddings, axis=1, keepdims=True)
    cell_embeddings = cell_embeddings / cell_embedding_norms

    regions.obsm["PPMI_embeddings"] = cell_embeddings

    # embeddings are the matrix product of U and S
    feature_embeddings = U @ S

    # Normalize embeddings
    feature_embedding_norms = np.linalg.norm(feature_embeddings, axis=1, keepdims=True)
    feature_embeddings = feature_embeddings / feature_embedding_norms

    return feature_embeddings


def generate_lsi_embeddings(adata, n_components=50, binarize=True):
    """
    Perform Latent Semantic Indexing (LSI) on scATAC-seq data stored in an AnnData object.

    Parameters:
    - adata: AnnData object containing scATAC-seq data.
    - n_components: Number of LSI components (default: 50).
    - binarize: Whether to binarize the data (default: True).

    Returns:
    - embedding matrix for features
    ##- Updated AnnData object with LSI results stored in `adata.obsm["X_lsi"]`.
    """
    if binarize:
        adata.X = (adata.X > 0).astype(np.float32)

    # Compute TF-IDF transformation
    tf = adata.X / adata.X.sum(axis=1)  # Term Frequency (TF)
    idf = np.log(1 + adata.shape[0] / (1 + np.array((adata.X > 0).sum(axis=0))))  # Inverse Document Frequency (IDF)

    X = tf.multiply(idf) 

    # Perform LSI using Truncated SVD
    svd = TruncatedSVD(n_components=n_components)
    feature_embeddings = svd.fit_transform(X)

    return feature_embeddings


def main(args):
    # Load data
    adata = sc.read_h5ad(args.input_file)

    if args.verbose:
        print( "var shape", adata.var.shape )
        print( "obs shape", adata.obs.shape )

    if args.min_reads > 0:
        sc.pp.filter_cells(adata, min_counts=args.min_reads)

    if args.min_promoter_fraction > 0.0:
        if 'promoter_fraction' in adata.obs:
            adata = adata[adata.obs['promoter_fraction'] >= args.min_promoter_fraction].copy()
        else:
            warnings.warn("Skipping promoter fraction filtering: 'promoter_fraction' not found in adata.obs.")

    if args.max_mito_fraction < 1.0:
        if 'mito_fraction' in adata.obs:
            adata = adata[adata.obs['mito_fraction'] <= args.max_mito_fraction].copy()
        else:
            warnings.warn("Skipping mitochondrial fraction filtering: 'mito_fraction' not found in adata.obs.")
 
    num_genes_before = adata.shape[1]
    sc.pp.filter_genes(adata, min_cells=args.min_cells)  
    num_genes_after = adata.shape[1]

    if args.verbose:
        print(f"Number of genes before filtering: {num_genes_before}")
        print(f"Number of genes after filtering: {num_genes_after}")

    # Find embeddings
    if args.embeddings == 'LSI':
        adata.varm["LSI_embeddings"] = generate_lsi_embeddings(adata, n_components=args.n_components, binarize=not args.no_binarize)
        sc.pp.neighbors(adata, use_rep="LSI_embeddings", n_pcs=min(30,args.n_components))
    else:
        adata.varm["PPMI_embeddings"] = generate_pssm_svd_embeddings(adata, n_components=args.n_components, verbose=args.verbose)
        sc.pp.neighbors(adata, use_rep="PPMI_embeddings", n_pcs=min(30,args.n_components))

    # Perform clustering and UMAP on LSI components
    sc.tl.umap(adata)
    sc.tl.leiden(adata, resolution=0.5) 
    sc.pl.umap(adata, color="leiden", title=f"Leiden Clustering of {args.embeddings} on UMAP", save=f"_{args.embeddings}_leiden.png")

    # Ensure adata.X is in CSR format before writing
    if isinstance(adata.X, np.ndarray):
        pass  # Already in dense format
    elif not isinstance(adata.X, np.ndarray):
        adata.X = adata.X.tocsr()  # Convert sparse COO to CSR format

    # Now save the AnnData object
    if args.output_file:
        adata.write(args.output_file)
 
        if args.verbose:
            print(f"Processed data saved to {args.output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Identify latent space using LSI or PPMI-SVD decomposition on scATAC-seq data stored in an AnnData file.\
        The embeddings will be saved in the output AnnData file under varm[\"PPMI_embeddings\"] or varm[\"LSI_emnbeddings\"].")

    parser.add_argument("input_file", type=str, help="Path to the input AnnData (.h5ad) file.")
    parser.add_argument("--output_file", "-o", type=str, default = "", help="Path to save the processed AnnData (.h5ad) file.")
    parser.add_argument("--embeddings", "-e", type=str, default="PPMI", help="Embedding type PPMI or LSI")
    parser.add_argument("--n_components", "-n", type=int, default=16, help="Number of LSI components (default: 16).")
    parser.add_argument("--no_binarize", action="store_true", help="Disable binarization of the data.")
    parser.add_argument("--min_cells", type=int, default=10,    help="Filter peaks.")
    parser.add_argument("--min_reads", type=int, default=100,   help="Filter cells.")
    parser.add_argument("--min_promoter_fraction", type=int, default=0,   help="Filter cells based on promoter reads.")
    parser.add_argument("--max_mito_fraction", type=int, default=1,   help="Filter cells based on mitochondrial reads.")
    parser.add_argument("--verbose", action="store_true", help="Print information to help with debugging.")

    args = parser.parse_args()

    main(args)
