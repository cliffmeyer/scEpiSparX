
def compute_log_likelihood_loss(model, gene_reg_data):
    """
    Computes the Negative Log-Likelihood (NLL) loss for the given model and data.
    """
    # Get the distribution from the model's forward pass
    _dist = model(gene_reg_data)
    
    # Compute the negative log-likelihood loss (Mean of the log-probability of observed gene expression)
    nll_loss = -_dist.log_prob(gene_reg_data.gene_expression_tensor)
    print( "nll_loss", nll_loss.shape, nll_loss ) 
    nll_loss_mean = nll_loss.mean()
    return nll_loss


def in_silico_deletion_log_likelihoods(model, gene_reg_data):
    """
    Compare the log-likelihood of the full model and the perturbed model for each CRE.
    
    Args:
    - model: The trained model.
    - gene_reg_data: The gene regulation data (including all CREs).
    
    Returns:
    - log_likelihood_diff: A tensor containing the log-likelihood differences for each gene and CRE.
    """
    # Compute the log-likelihood for the full model
    full_model_loss = compute_log_likelihood_loss(model, gene_reg_data)

    # Initialize a tensor to store log-likelihood differences for each CRE (gene x CRE matrix)
    num_cells = gene_reg_data.gene_expression_tensor.shape[0]
    num_genes = gene_reg_data.gene_expression_tensor.shape[1]
    num_cres = gene_reg_data.cre_tensor.shape[2]
    loss_diff = torch.zeros((num_cells, num_genes, num_cres), device=gene_reg_data.gene_expression_tensor.device)
    print( f"genes {num_genes} cells {num_cells} cres {num_cres}" )
 
    # Iterate over each CRE, perturb it (zero it out), and compute the log-likelihood for the perturbed model
    for cre_idx in range(num_cres):
        cre_tensor = gene_reg_data.cre_tensor.clone()
        perturbed_cre_tensor = gene_reg_data.cre_tensor.clone()
        perturbed_cre_tensor[:, :, cre_idx] = 0  # Zero out the selected CRE
        
        # Update gene_reg_data with the perturbed CRE tensor
        gene_reg_data.cre_tensor = perturbed_cre_tensor
        
        # Compute the loss for the perturbed model
        perturbed_model_loss = compute_log_likelihood_loss(model, gene_reg_data)
        
        # Calculate the difference in log-likelihood between the full and perturbed models
        # NOTE: the CREs with the larger effects induce the larger positive loss
        loss_diff[:, :, cre_idx] = - full_model_loss + perturbed_model_loss

        # Reset the data
        gene_reg_data.cre_tensor = cre_tensor

    return loss_diff


def analyze_cre_effects_per_cell(loss_diff):
    """Flatten (gene, CRE) per cell and sort independently."""
    num_genes, num_cells, num_cres = loss_diff.shape

    # Permute (num_genes, num_cells, num_cres) -> (num_cells, num_genes, num_cres), then flatten
    loss_diff_flat = loss_diff.permute(1, 0, 2).reshape(num_cells, num_genes * num_cres)

    # Sort flattened (gene, CRE) indices in descending order of effect for each cell
    sorted_cre_indices = torch.argsort(loss_diff_flat, dim=1, descending=True)

    return sorted_cre_indices


def parse_region(region):
    region = region.replace(":", "-")
    chrom, start, end = region.split("-", 2)
    return chrom, int(start.replace(",", "")), int(end.replace(",", ""))


def write_bed_file(sorted_regions, sorted_scores, bed_file_path):
    """
    Write genomic regions and associated scores to a BED file.

    Args:
    - sorted_regions (list of str): List of genomic regions in 'chrom:start-end' format.
    - sorted_scores (list of float): List of scores for the corresponding regions.
    - bed_file_path (str): Path to the output BED file.

    Returns:
    - None
    """
    assert isinstance(sorted_regions, list), "sorted_regions must be a list"
    assert isinstance(sorted_scores, list), "sorted_scores must be a list"
    assert len(sorted_regions) == len(sorted_scores), "sorted_regions and sorted_scores must have the same length"
    
    with open(bed_file_path, "w") as bed_file:
        for region, score in zip(sorted_regions, sorted_scores):
            chrom, start, end = parse_region(region)
            bed_line = f"{chrom}\t{start}\t{end}\t{score:.6f}\n"  # Build string safely
            bed_file.write(bed_line)


def save_cre_effects_to_bed(loss_diff, cre_indices_tensor, atac_adata, output_dir="bed_files"):
    """ 
    Save sorted CREs per cell into BED files.
    
    Args:
    - loss_diff: Tensor of shape (num_genes, num_cells, num_cres)
    - cre_indices_tensor: Tensor mapping genes to CRE indices (num_genes, num_cres)
    - atac_adata: Anndata object containing var_names (CRE genomic coordinates)
    - output_dir: Directory to save BED files
    """

    # Assertions to ensure correct input types and shapes
    assert isinstance(loss_diff, torch.Tensor), "loss_diff must be a torch.Tensor"
    assert isinstance(cre_indices_tensor, torch.Tensor), "cre_indices_tensor must be a torch.Tensor"
    assert hasattr(atac_adata, "var_names"), "atac_adata must have var_names (CRE genomic coordinates)"
    
    num_cells, num_genes, num_cres = loss_diff.shape
    print( f"genes {num_genes} cells {num_cells} cres {num_cres}" )
 
    assert cre_indices_tensor.shape == (num_genes, num_cres), (
        f"cre_indices_tensor must have shape ({num_genes}, {num_cres}), got {cre_indices_tensor.shape}"
    )

    os.makedirs(output_dir, exist_ok=True)
    # Flatten (gene, CRE) while keeping cell separate
    loss_diff_flat = loss_diff.permute(1, 0, 2).reshape(num_cells, num_genes * num_cres)
    cre_indices_flat = cre_indices_tensor.reshape(num_genes * num_cres).cpu().numpy()  # Flatten CRE indices

    # Convert indices to genomic regions
    cre_names = list(atac_adata.var_names)  # CRE names from ATAC-seq data
    genomic_regions = [cre_names[i] for i in cre_indices_flat]  # Map indices to regions

    # Process per cell
    for cell_idx in range(num_cells):
        sorted_indices = torch.argsort(loss_diff_flat[cell_idx], descending=True)  # Sort per cell

        # Get sorted CREs and their genomic coordinates
        sorted_regions = [genomic_regions[i] for i in sorted_indices.cpu().numpy()]
        #sorted_scores = loss_diff_flat[cell_idx][sorted_indices].cpu().numpy()
        sorted_scores = loss_diff_flat[cell_idx][sorted_indices].detach().cpu().numpy()

        # Write to BED file
        bed_file_path = os.path.join(output_dir, f"cell_{cell_idx}.bed")
        sorted_scores = sorted_scores.tolist()
        write_bed_file(sorted_regions, sorted_scores, bed_file_path)

    print(f"BED files saved in {output_dir}")


def in_silico_deletion_analysis(model, gene_reg_data):
    loss_diff = in_silico_deletion_log_likelihoods(model, gene_reg_data)
    save_cre_effects_to_bed(
        loss_diff, 
        gene_reg_data.cre_indices_tensor, 
        gene_reg_data.atac_adata, 
        output_dir="in_silico_deletion_bed_files"
    )

