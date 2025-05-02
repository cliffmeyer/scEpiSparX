import itertools
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta, nbinom
from scipy.stats import gamma, poisson

import sim_io

def convert_to_int_beta(values, alpha=2, beta_param=5, scale_factor=10):
    """
    Convert floating point gene expression values to integers using the Beta distribution.

    Args:
        values (ndarray): Array of continuous gene expression values.
        alpha (float): Shape parameter α of the Beta distribution.
        beta_param (float): Shape parameter β of the Beta distribution.
        scale_factor (float): Scaling factor to adjust the range.

    Returns:
        ndarray: Integer-converted gene expression values.
    """
    values = np.clip(values, 0, None)  # Ensure values are non-negative
    normalized_values = values / np.max(values) if np.max(values) > 0 else values  # Normalize to [0,1]
    beta_samples = beta.rvs(alpha, beta_param, size=len(values))  # Sample from Beta distribution
    return np.round(beta_samples * normalized_values * scale_factor).astype(int)


def convert_to_int_poisson(values, scale_factor=10):
    """
    Convert floating point gene expression values to integers using the Negative Binomial distribution.

    Args:
        values (ndarray): Array of continuous gene expression values.
        dispersion (float): Dispersion parameter for the Negative Binomial distribution.
        scale_factor (float): Scaling factor for conversion.

    Returns:
        ndarray: Integer-converted gene expression values.
    """
    values = np.clip(values * scale_factor, 1e-2, None)  # Avoid zero issues
    return poisson.rvs(values).astype(int) 


def define_gene_sets(num_genes, num_states, fraction_active=0.5):
    """
    Randomly partition genes into multiple sets, each active in a different state.

    Args:
        num_genes (int): Total number of genes.
        num_states (int): Number of cell states.
        fraction_active (float): Fraction of genes to be active in each state.

    Returns:
        list of sets: A list where each entry is a set of genes active in a particular state.
    """
    all_genes = np.arange(num_genes)
    np.random.shuffle(all_genes)

    num_active_per_state = int(num_genes * fraction_active)
    state_gene_sets = []

    for i in range(num_states):
        active_genes = set(np.random.choice(all_genes, num_active_per_state, replace=False))
        state_gene_sets.append(active_genes)

    return state_gene_sets


def assign_cis_regulatory_elements_per_gene(num_genes, min_cres_per_gene=2, max_cres_per_gene=5):
    """
    Assign each gene a unique set of cis-regulatory elements (CREs).
    
    Args:
        num_genes (int): Number of genes.
        min_cres_per_gene (int): Minimum number of CREs per gene.
        max_cres_per_gene (int): Maximum number of CREs per gene.

    Returns:
        dict: Mapping of genes to assigned CREs.
    """
    gene_to_cres = {}
    cre_id = 0
    
    for gene in range(num_genes):
        num_cres = np.random.randint( min_cres_per_gene, max_cres_per_gene + 1)
        gene_to_cres[gene] = list(range(cre_id, cre_id + num_cres))
        cre_id += num_cres
    
    return gene_to_cres, cre_id


def create_cre_gene_distance_map(gene_to_cres, max_distance=100):
    """
    Generate a fixed distance map between each CRE and its corresponding gene.
    
    Args:
        gene_to_cres (dict): Mapping of genes to assigned CREs.
        max_distance (int): Maximum possible distance between CRE and gene.
    
    Returns:
        dict: Mapping of (gene, cre) pairs to distances.
    """
    cre_gene_distances = {}
    for gene, cres in gene_to_cres.items():
        for i,cre in enumerate(cres):
            if i == 0:
                distance = np.random.uniform(0, 1)
            else:
                distance = np.random.uniform(1, max_distance)
            sign = np.random.choice([-1, 1])  # Randomly assign sign
            cre_gene_distances[(gene, cre)] = sign * distance  # Assign signed distance
    return cre_gene_distances


import numpy as np

def assign_cre_types(num_cres, num_cre_types):
    """
    Randomly assigns a CRE type to each CRE.
    
    Args:
        num_cres (int): Number of CREs.
        num_cre_types (int): Number of distinct CRE types.

    Returns:
        np.ndarray: A (num_cres,) array where each element is a CRE type index.
    """
    return np.random.randint(0, num_cre_types, size=num_cres)

def generate_cre_one_hot(cre_types, num_cre_types):
    """
    Converts a vector of CRE types into a one-hot encoded matrix.

    Args:
        cre_types (np.ndarray): A (num_cres,) array of CRE type indices.
        num_cre_types (int): Number of distinct CRE types.

    Returns:
        np.ndarray: A (num_cres, num_cre_types) one-hot encoded matrix.
    """
    return np.eye(num_cre_types)[cre_types]

def generate_cre_embeddings(cre_types, num_cre_types, dim_embed):
    """
    Generates an embedding vector for each CRE based on its type.

    Args:
        cre_types (np.ndarray): A (num_cres,) array of CRE type indices.
        num_cre_types (int): Number of distinct CRE types.
        dim_embed (int): Dimension of embedding vectors.

    Returns:
        np.ndarray: A (num_cres, dim_embed) matrix where each row is an embedding vector.
    """
    # Create a random embedding matrix for CRE types
    embedding_matrix = np.random.randn(num_cre_types, dim_embed)

    # Map each CRE's type to its corresponding embedding vector
    return embedding_matrix[cre_types]


def compute_gene_expression(cre_gene_distances, cre_activity, decay_factor=0.05):
    """
    Compute gene expression based on CRE activity and distance-based weighting.
    
    Args:
        cre_gene_distances (dict): Mapping of (gene, cre) pairs to distances.
        cre_activity (np.ndarray): Array of CRE activity levels.
        decay_factor (float): Decay factor for distance-based effect.
    
    Returns:
        np.ndarray: Array of computed gene expression levels.
    """
    num_genes = max(gene for gene, _ in cre_gene_distances.keys()) + 1  # Determine total number of genes
    gene_expression = np.zeros(num_genes)

    for (gene, cre), distance in cre_gene_distances.items():
        gene_expression[gene] += cre_activity[cre] * np.exp(-decay_factor * distance)

    return gene_expression


def simulate_state_data(num_genes, gene_to_cres, cre_gene_distances, state_gene_sets,
                        cre_type_matrix, cre_state_matrix, cre_accessibility,
                        state_index):
    """
    Simulate chromatin accessibility and gene expression for a given state.

    Args:
        num_genes (int): Number of genes.
        gene_to_cres (dict): Mapping of genes to their CREs.
        cre_gene_distances (dict): Mapping of CREs to gene distances.
        state_gene_sets (list of sets): List where each set contains genes active in a specific state.
        cre_type_matrix (numpy.ndarray): One-hot matrix mapping CREs to CRE types.
        cre_state_matrix (numpy.ndarray): Activity levels for each CRE type across states.
        cre_accessibility (numpy.ndarray): Accessibility values for each CRE.
        state_index (int): Index of the state to simulate.

    Returns:
        tuple: (Simulated gene expression, Simulated chromatin accessibility), both as NumPy arrays.
    """

    # Compute CRE activity using matrix multiplication
    cre_activity = np.dot(cre_type_matrix, cre_state_matrix[:, state_index])

    # Overall CRE activity is modulated by accessibility
    overall_cre_activity = cre_accessibility * cre_activity

    print('cre_activity', cre_activity.shape)
    print('cre_access', cre_accessibility.shape)

    # Compute gene expression using aggregated CRE activity
    gene_expression = np.zeros(num_genes)  # Initialize array for gene expression

    for gene, cre_indices in gene_to_cres.items():
        if cre_indices:
            gene_expression[gene] = np.mean(overall_cre_activity[cre_indices])  # Average CRE effect

    return gene_expression, overall_cre_activity


def simulate_system_data(num_genes=100, num_states=3, num_cre_types=4, dim_embed=16):
    """
    Simulate gene expression and CRE activity for a given number of cell states.

    Args:
        num_genes (int): Number of genes to simulate.
        num_states (int): Number of cell states to simulate.
        num_cre_types (int): Number of CRE types.
        dim_emb (int): Dimension of CRE embedding vectors.

    Returns:
        dict: A dictionary containing gene-to-CRE mappings, gene expression, and CRE activity for all states.
    """

    # Assign CREs per gene (fixed across states)
    gene_to_cres, n_cres = assign_cis_regulatory_elements_per_gene(num_genes)

    # Generate CRE type matrix (one-hot encoding)
    cre_types = assign_cre_types(n_cres, num_cre_types)
    cre_type_matrix = generate_cre_one_hot(cre_types, num_cre_types)
    cre_embeddings = generate_cre_embeddings(cre_types, num_cre_types, dim_embed)

    # Generate CRE state matrix (num_cre_types x num_states)
    cre_state_matrix = np.random.uniform(0.0, 20.0, size=(num_cre_types, num_states))

    print( 'cre_type_matrix', cre_type_matrix.shape )
    print( 'cre_state_matrix', cre_state_matrix.shape )

    # Generate CRE accessibility values (intrinsic to each CRE)
    cre_accessibility = np.random.uniform(0.2, 1.0, size=n_cres)

    # Generate a fixed CRE-gene distance map
    cre_gene_distances = create_cre_gene_distance_map(gene_to_cres)

    # Define state-specific gene sets
    state_gene_sets = define_gene_sets(num_genes, num_states)

    # Generate baseline expression per gene
    gene_baseline_expression = {gene: np.random.lognormal(mean=1.0, sigma=0.5) for gene in gene_to_cres.keys()}
    # Convert baseline expression dict to a NumPy array
    gene_baseline_array = np.array([gene_baseline_expression[g] for g in range(num_genes)])


    # Containers for outputs
    gene_expression = []
    cre_activity = []
    cre_activity_int_beta = []
    cre_accessibility_int = []
    gene_expression_int_poisson = []

    # Simulate for each state
    for i in range(num_states):
        state_genes = state_gene_sets[i]  # Retrieve gene sets for this state
 
        # Simulate expression and CRE activity
        gene_exp, cre_act = simulate_state_data(
            num_genes, gene_to_cres, cre_gene_distances, None,
            cre_type_matrix=cre_type_matrix, cre_state_matrix=cre_state_matrix,
            cre_accessibility=cre_accessibility, state_index=i
        )
 
        # Adjust gene expression using NumPy array addition (element-wise)
        gene_exp += gene_baseline_array
 
        # Convert to integer-based distributions
        cre_activity_int_beta.append(convert_to_int_beta(cre_act))
        gene_expression_int_poisson.append(convert_to_int_poisson(gene_exp))
 
        # Store raw outputs
        gene_expression.append(gene_exp)
        cre_activity.append(cre_act)

        cre_accessibility_int.append(convert_to_int_poisson(cre_accessibility))
 

    return {
        "gene_to_cres": gene_to_cres,
        "cre_gene_distances": cre_gene_distances,
        "cre_type": cre_type_matrix,
        "cre_embeddings": cre_embeddings,
        "cre_state_matrix": cre_state_matrix,
        "cre_accessibility": cre_accessibility_int,
        "gene_baseline_expression": gene_baseline_expression,
        "gene_expression": gene_expression,
        "cre_activity": cre_activity,
        "gene_expression_int_poisson": gene_expression_int_poisson,
        "cre_activity_int_beta": cre_activity_int_beta,
    }




# Example usage
sim_res = simulate_system_data()

# Print type and dimensions of the returned data
for key, value in sim_res.items():
    if isinstance(value, np.ndarray):  # If it's a NumPy array, print its shape
        print(f"{key}: type={type(value)}, shape={value.shape}")
    elif isinstance(value, list):  # If it's a list, print its length and shape if applicable
        print(f"{key}: type={type(value)}, length={len(value)}")
        if len(value) > 0 and isinstance(value[0], np.ndarray):
            print(f"  └── First element shape: {value[0].shape}")
    elif isinstance(value, dict):  # If it's a dictionary, print the number of keys
        print(f"{key}: type={type(value)}, num_keys={len(value)}")
    else:  # For any other type, just print it
        print(f"{key}: type={type(value)}")

# Write data
sim_io.write_simulated_data(sim_res, file_prefix="simulated_data")

# Read data
loaded_data = sim_io.read_simulated_data(file_prefix="simulated_data")

# Extract states dynamically
print( sim_res["gene_expression"] )

# Extract values from the dictionary
gene_expression = sim_res["gene_expression"]
gene_expression_int_poisson = sim_res["gene_expression_int_poisson"]
cre_activity = sim_res["cre_activity"]
cre_activity_int_beta = sim_res["cre_activity_int_beta"]

# Generate scatter plots for all possible state pairs

states = list(range(len(gene_expression)))
for state_A, state_B in itertools.combinations(states, 2):
    # Gene expression scatter plot (continuous values)
    plt.scatter(gene_expression[state_A], gene_expression[state_B], alpha=0.5)
    plt.xlabel(f"Gene Expression in {state_A}")
    plt.ylabel(f"Gene Expression in {state_B}")
    plt.title(f"Comparison of Gene Expression Between {state_A} and {state_B}")
    plt.show()

    # Gene expression scatter plot (Poisson-converted values)
    plt.scatter(gene_expression_int_poisson[state_A], 
                gene_expression_int_poisson[state_B], alpha=0.5)
    plt.xlabel(f"Gene Expression in {state_A} (Poisson)")
    plt.ylabel(f"Gene Expression in {state_B} (Poisson)")
    plt.title(f"Comparison of Poisson Gene Expression Between {state_A} and {state_B}")
    plt.show()

    # CRE activity scatter plot (continuous values)
    plt.scatter(cre_activity[state_A], cre_activity[state_B], alpha=0.5)
    plt.xlabel(f"CRE Activity in {state_A}")
    plt.ylabel(f"CRE Activity in {state_B}")
    plt.title(f"Comparison of CRE Activity Between {state_A} and {state_B}")
    plt.show()

    # CRE activity scatter plot (Beta-converted values)
    plt.scatter(cre_activity_int_beta[states.index(state_A)], 
                cre_activity_int_beta[states.index(state_B)], alpha=0.5)
    plt.xlabel(f"CRE Activity in {state_A} (Beta)")
    plt.ylabel(f"CRE Activity in {state_B} (Beta)")
    plt.title(f"Comparison of Beta CRE Activity Between {state_A} and {state_B}")
    plt.show()

# Observed vs. true gene expression for each state
for state in states:
    plt.scatter(gene_expression[state], gene_expression_int_poisson[state], alpha=0.5)
    plt.xlabel(f"True Gene Expression in {state}")
    plt.ylabel(f"Observed Gene Expression in {state} (Poisson)")
    plt.title(f"Observed vs. True Gene Expression in {state}")
    plt.show()

