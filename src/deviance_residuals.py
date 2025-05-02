import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any

def calculate_deviance_residuals(observed, expected, epsilon=1e-8):
    """
    Calculate the deviance residuals for Poisson regression, with NaN checks at each step.
    
    Parameters:
    - observed: Array of observed gene expression values.
    - expected: Array of expected gene expression values (from the model).
    - epsilon: Small constant to avoid division by zero and log(0).
    
    Returns:
    - dev_res: Deviance residuals (NaN values are handled).
    """
    if np.isnan(observed).sum() > 0:
        print("NaNs detected in observed values.")
    if np.isnan(expected).sum() > 0:
        print("NaNs detected in expected values.")
    if (expected == 0).sum() > 0:
        print("Zeros detected in expected values, which could cause log issues.")
    
    log_term = np.log((observed + epsilon) / (expected + epsilon))
    if np.isnan(log_term).sum() > 0:
        print("NaNs detected in log_term.")
    
    dev_term = 2 * (observed * log_term - (observed - expected))
    if np.isnan(dev_term).sum() > 0:
        print("NaNs detected in dev_term.")
    
    sqrt_term = np.sqrt(np.maximum(dev_term, 0))  # Avoid negative values inside sqrt
    if np.isnan(sqrt_term).sum() > 0:
        print("NaNs detected in sqrt_term.")
    
    sign_term = np.sign(observed - expected)
    dev_res = sign_term * sqrt_term
    if np.isnan(dev_res).sum() > 0:
        print("NaNs detected in deviance residuals.")
    
    return dev_res


def compute_predictions_and_residuals(models, gene_reg_data) -> List[Dict[str, Any]]:
    """Compute predicted rates, residuals, and deviance residuals for each model across all cell states."""
    observed = gene_reg_data.gene_expression_tensor.detach().cpu().numpy()
    all_results = []

    for model in models:
        poisson_dist = model(gene_reg_data)
        expected = poisson_dist.rate.detach().cpu().numpy()
        print( '====>>>>', expected[0:10] )
        expected = np.maximum(expected, 1e-8)  # Avoid division by zero

        residuals = expected - observed
        deviance_residuals = calculate_deviance_residuals(observed, expected)

        all_results.append({
            "observed": observed,
            "expected": expected,
            "residuals": residuals,
            "deviance_residuals": deviance_residuals,
        })

    return all_results


def assign_styles(model_categories):
    style_cycle = ['o','D','s','^','v']
    unique_cats = set(model_categories)

    # Mapping each category to a style from the style_cycle
    cat_to_style = {cat: style_cycle[i % len(style_cycle)] for i, cat in enumerate(unique_cats)}
    styles = [cat_to_style[cat] for cat in model_categories]

    # Define size and marker styles
    size_style = [100, 80, 80, 80, 80]
    color_style = ['grey', 'tomato', 'deepskyblue', 'greenyellow']

    # Map style names to their respective indices in style_cycle
    style_to_index = {style: i for i, style in enumerate(style_cycle)}

    # Assign markers and sizes based on styles
    markers = [style_cycle[style_to_index[style]] for style in styles]
    sizes = [size_style[style_to_index[style]] for style in styles]

    # For color, assign different colors within each category (cycling through the color_style list for each category)
    cat_to_colors = {}
    colors = []

    for cat in model_categories:
        if cat not in cat_to_colors:
            cat_to_colors[cat] = iter(color_style)  # Start a new iterator for each category

        # Get the next color in the cycle for this category
        color = next(cat_to_colors[cat])
        colors.append(color)

    return markers, sizes, colors


def plot_mad_deviance_by_cell_state_rows(
    all_results: List[Dict[str, np.ndarray]],
    model_labels: List[str] = None,
    model_categories: List[int] = None,
    name: str = "",
    cell_types = []
):
    """
    Plot the MAD deviance residuals for each cell state as rows and each model as colored points on that row.

    Parameters:
        all_results (List[Dict[str, np.ndarray]]): Output from compute_predictions_and_residuals
        model_labels (List[str], optional): Labels for models. Defaults to Model 1, 2, ...
        model_categories (List[int], optional): Categories for models. Defaults to Model 1, 2, ...
        name (str, optional): Name to prefix output plot file.
        cell_types (List[str]): List of cell types.
    """
    num_models = len(all_results)
    num_cell_states = all_results[0]['deviance_residuals'].shape[0]
    model_labels = model_labels or [f"Model {i+1}" for i in range(num_models)]
    markers, sizes, colors = assign_styles(model_categories)

    # Compute MAD matrix: shape (num_models, num_cell_states)
    mad_matrix = np.zeros((num_models, num_cell_states))
    for m_idx, result in enumerate(all_results):
        dev_res = result["deviance_residuals"]  # shape: (num_cell_states, num_genes)
        mad_matrix[m_idx] = np.median(np.abs(dev_res), axis=1)

    # Plot: horizontal rows for each cell state
    plt.figure(figsize=(16, max(4, num_cell_states * 0.4)))
    #colors = plt.cm.tab10(np.linspace(0, 1, num_models))

    for c_idx in range(num_cell_states):
        mad_values = mad_matrix[:, c_idx]
        min_val = np.min(mad_values)
        max_val = np.max(mad_values)
        # Grey line under the points
        plt.plot([min_val, max_val], [c_idx, c_idx], color='grey', alpha=0.5, linestyle='-', linewidth=1, zorder=0)

        for m_idx in range(num_models):
            plt.scatter(
                x=mad_matrix[m_idx, c_idx],
                y=c_idx,
                color=colors[m_idx],
                label=model_labels[m_idx] if c_idx == 0 else None,
                s=sizes[m_idx],
                alpha=0.7,
                marker=markers[m_idx]
            )

    if cell_types.empty:
        cell_types = [f"Cell State {i}" for i in range(num_cell_states)]

    #plt.yticks(ticks=np.arange=[f"Cell State {i}" for i in range(num_cell_states)])
    plt.yticks(ticks=np.arange(num_cell_states), labels=cell_types)
    plt.xlabel("Median Absolute Deviance Residual (MAD)")
    plt.title("MAD Deviance Residuals by Cell State and Model")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend(title="Model", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()

    if name:
        plt.savefig(f"{name}_mad_deviance_rows.svg", dpi=300)
    plt.show()


def plot_deviance_residual_histograms_by_cell_state(
    all_results: List[Dict[str, np.ndarray]],
    model_labels: List[str] = None,
    name: str = ""
):
    """
    Plot histograms of deviance residuals for each cell state. For each model, 
    the model's histogram is shown in color and all other models are shown in light grey as a reference.
    Bin width is fixed at 2. Histograms are trimmed to show the central 99% of residuals.

    Parameters:
        all_results (List[Dict[str, np.ndarray]]): Output from compute_predictions_and_residuals.
        model_labels (List[str], optional): Labels for models. Defaults to Model 1, 2, ...
        name (str, optional): Name to prefix output plot file.
    """
    num_models = len(all_results)
    num_cell_states = all_results[0]['deviance_residuals'].shape[0]
    model_labels = model_labels or [f"Model {i+1}" for i in range(num_models)]

    colors = plt.cm.tab10(np.linspace(0, 1, num_models))

    for cell_idx in range(num_cell_states):
        plt.figure(figsize=(10, 10))

        # Combine all residuals for this cell state
        all_cell_residuals = np.concatenate([
            result["deviance_residuals"][cell_idx] for result in all_results
        ])

        # Determine 0.5 and 99.5 percentiles for trimming
        lower_bound = np.percentile(all_cell_residuals, 1.0)
        upper_bound = np.percentile(all_cell_residuals, 99.0)

        # Compute bins with fixed width = 1
        min_val = np.floor(lower_bound / 2) * 2
        max_val = np.ceil(upper_bound / 2) * 2
        bins = np.arange(min_val, max_val + 1, 1)

        for m_idx in range(num_models):
            current_residuals = all_results[m_idx]["deviance_residuals"][cell_idx]
            current_residuals = current_residuals[
                (current_residuals >= lower_bound) & (current_residuals <= upper_bound)
            ]

            other_residuals = [
                all_results[other_idx]["deviance_residuals"][cell_idx]
                for other_idx in range(num_models) if other_idx != m_idx
            ]
        
            if other_residuals:
                other_residuals_concat = np.concatenate(other_residuals)
                other_residuals_concat = other_residuals_concat[
                    (other_residuals_concat >= lower_bound) & (other_residuals_concat <= upper_bound)
                ]

                plt.hist(
                    other_residuals_concat,
                    bins=bins,
                    color='lightgrey',
                    label='Other Models' if m_idx == 0 else None,
                    alpha=0.6,
                    density=True
                )

            plt.hist(
                current_residuals,
                bins=bins,
                color=colors[m_idx],
                label=model_labels[m_idx],
                alpha=0.9,
                histtype='step',
                linewidth=4,
                density=True
            )

            plt.axvline(x=0, color='black', linestyle='dotted', linewidth=1)

        plt.title(f"Deviance Residuals — Cell State {cell_idx}")
        plt.xlabel("Deviance Residual")
        plt.ylabel("Density")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.3)
        plt.tight_layout()

        if name:
            plt.savefig(f"{name}_cellstate_{cell_idx}_deviance_histograms.png", dpi=300)

        plt.show()


