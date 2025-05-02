import torch
import pandas as pd

# Define constants
PROXIMAL_RANGE = 1000
PROXIMAL = 0  
UPSTREAM = 1  
DOWNSTREAM = 2  

class GeneRegData:
    """Handles loading, preprocessing, and matrix construction for gene and CRE data."""

    def __init__(self, file_prefix="simulated_data", max_cres_per_gene=10):
        self.file_prefix = file_prefix
        self.max_cres_per_gene = max_cres_per_gene
        self.load_data()
        self.preprocess_data()
        self.initialize_matrices()
        self.populate_cre_matrices()

    def load_data(self):
        """Loads gene expression, CRE activity, CRE type, accessibility, embeddings, and gene-CRE distances."""
        self.gene_expression = pd.read_csv(f"{self.file_prefix}_gene_expression.csv", index_col=0)
        self.cre_activity = pd.read_csv(f"{self.file_prefix}_cre_activity.csv", index_col=0)
        self.cre_type = pd.read_csv(f"{self.file_prefix}_cre_type.csv", index_col=0)
        self.cre_accessibility = pd.read_csv(f"{self.file_prefix}_cre_accessibility.csv", index_col=0)
        self.cre_embeddings = pd.read_csv(f"{self.file_prefix}_cre_embeddings.csv", index_col=0)
        self.gene_cre_distances = pd.read_csv(f"{self.file_prefix}_gene_cre_distances.csv")
        print( '===xx==xx>>', self.cre_embeddings.shape )

    def preprocess_data(self):
        """Converts data to tensors and sets up index mappings."""
        self.num_genes = self.gene_expression.shape[0]
        self.gene_expression_tensor = torch.tensor(self.gene_expression.values, dtype=torch.float32).T
        self.chromatin_accessibility = torch.tensor(self.cre_activity.values, dtype=torch.float32).T
        self.cre_type_tensor = torch.tensor(self.cre_type.values, dtype=torch.float32)
        self.cre_embeddings_tensor = torch.tensor(self.cre_embeddings.values, dtype=torch.float32)

        # Index mappings
        self.gene_to_index = {gene: i for i, gene in enumerate(self.gene_expression.index)}
        self.cre_to_index = {cre: i for i, cre in enumerate(self.cre_activity.index)}

        # Identify closest peaks per gene
        self.closest_peaks = {
            gene: list(zip(
                self.gene_cre_distances.loc[self.gene_cre_distances["Gene"] == self.gene_to_index[gene]]
                .sort_values(by="Distance", key=abs)
                .iloc[:self.max_cres_per_gene]["CRE"],
                self.gene_cre_distances.loc[self.gene_cre_distances["Gene"] == self.gene_to_index[gene]]
                .sort_values(by="Distance", key=abs)
                .iloc[:self.max_cres_per_gene]["Distance"]
            ))
            for gene in self.gene_expression.index
        }

    @staticmethod
    def assign_cre_class(dist, promoter_dist=PROXIMAL_RANGE):
        """Assigns CRE class based on proximity to the gene."""
        if -promoter_dist <= dist <= promoter_dist:
            return PROXIMAL
        elif dist < 0:
            return UPSTREAM
        else:
            return DOWNSTREAM

    def initialize_matrices(self):
        """Initialize distance, CRE, and embedding matrices."""
        num_cell_states = self.chromatin_accessibility.shape[0]
        num_cre_types = self.cre_type_tensor.shape[1]
        embedding_dim = self.cre_embeddings_tensor.shape[1]
        print( '===xx>>', self.cre_embeddings_tensor.shape )

        shape_cre = (num_cell_states, self.num_genes, self.max_cres_per_gene)
        shape_dist = (self.num_genes, self.max_cres_per_gene)
        shape_cre_type = (num_cre_types, self.num_genes, self.max_cres_per_gene)
        shape_cre_embeddings = (embedding_dim, self.num_genes, self.max_cres_per_gene)

        self.cre = torch.zeros(shape_cre)
        self.distances = torch.full(shape_dist, float("inf"))
        self.cre_type_matrix = torch.zeros(shape_cre_type)
        self.cre_class = torch.full(shape_dist, -1, dtype=torch.int)
        self.cre_embeddings_matrix = torch.zeros(shape_cre_embeddings)

    def populate_cre_matrices(self):
        """Populate CRE activity, distance, type, and embedding matrices."""
        num_cell_states = self.chromatin_accessibility.shape[0]

        def populate_for_state(cre_access):
            """Populate matrices for a single cell state."""
            for gene, peak_data in self.closest_peaks.items():
                gene_idx = self.gene_to_index[gene]
                for j, (cre_idx, dist_value) in enumerate(peak_data):
                    category = self.assign_cre_class(dist_value)
                    self.distances[gene_idx, j] = abs(dist_value)
                    self.cre_type_matrix[:, gene_idx, j] = self.cre_type_tensor[cre_idx]
                    self.cre_class[gene_idx, j] = category
                    self.cre[:, gene_idx, j] = cre_access[cre_idx]
                    self.cre_embeddings_matrix[:, gene_idx, j] = self.cre_embeddings_tensor[cre_idx]

        for cell_state in range(num_cell_states):
            populate_for_state(self.chromatin_accessibility[cell_state, :])

    def get_processed_data(self):
        """Return the processed tensors, including CRE embeddings."""
        return (
            self.gene_expression_tensor, 
            self.cre, 
            self.distances, 
            self.cre_type_matrix, 
            self.cre_class, 
            self.cre_embeddings_matrix
        )

    def print_shapes(self):
        """Prints the shapes of key tensors for verification."""
        print("Gene Expression Tensor Shape:", self.gene_expression_tensor.shape)
        print("Chromatin Accessibility Shape:", self.chromatin_accessibility.shape)
        print("CRE Shape:", self.cre.shape)
        print("CRE Type Shape:", self.cre_type_matrix.shape)
        print("Distances Shape:", self.distances.shape)
        print("CRE Class Shape:", self.cre_class.shape)
        print("CRE Embeddings Shape:", self.cre_embeddings_matrix.shape)


# Example usage:
def model_setup():
    """Function to execute the processing pipeline and return tensors."""
    data = GeneRegData()
    data.print_shapes()
    return data.get_processed_data()


if __name__ == '__main__':
    model_setup()

