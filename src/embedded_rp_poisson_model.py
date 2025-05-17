import torch
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


class EmbeddedRegPotPoissonRegressor(nn.Module):

    def __init__(
        self,
        *,
        num_genes: int,
        max_cres_per_gene: int,
        num_cell_states: int,
        num_pro_types: int,
        num_enh_types: int,
        num_classes: int,
        cre_decay_distance: int,
        cre_max_distance: int,
        embedding_dim: int,
        use_embeddings: bool,
        use_signal: bool = True
    ):
        super().__init__()

        self.num_genes = num_genes
        self.max_cres_per_gene = max_cres_per_gene
        self.num_cell_states = num_cell_states
        self.num_pro_types = num_pro_types
        self.num_enh_types = num_enh_types
        self.num_classes = num_classes

        self.decay_distance = cre_decay_distance
        self.max_distance = cre_max_distance

        self.use_embeddings = use_embeddings # use embeddings
        self.use_signal = use_signal  # use ATAC-seq signal for cell type -- assumes matched ATAC and RNA cell types

        self.embedding_dim = embedding_dim

        # Learnable CRE activity matrices
        self.pro_act = nn.Parameter(torch.randn( self.num_pro_types, self.num_cell_states))  # Promoter activity
        self.enh_act = nn.Parameter(torch.randn( self.num_enh_types, self.num_cell_states))  # Enhancer activity

        self.dropout = torch.nn.Dropout(p=0.05) 

        # Total reads per cell state
        self.raw_log_n_total = nn.Parameter(torch.zeros( self.num_cell_states ))  # Learnable per-cell state log-scale

        # Separate linear layers for mapping embeddings to promoter and enhancer types
        self.embedding_to_pro_type = nn.Linear( self.embedding_dim, self.num_pro_types)
        self.embedding_to_enh_type = nn.Linear( self.embedding_dim, self.num_enh_types)

        # Gene-specific parameters
        self.register_buffer(
            "log_delta",
            torch.full((self.num_classes, self.num_cell_states, self.num_genes), torch.log(torch.tensor(self.decay_distance, dtype=torch.float32)))
        )

        #self.gamma = nn.Parameter(torch.randn(self.num_genes))  # Shape: (num_genes,)
        self.bias = nn.Parameter(torch.ones(self.num_genes)) # * torch.log(torch.tensor(10.0)))

        self.optimizer = None
        self.debug = False


    def summarize_model_state(self):
        """
        Returns a short string representing the model's parameter state.
        Useful for checking consistency after loading.
        """
        summary = []
        for name, param in self.named_parameters():
            stats = f"{name}: shape={tuple(param.shape)}, mean={param.mean():.4f}, std={param.std():.4f}"
            summary.append(stats)
        return "\n".join(summary)


    def freeze_non_gene_parameters(self):
        """Freezes all parameters except gene-specific ones."""
        for name, param in self.named_parameters():
            if name not in [ "bias", "log_delta", "n_total", "raw_log_n_total" ]:  # Keep these trainable
                param.requires_grad = False


    def initialize_gene_parameters(self, num_genes=0):
        """Reset the number of genes and initialize gene-specific parameters."""
        self.num_genes = num_genes
        # Register log_delta buffer: [num_classes, num_cell_states, num_genes]
        log_decay = torch.log(torch.tensor(self.decay_distance, dtype=torch.float32))
        self.register_buffer(
            "log_delta",
            torch.full(
                (self.num_classes, self.num_cell_states, num_genes),
                fill_value=log_decay
            )
        )
        # Per-gene bias term
        self.bias = nn.Parameter(torch.ones(num_genes))

        # TODO move this out of function
        self.set_optimizer()


    def set_optimizer(self):
        # Separate parameters into those with and without weight decay
        decay_names_gene = {'pro_act','bias'}
        lambda_gene = 0.05
        decay_names_enh = {'enh_act'}
        lambda_enh = 0.01
        decay_params_gene = []
        decay_params_enh = []
        no_decay_params = []

        for name, param in self.named_parameters():
            if name in decay_names_gene:
                decay_params_gene.append(param)
            elif name in decay_names_enh:
                decay_params_enh.append(param)
            else:
                no_decay_params.append(param)

        self.optimizer = torch.optim.Adam([
            {'params': no_decay_params, 'weight_decay': 0.0},
            {'params': decay_params_gene, 'weight_decay': lambda_gene},
            {'params': decay_params_enh,'weight_decay': lambda_enh}
        ], lr=1e-2)

        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=100, gamma=0.95
        )


    def forward(self, data):

        eps = 1e-5
        PROXIMAL_INDEX = 0  

        if ( self.use_signal or self.use_embeddings ) == False:
            # TODO check
            log_n_total = self.raw_log_n_total # - self.raw_log_n_total.mean()
            log_mu = log_n_total.unsqueeze(1) + self.bias.unsqueeze(0)
            mu = torch.exp(log_mu).clamp(min=eps)
            return dist.Poisson(rate=mu)

        # use observed ATAC-seq signal as multiplier in multiome datasets
        if self.use_signal:
            cre = data.cre_tensor
        else:
            cre = torch.ones_like(data.cre_tensor) 

        if self.use_embeddings:
            cre_embeddings = data.cre_embeddings_tensor
            # Map embeddings into separate promoter and enhancer spaces
            pro_type = torch.einsum("egm,te->tgm", cre_embeddings, self.embedding_to_pro_type.weight).softmax(dim=0)
            enh_type = torch.einsum("egm,te->tgm", cre_embeddings, self.embedding_to_enh_type.weight).softmax(dim=0)
            # Compute separate promoter and enhancer activities
            pro_act = self.dropout(F.relu(torch.einsum("tgm,tc->cgm", pro_type, self.pro_act)))
            enh_act = self.dropout(F.relu(torch.einsum("tgm,tc->cgm", enh_type, self.enh_act)))

        delta = torch.exp(self.log_delta) + eps  # Ensure deltas are positive
        distances = data.distances_tensor
        cre_class = data.cre_class_tensor
 
        mask = (distances != float('inf')) & (cre != 0)
        reg = torch.zeros([cre.shape[0], cre.shape[1]])

        for class_idx in range(self.num_classes):
            class_mask = cre_class == class_idx

            if 0:
             print( 'A:', distances.shape );print( 'B:', delta[class_idx].shape );print( 'C:', cre.shape )
             print( 'D:', pro_act.shape );print( 'E:', enh_act.shape );print( 'F:', self.bias.shape )

            relative_dist = torch.clamp(torch.abs(distances), max=self.max_distance) / delta[class_idx].unsqueeze(-1)
            exp_term = torch.exp(-relative_dist)

            if self.use_embeddings == False:
                reg += (cre *           exp_term * mask * class_mask).sum(dim=-1) # * self.a[class_idx].unsqueeze(0)
            elif class_idx == PROXIMAL_INDEX:  # PROXIMAL (Promoter)
                reg += (cre * pro_act * exp_term * mask * class_mask).sum(dim=-1) # * self.a[class_idx].unsqueeze(0)
            else:  # UPSTREAM, DOWNSTREAM (Enhancers)
                # TODO include a learnable parameter to determine whether this term should be included for gene gene
                reg += (cre * enh_act * exp_term * mask * class_mask).sum(dim=-1) # * self.a[class_idx].unsqueeze(0)

        # Apply log transformation to prevent overflow
        log_reg = torch.log1p(torch.clamp(reg, min=eps))

        # Centered log-scale for total expression per sample
        log_n_total = self.raw_log_n_total # TODO - self.raw_log_n_total.mean()

        # Additive model for log-expression
        log_mu = log_reg + log_n_total.unsqueeze(1) + self.bias.unsqueeze(0)

        mu = torch.exp(log_mu).clamp(min=eps)

        if self.debug:
            print( 'use embeddings:', self.use_embeddings )
            print( 'cre:', torch.mean(cre) )
            print( 'exp term:', torch.mean(exp_term) )
            print( 'rel dist:', torch.mean(relative_dist) )
            if self.use_embeddings:
                print( 'mean pro type:', torch.mean( pro_type ) )
                print( 'mean enh type:', torch.mean( enh_type ) )
                print( 'mean pro act:', torch.mean( pro_act ) )
                print( 'mean enh act:', torch.mean( enh_act ) )
            print( 'mean raw log n_total:', torch.mean( self.raw_log_n_total ) )
            print( 'max mu:', torch.max( log_mu ) )
            print( 'min mu:', torch.min( log_mu ) )

        debug = False
        if debug:
            if self.use_embeddings:
                print(f"cre_embeddings contain NaNs: {torch.isnan(cre_embeddings).any()}")
                print(f"embedding_type_weight NaNs: {torch.isnan(self.embedding_to_pro_type.weight).any()}")
                print(f"embedding_type_weight NaNs: {torch.isnan(self.embedding_to_enh_type.weight).any()}")
                print(f"pro_type contains NaNs: {torch.isnan(pro_type).any()}")
                print(f"enh_type contains NaNs: {torch.isnan(enh_type).any()}")
                print(f"pro_act contains NaNs: {torch.isnan(pro_act).any()}")
                print(f"enh_act contains NaNs: {torch.isnan(enh_act).any()}")
                print(f"reg contains NaNs: {torch.isnan(reg).any()}")
            print(f"n_total contains NaNs: {torch.isnan(self.raw_log_n_total).any()}")
            print(f"log_mu contains NaNs: {torch.isnan(log_mu).any()}")

        return dist.Poisson(rate=mu)


    def train_model(self, gene_reg_data, num_epochs=500 ):
        if self.optimizer is None:
            self.set_optimizer()
 
        for epoch in range(num_epochs):
            self.optimizer.zero_grad()
            _dist = self.forward(gene_reg_data)
            loss = -_dist.log_prob(gene_reg_data.gene_expression_tensor).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1000.0)
            self.optimizer.step()
            self.scheduler.step()
 
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item()}")
                print(f"Gradient norm: {torch.nn.utils.clip_grad_norm_(self.parameters(), 1e9):.4f}")
 
        return loss


    def evaluate_loss(self, gene_reg_data):
        # No parameter modification: no gradient computation
        self.debug = True
        with torch.no_grad():  # Disable gradient tracking to save memory and computation
            _dist = self.forward(gene_reg_data)  # Forward pass
            loss = -_dist.log_prob(gene_reg_data.gene_expression_tensor).mean()  # Calculate loss
        self.debug = False
        return loss.item()  # Return loss value as a scalar


    def save_model(self, path='', optimizer=None, data_files=None ):

        model_params = {
            "num_genes": self.num_genes,
            "max_cres_per_gene": self.max_cres_per_gene,
            "num_cell_states": self.num_cell_states,
            "num_pro_types": self.num_pro_types,
            "num_enh_types": self.num_enh_types,
            "num_classes": self.num_classes,
            "embedding_dim": self.embedding_dim,
            "use_embeddings": self.use_embeddings,
            "use_signal": self.use_signal,
            # TODO "use_rp": self.use_rp,
            "cre_decay_distance": self.decay_distance,
            "cre_max_distance": self.max_distance
        }

        torch.save({
            "state_dict": self.state_dict(),
            "params": model_params,
            "data_files": data_files,
            "optimizer_state_dict": self.optimizer.state_dict() if self.optimizer else None,
        }, path)


    def plot_cre_heatmaps(self, cell_state_labels=None, pro_type_labels=None, enh_type_labels=None):
        """
        Plot heatmaps for the learnable CRE activity matrices.
    
        Args:
        - cell_state_labels (list, optional): Labels for cell states.
        - pro_type_labels (list, optional): Labels for promoter types.
        - enh_type_labels (list, optional): Labels for enhancer types.
        """

        # Assign default labels if None
        if cell_state_labels is None:
            cell_state_labels = [f"Cell {i}" for i in range(self.num_cell_states)]
        if pro_type_labels is None:
            pro_type_labels = [f"Pro {i}" for i in range(self.num_pro_types)]
        if enh_type_labels is None:
            enh_type_labels = [f"Enh {i}" for i in range(self.num_enh_types)]

        pro_act_np = self.pro_act.detach().cpu().numpy()
        enh_act_np = self.enh_act.detach().cpu().numpy()
    
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
        sns.heatmap(pro_act_np, ax=axes[0], cmap='coolwarm', xticklabels=cell_state_labels, yticklabels=pro_type_labels)
        axes[0].set_title("Promoter Activity")
        axes[0].set_xlabel("Cell States")
        axes[0].set_ylabel("Promoter Types")
    
        sns.heatmap(enh_act_np, ax=axes[1], cmap='coolwarm', xticklabels=cell_state_labels, yticklabels=enh_type_labels)
        axes[1].set_title("Enhancer Activity")
        axes[1].set_xlabel("Cell States")
        axes[1].set_ylabel("Enhancer Types")
    
        plt.tight_layout()
        plt.show()

