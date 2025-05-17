<img src="images/scEpiSparX_logo_text.png" alt="logo" height="100" style="vertical-align: middle;">

# scEpiSparX: Single Cell Analysis of Cis-Regulatory Elements through Regression with Epigenetic Embeddings

**scEpiSparX** is a suite of tools for **cis-regulatory analysis** of **single-cell epigenomic and transcriptomic data**. 
It supports scRNA-seq alone, scRNA-seq with scATAC-seq from similar samples, and barcode-matched multiome scRNA-seq with scATAC-seq input (e.g., from 10X Multiome or SHARE-seq).
The model uses **epigenetic embeddings**—either from Cistrome DB via CistromeSparX, or derived from scATAC-seq data—for modeling and interpretation.
scEpiSparX analyses gene read counts for scRNA-seq data aggregated into cell states, identifying the CREs that regulate these genes.
Given the active CRE-seq, regulatory TFs can be identified using the Cistrome DB TF ChIP-seq similarity search (https://db3.cistrome.org), or using motif-based tools.
CREs are typically located within 100kb of the corresponding gene transcription start sites (TSSs), a span of genomic sequence that is not amenable to effective analysis by DNA sequence alone. 
In this model we use epigenetic embeddings to infer CREs that regulate the genes with variable expression between states. To model gene regulation we use a regulatory potential (RP) 
model to link putative CREs with genes. In this model, epigenetic embeddings are translated to CRE activities using a linear transformation followed by ReLu activations. 
We model the influence of CREs on genes using an exponential decay function of the distance between the CREs and the TSSs, 
contributions of multiple enhancer elements are assumed to be additive, and the promoter and enhancer CREs are modelled allowing for interactions between the two types of CRE.


## Key Features

- Fit cis-regulatory models from:
  - scRNA-seq
  - scRNA-seq and scATAC-seq from similar samples
  - Barcode-matched scRNA-seq and scATAC-seq data (eg 10x Multiome or SHARE-seq)
- Use pre-trained or custom epigenetic embeddings from scATAC-seq data
- Identify cell-type-specific cis-regulatory elements (CREs)
- Build new embeddings from scATAC-seq data
- Evaluate and compare multiple models

---

## Installation

```bash
git clone https://github.com/yourusername/scEpiSparX.git
cd scEpiSparX
pip install -r requirements.txt
```

---

## Usage

Run the tool with:

```bash
python scEpiSparX.py <command> [options]
```

### Available Commands:

#### `fit_model`

Train a cis-regulatory model using a configuration file.

```bash
python scEpiSparX.py fit_model path/to/config.yaml [--verbose] [--output path/to/output]
```

#### `find_cres`

Evaluate CRE activity using a trained model and export BED files (one per cell type).
Use the **<name>all.pth** model finds to run this analysis.
The BED files can be interpreted using the Cistrome Data Browser  **similar cistromes** search at 
https://db3.cistrome.org, or through motifs analysis tools.

```bash
python scEpiSparX.py find_cres path/to/model.pth --config path/to/config.yaml
```

#### `compare_models`

Evaluate and compare multiple models in terms of fit.
Use the **<name>validation.pth** models when comparing model performance.

```bash
python scEpiSparX.py compare_models model1.pth model2.pth ... \
  --configs config1.yaml config2.yaml ... [--verbose]
```

#### `make_embeddings`

Generate epigenetic embeddings from scATAC-seq data in `.h5ad` format.

```bash
python scEpiSparX.py make_embeddings input.h5ad \
  --output_file output.h5ad \
  --embeddings PPMI \
  --n_components 16 \
  --min_cells 10 \
  --min_reads 100 \
  --verbose
```

---


## Configuration File Example: `config_sig_cistrome.yaml`

The modeling behavior of scEpiSparX is driven by a YAML config file. Below is a breakdown of the fields in the example `config_sig_cistrome.yaml`.

### `model` Section

Defines model architecture and data dimensionality:
```yaml
model:
  num_pro_types: 10           # Number of promoter types 
  num_enh_types: 15           # Number of enhancer types (larger for more complex systems)
  num_classes: 3              # Number of output classes (don't change this value)
  num_genes: 2000             # Number of genes modeled
  num_models: 10              # Number of models trained (to avoid poor local solutions)
  max_cres_per_gene: 40       # Max number of CREs per gene
  num_cell_states: 8          # Number of distinct cell states
  cre_decay_distance: 10000   # Distance for exponential decay of CRE effect (10kb works well)
  cre_max_distance: 100000    # Maximum distance to consider a CRE 
```

### `features` Section

Controls the type of input features:
```yaml
features:
  use_signal: true              # Use signal values from Cistrome DB
  use_local_embeddings: false  # Whether to use local (scATAC-derived) embeddings (use **make_embeddings** before using this option.)
```

### `files` Section

Paths to input files:
```yaml
files:
  atac_file: "adata_atac_aggregated_eryth_mono_ppmi.h5ad"  # scATAC AnnData input
  gex_file: "adata_gex_aggregated_eryth_mono.h5ad"          # scRNA AnnData input
  embed_file: "/path/to/ATAC_HG38_regions.h5"               # Epigenetic embeddings (e.g., from CistromeSparX)
```

### `training` Section

Training parameters:
```yaml
training:
  epochs: 4000  # Number of training epochs
```

### `experiment` Section

Metadata for tracking experiments:
```yaml
experiment:
  name: "sig_cistrome"  # Identifier for the experiment
```

You can supply this config file when fitting a model using the `fit_model` command:

```bash
python scEpiSparX.py fit_model config_sig_cistrome.yaml
```



### Key Fields:

#### **`data_files`**
Paths to input files and directories containing gene data, embeddings, and results:

- **`h5_filenames`**: Paths to one or more CistromeSparX HDF5 files. These files can be downloaded from [Cistrome DB Downloads](https://db3.cistrome.org/browser/#/download).
- **`gene_ref_filename`**: Path to the gene reference file. Example command to download:
  ```bash
  wget -c -O hg38.refGene.txt.gz http://hgdownload.soe.ucsc.edu/goldenPath/hg38/database/refGene.txt.gz
  ```
- **`gene_alias_filename`**: Path to the gene alias file. Example:
  [HGNC JSON File](https://storage.googleapis.com/public-download-files/hgnc/json/json/hgnc_complete_set.json)
- **`model_path`**: Path to save PyTorch models.
- **`results_path`**: Path to save results.

#### **`model_params`**
Parameters related to genomic regions and dataset splitting:

- **`max_regions_per_gene`**: Number of putative CRE elements to consider, starting from the transcription start site (TSS) and moving outward.

---

## Outputs

- `.pth`: Trained PyTorch model files
- `.bed`: BED files of CRE activity by cell type
- `.png`/`.pdf`: Heatmaps and diagnostic plots
- `.h5ad`: Processed AnnData objects with embeddings


## Getting Started

To test the software download and preprocess a 10x Multiome dataset:
```bash
python src/extract_neurips_data.py
```
Several `.h5ad` should appear in the `data` folder. 
Edit the paths in the config file `configs/configs_0:sc.yaml`, then 
run the regression model:
```bash
python src/scEpiSparX.py fit_model configs/config_0_sc.yaml
```
Find the active cis-regulatory regions from the most recent model.
```bash
python src/scEpiSparX.py find_cres models/model*all.pth --config configs/config_0_sc.yaml
```

## Citation
If you use **scEpiSparX** in your research, please cite:

```
Meyer C.A., Dandawate A., Taing L., Brown M. 
CistromeSparX: Epigenetic Embeddings for AI Models of Cis-Regulatory Elements
```

---

## Dependencies
- Python 3.10.13  
- Numpy 1.26.4 
- PyTorch 2.2.2  
- CistromeSparX embeddings
- Other dependencies as specified in `requirements.txt`


