#pip install pandas
#pip uninstall cupy-cuda115
#pip uninstall cupy-cuda11x
#pip install cupy-cuda12x
#pip install scanpy
#pip install scanpy matplotlib pandas anndata h5py

import os
import sys
import scipy
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import scipy.io as sio
import scanpy.external as sce
import matplotlib.pyplot as plt

#wget "https://cdn.parsebiosciences.com/1M_PBMC_T1D_Parse.zip"
#unzip 1M_PBMC_T1D_Parse.zip

sc.settings.verbosity = 1
sc.settings.set_figure_params(dpi=100, fontsize=10, dpi_save=400,
    facecolor = 'white', figsize=(6,6), format='png')

# The DGE_filtered folder contains the expression matrix, genes, and files
# Long runtime
adata = sc.read_mtx('DGE_1M_PBMC.mtx')
adata.write('adata_obj1.h5ad')

# reading in gene and cell data
# longish runtime
gene_data = pd.read_csv('all_genes_1M_PBMC.csv')
cell_meta = pd.read_csv('cell_metadata_1M_PBMC.csv')

# find genes with nan values and filter
gene_data = gene_data[gene_data.gene_name.notnull()]
notNa = gene_data.index
notNa = notNa.to_list()

# remove genes with nan values and assign gene names
adata = adata[:,notNa]
adata.var = gene_data
adata.var.set_index('gene_name', inplace=True)
adata.var.index.name = None
adata.var_names_make_unique()

# add cell meta data to anndata object
adata.obs = cell_meta
adata.obs.set_index('bc_wells', inplace=True)
adata.obs.index.name = None
adata.obs_names_make_unique()

sc.pp.filter_cells(adata, min_counts=100)
sc.pp.filter_genes(adata, min_cells=5)
adata.shape

# Save rawdata with raw counts for later
rawdata = adata

# adata.var['mt'] = adata.var_names.str.startswith('MT-')
sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)

# Scanpy will prepend the string in the save argument with "violin"
# and save it to our figure directory defined in the first step.
sc.pl.violin(adata, ['n_genes_by_counts'], save='_n_genes', jitter=0.4)
sc.pl.violin(adata, ['total_counts'], save='_total_counts', jitter=0.4)
sc.pl.violin(adata, ['pct_counts_mt'], save='_mito_pct', jitter=0.4)

# Check cells counts after filtering before assigning
adata[(adata.obs.pct_counts_mt < 25) & 
(adata.obs.n_genes_by_counts < 5000) & 
(adata.obs.total_counts < 25000),:].shape

# Do the filtering
adata = adata[(adata.obs.pct_counts_mt < 25) & 
(adata.obs.n_genes_by_counts < 5000) & 
(adata.obs.total_counts < 25000),:]

adata.write('rawadata.h5ad')

# Normalize data
# Long runtime
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.25)

adata = adata[:,adata.var.highly_variable]
sc.pp.scale(adata, max_value=10)

# Long runtime
sc.tl.pca(adata, svd_solver='arpack')
sc.pl.pca_variance_ratio(adata, log=True, n_pcs=50, save='') # scanpy generates the filename automatically

#pip3 install igraph
#pip3 install leidenalg

# Long runtime
sc.pp.neighbors(adata, n_neighbors=10, n_pcs=50)
sc.tl.umap(adata)

def one_col_lgd(umap):
    legend = umap.legend(bbox_to_anchor=[1.00, 0.5],
    loc='center left', ncol=1, prop={'size': 6})
    legend.get_frame().set_linewidth(0.0)
    for handle in legend.legendHandles:
        handle.set_sizes([25.0])
    return legend

# Make a umap plot with donor information
donor_umap = sc.pl.umap(adata, color=['sample'],
show=False, palette=sns.color_palette("husl", 24),
    legend_fontsize=6, frameon=True, title='Donor')

lgd = one_col_lgd(donor_umap)

fig = donor_umap.get_figure()
fig.set_size_inches(5, 5)
fig.savefig(str(sc.settings.figdir) + '/umap_lgd_sample',
    dpi=400, bbox_extra_artists=(lgd,), bbox_inches='tight')

sc.tl.leiden(adata, resolution=0.5, flavor="igraph", directed=False, n_iterations=2)

# make umap by cluster
leiden_umap = sc.pl.umap(adata, color=['leiden'],
    show=False, palette=sns.color_palette("husl", 24),
legend_fontsize=6, frameon=True, title='Leiden')

lgd = one_col_lgd(leiden_umap)

fig = leiden_umap.get_figure()
fig.set_size_inches(5, 5)
fig.savefig(str(sc.settings.figdir) + '/umap_lgd_leiden',
    dpi=400, bbox_extra_artists=(lgd,), bbox_inches='tight')

#pip install decoupler
#pip install omnipath

# Query Omnipath and get PanglaoDB
import decoupler as dc
markers = dc.get_resource('PanglaoDB')
markers

# Filter by canonical_marker and human
markers = markers[markers['human'] & markers['canonical_marker'] & (markers['human_sensitivity'] > 0.5)]

# Remove duplicated entries
markers = markers[~markers.duplicated(['cell_type', 'genesymbol'])]
markers

# Run ora for over representation analysis
# Long runtime
dc.run_ora(
    mat=adata,
    net=markers,
    source='cell_type',
    target='genesymbol',
    min_n=3,
    verbose=True,
    use_raw=False
)
acts = dc.get_acts(adata, obsm_key='ora_estimate')

# We need to remove inf and set them to the maximum value observed for pvals=0
acts_v = acts.X.ravel()
max_e = np.nanmax(acts_v[np.isfinite(acts_v)])
acts.X[~np.isfinite(acts.X)] = max_e

acts
df = dc.rank_sources_groups(acts, groupby='leiden', reference='rest', method='t-test_overestim_var')
df
n_ctypes = 3
ctypes_dict = df.groupby('group').head(n_ctypes).groupby('group')['names'].apply(lambda x: list(x)).to_dict()
ctypes_dict

# Plot heatmap of cell types
sc.pl.matrixplot(acts, ctypes_dict, 'leiden', dendrogram=True, standard_scale='var',
                 colorbar_title='Z-scaled scores', cmap='RdBu_r')
annotation_dict = df.groupby('group').head(1).set_index('group')['names'].to_dict()
annotation_dict

# Add cell type column based on annotation
adata.obs['cell_type'] = [annotation_dict[clust] for clust in adata.obs['leiden']]

# Visualize cell types
sc.pl.umap(adata, color='cell_type')

# Check available cell types
print(adata.obs['cell_type'].unique())

# Replace normalized data with raw counts
adata.X = rawdata[adata.obs_names, adata.var_names].X

final_adata = adata
final_adata.write('finaladata.h5ad')

genelist = adata.var_names
import csv


# Save to a CSV file
with open("genelist.csv", "w", newline="") as file:
    writer = csv.writer(file)
    
    # If you want each string in its own row:
    for string in genelist:
        writer.writerow([string])

# select macrophages
specific_cell_type = 'Macrophages'  # Replace with your cell type
maccell_adata = adata[adata.obs['cell_type'] == specific_cell_type].copy()

# Verify the subset
print(maccell_adata.shape)

# Save the subsetted data
maccell_adata.write('maccell_data.h5ad')

# Select gamma delta T-cells
specific_cell_type = 'Gamma delta T cells'  # Replace with your cell type
gdtcell_adata = adata[adata.obs['cell_type'] == specific_cell_type].copy()

# Verify the subset
print(gdtcell_adata.shape)

# Save the subsetted data
gdtcell_adata.write('gdtcell_data.h5ad')

# Select T-cells
specific_cell_type = 'T cells'  # Replace with your cell type
tcell_adata = adata[adata.obs['cell_type'] == specific_cell_type].copy()

# Verify the subset
print(tcell_adata.shape)

# Save the subsetted data
tcell_adata.write('bcell_data.h5ad')

# Select B-cells
specific_cell_type = 'B cells naive'  # Replace with your cell type
bcell_adata = adata[adata.obs['cell_type'] == specific_cell_type].copy()

# Verify the subset
print(bcell_adata.shape)

# Save the subsetted data
bcell_adata.write('bcell_data.h5ad')

# Access the unique sample names
sample_names = tcell_adata.obs['sample'].unique()  # Replace 'sample' with your column name
print(sample_names)