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

tcell_adata = sc.read("tcell_data.h5ad")
bcell_adata = sc.read("bcell_data.h5ad")
nkcell_adata = sc.read("nkcell_data.h5ad")
gdtcell_adata = sc.read("gdtcell_data.h5ad")
maccell_adata = sc.read("maccell_data.h5ad")

tcellgenes = tcell_adata.var_names
marker_genes = {
    "gamma d T cells": ["TRDC", "TRGC1/2", "TRDV1/2/3"],
    "CD14+ Mono": ["FCN1", "CD14"],
    "CD16+ Mono": ["TCF7L2", "FCGR3A", "LYN"],
    "ID2-hi myeloid prog": [
        "CD14",
        "ID2",
        "VCAN",
        "S100A9",
        "CLEC12A",
        "KLF4",
        "PLAUR",
    ],
    "cDC1": ["CLEC9A", "CADM1"],
    "cDC2": [
        "CST3",
        "COTL1",
        "LYZ",
        "DMXL2",
        "CLEC10A",
        "FCER1A",
    ],  # Note: DMXL2 should be negative
    "Normoblast": ["SLC4A1", "SLC25A37", "HBB", "HBA2", "HBA1", "TFRC"],
    "Erythroblast": ["MKI67", "HBA1", "HBB"],
    "Proerythroblast": [
        "CDK6",
        "SYNGR1",
        "HBM",
        "GYPA",
    ],  # Note HBM and GYPA are negative markers
    "NK": ["GNLY", "NKG7", "CD247", "GRIK4", "FCER1G", "TYROBP", "KLRG1", "FCGR3A"],
    "ILC": ["ID2", "PLCG2", "GNLY", "SYNE1"],
    "Lymph prog": [
        "VPREB1",
        "MME",
        "EBF1",
        "SSBP2",
        "BACH2",
        "CD79B",
        "IGHM",
        "PAX5",
        "PRKCE",
        "DNTT",
        "IGLL1",
    ],
    "Naive CD20+ B": ["MS4A1", "IL4R", "IGHD", "FCRL1", "IGHM"],
    "B1 B": [
        "MS4A1",
        "SSPN",
        "ITGB1",
        "EPHA4",
        "COL4A4",
        "PRDM1",
        "IRF4",
        "CD38",
        "XBP1",
        "PAX5",
        "BCL11A",
        "BLK",
        "IGHD",
        "IGHM",
        "ZNF215",
    ],  # Note IGHD and IGHM are negative markers
    "Transitional B": ["MME", "CD38", "CD24", "ACSM3", "MSI2"],
    "Plasma cells": ["MZB1", "HSP90B1", "FNDC3B", "PRDM1", "IGKC", "JCHAIN"],
    "Plasmablast": ["XBP1", "RF4", "PRDM1", "PAX5"],  # Note PAX5 is a negative marker
    "CD4+ T activated": ["CD4", "IL7R", "TRBC2", "ITGB1"],
    "CD4+ T naive": ["CD4", "IL7R", "TRBC2", "CCR7"],
    "CD8+ T": ["CD8A", "CD8B", "GZMK", "GZMA", "CCL5", "GZMB", "GZMH", "GZMA"],
    "T activation": ["CD69", "CD38"],  # CD69 much better marker!
    "T naive": ["LEF1", "CCR7", "TCF7"],
    "pDC": ["GZMB", "IL3RA", "COBLL1", "TCF4"],
    "G/M prog": ["MPO", "BCL2", "KCNQ5", "CSF3R"],
    "HSC": ["NRIP1", "MECOM", "PROM1", "NKAIN2", "CD34"],
    "MK/E prog": [
        "ZNF385D",
        "ITGA2B",
        "RYR3",
        "PLCB1",
    ],  # Note PLCB1 is a negative marker
}
marker_genes_in_data = {}
for celltype, genes in marker_genes.items():
    markers_found = []
    for gene in genes:
        if gene in tcellgenes:
            markers_found.append(gene)
    marker_genes_in_data[celltype] = markers_found

print(marker_genes_in_data)

# Find true cell type for tcells
adata = tcell_adata
sc.tl.umap(adata)

t_cell_cts = [
    "CD4+ T activated",
    "CD4+ T naive",
    "CD8+ T",
    "T activation",
    "T naive"
]

for ct in t_cell_cts:
    print(f"{ct.upper()}:")  # print cell subtype name
    sc.pl.umap(
        adata,
        color=marker_genes_in_data[ct],
        vmin=0,
        vmax="p99",  # set vmax to the 99th percentile of the gene count instead of the maximum, to prevent outliers from making expression in other cells invisible. Note that this can cause problems for extremely lowly expressed genes.
        sort_order=False,  # do not plot highest expression on top, to not get a biased view of the mean expression among cells
        frameon=False,
        cmap="Reds",  # or choose another color map e.g. from here: https://matplotlib.org/stable/tutorials/colors/colormaps.html
    )
    print("\n\n\n")  # print white space for legibility

# annotate as T cell naive

# find true cell type for b cells
adata = bcell_adata
sc.tl.umap(adata)

B_plasma_cts = [
    "Naive CD20+ B",
    "B1 B",
    "Transitional B",
    "Plasma cells",
    "Plasmablast",
]

for ct in B_plasma_cts:
    print(f"{ct.upper()}:")  # print cell subtype name
    sc.pl.umap(
        adata,
        color=marker_genes_in_data[ct],
        vmin=0,
        vmax="p99",  # set vmax to the 99th percentile of the gene count instead of the maximum, to prevent outliers from making expression in other cells invisible. Note that this can cause problems for extremely lowly expressed genes.
        sort_order=False,  # do not plot highest expression on top, to not get a biased view of the mean expression among cells
        frameon=False,
        cmap="Reds",  # or choose another color map e.g. from here: https://matplotlib.org/stable/tutorials/colors/colormaps.html
    )
    print("\n\n\n")  # print white space for legibility

# Annotate as naive B cd20+ b cell

# Find true cell type for gamma delta t cells
adata = gdtcell_adata
sc.tl.umap(adata)

t_cell_cts = marker_genes_in_data.keys()
for ct in t_cell_cts:
    print(f"{ct.upper()}:")  # print cell subtype name
    sc.pl.umap(
        adata,
        color=marker_genes_in_data[ct],
        vmin=0,
        vmax="p99",  # set vmax to the 99th percentile of the gene count instead of the maximum, to prevent outliers from making expression in other cells invisible. Note that this can cause problems for extremely lowly expressed genes.
        sort_order=False,  # do not plot highest expression on top, to not get a biased view of the mean expression among cells
        frameon=False,
        cmap="Reds",  # or choose another color map e.g. from here: https://matplotlib.org/stable/tutorials/colors/colormaps.html
    )
    print("\n\n\n")  # print white space for legibility

# Could be "Erythroblast", "ILC"

# Find true cell type for macrophages
adata = maccell_adata
sc.tl.umap(adata)
cell_cts = marker_genes_in_data.keys()
for ct in cell_cts:
    print(f"{ct.upper()}:")  # print cell subtype name
    sc.pl.umap(
        adata,
        color=marker_genes_in_data[ct],
        vmin=0,
        vmax="p99",  # set vmax to the 99th percentile of the gene count instead of the maximum, to prevent outliers from making expression in other cells invisible. Note that this can cause problems for extremely lowly expressed genes.
        sort_order=False,  # do not plot highest expression on top, to not get a biased view of the mean expression among cells
        frameon=False,
        cmap="Reds",  # or choose another color map e.g. from here: https://matplotlib.org/stable/tutorials/colors/colormaps.html
    )
    print("\n\n\n")  # print white space for legibility


# Make labeled matrix
# T cell
sample_names = tcell_adata.obs['sample'].unique()  # Replace 'sample' with your column name
print(sample_names)
donor_disease = []
for sample in sample_names:
    if sample[0] == 'H':
        donor_disease.append('Healthy')
    else:
        donor_disease.append('Diabetes')
# Example donor information DataFrame
donor_info = pd.DataFrame({
    'donor_id': sample_names,
    'disease': donor_disease
})

# Merge with adata.obs
merged_data = tcell_adata.obs.merge(donor_info, left_on='sample', right_on='donor_id', how='left')
tcell_adata.obs = merged_data

# Create SSNMF labeling matrix for t-cells
# Create a binary vector for each condition
is_healthy = (tcell_adata.obs["disease"] == "Healthy").astype(int).values  # 1 if healthy, 0 otherwise
is_diabetes = (tcell_adata.obs["disease"] == "Diabetes").astype(int).values  # 1 if diabetes, 0 otherwise

# Stack these into a matrix of shape (2, # of cells)
tcell_labeled = np.vstack([is_healthy, is_diabetes])

# disease_matrix now has:
# - Row 0: 1 if healthy, 0 otherwise
# - Row 1: 1 if diabetes, 0 otherwise
print(tcell_labeled.shape)  # Should be (2, number of cells)


disease_umap = sc.pl.umap(tcell_adata, color=['disease'],
    show=False, palette=sns.color_palette("husl", 2),
legend_fontsize=6, frameon=True, title='T cell UMAP')

lgd = one_col_lgd(disease_umap)

fig = leiden_umap.get_figure()
fig.set_size_inches(5, 5)
#fig.savefig(str(sc.settings.figdir) + '/umap_lgd_leiden',
    #dpi=400, bbox_extra_artists=(lgd,), bbox_inches='tight')

# B cell
sample_names = bcell_adata.obs['sample'].unique()  # Replace 'sample' with your column name
print(sample_names)
donor_disease = []
for sample in sample_names:
    if sample[0] == 'H':
        donor_disease.append('Healthy')
    else:
        donor_disease.append('Diabetes')
# Example donor information DataFrame
donor_info = pd.DataFrame({
    'donor_id': sample_names,
    'disease': donor_disease
})

# Merge with adata.obs
merged_data = bcell_adata.obs.merge(donor_info, left_on='sample', right_on='donor_id', how='left')
bcell_adata.obs = merged_data

# Create SSNMF labeling matrix for t-cells
# Create a binary vector for each condition
is_healthy = (bcell_adata.obs["disease"] == "Healthy").astype(int).values  # 1 if healthy, 0 otherwise
is_diabetes = (bcell_adata.obs["disease"] == "Diabetes").astype(int).values  # 1 if diabetes, 0 otherwise

# Stack these into a matrix of shape (2, # of cells)
bcell_labeled = np.vstack([is_healthy, is_diabetes])

# disease_matrix now has:
# - Row 0: 1 if healthy, 0 otherwise
# - Row 1: 1 if diabetes, 0 otherwise
print(bcell_labeled.shape)  # Should be (2, number of cells)


disease_umap = sc.pl.umap(bcell_adata, color=['disease'],
    show=False, palette=sns.color_palette("husl", 2),
legend_fontsize=6, frameon=True, title='B cell UMAP')

#lgd = one_col_lgd(disease_umap)

fig = disease_umap.get_figure()
fig.set_size_inches(5, 5)
#fig.savefig(str(sc.settings.figdir) + '/umap_lgd_leiden',
    #dpi=400, bbox_extra_artists=(lgd,), bbox_inches='tight')

# gamma delta tcells
sample_names = gdtcell_adata.obs['sample'].unique()  # Replace 'sample' with your column name
print(sample_names)
donor_disease = []
for sample in sample_names:
    if sample[0] == 'H':
        donor_disease.append('Healthy')
    else:
        donor_disease.append('Diabetes')
# Example donor information DataFrame
donor_info = pd.DataFrame({
    'donor_id': sample_names,
    'disease': donor_disease
})

# Merge with adata.obs
merged_data = gdtcell_adata.obs.merge(donor_info, left_on='sample', right_on='donor_id', how='left')
gdtcell_adata.obs = merged_data

# Create SSNMF labeling matrix for t-cells
# Create a binary vector for each condition
is_healthy = (gdtcell_adata.obs["disease"] == "Healthy").astype(int).values  # 1 if healthy, 0 otherwise
is_diabetes = (gdtcell_adata.obs["disease"] == "Diabetes").astype(int).values  # 1 if diabetes, 0 otherwise

# Stack these into a matrix of shape (2, # of cells)
gdtcell_labeled = np.vstack([is_healthy, is_diabetes])

# disease_matrix now has:
# - Row 0: 1 if healthy, 0 otherwise
# - Row 1: 1 if diabetes, 0 otherwise
print(gdtcell_labeled.shape)  # Should be (2, number of cells)
print(gdtcell_labeled)

disease_umap = sc.pl.umap(gdtcell_adata, color=['disease'],
    show=False, palette=sns.color_palette("husl", 2),
legend_fontsize=6, frameon=True, title='Unspecified/Erythroblast UMAP')

#lgd = one_col_lgd(disease_umap)

fig = disease_umap.get_figure()
fig.set_size_inches(5, 5)
#fig.savefig(str(sc.settings.figdir) + '/umap_lgd_leiden',
    #dpi=400, bbox_extra_artists=(lgd,), bbox_inches='tight')

# Macrophages
sample_names = maccell_adata.obs['sample'].unique()  # Replace 'sample' with your column name
print(sample_names)
donor_disease = []
for sample in sample_names:
    if sample[0] == 'H':
        donor_disease.append('Healthy')
    else:
        donor_disease.append('Diabetes')
# Example donor information DataFrame
donor_info = pd.DataFrame({
    'donor_id': sample_names,
    'disease': donor_disease
})

# Merge with adata.obs
merged_data = maccell_adata.obs.merge(donor_info, left_on='sample', right_on='donor_id', how='left')
maccell_adata.obs = merged_data

# Create SSNMF labeling matrix for t-cells
# Create a binary vector for each condition
is_healthy = (maccell_adata.obs["disease"] == "Healthy").astype(int).values  # 1 if healthy, 0 otherwise
is_diabetes = (maccell_adata.obs["disease"] == "Diabetes").astype(int).values  # 1 if diabetes, 0 otherwise

# Stack these into a matrix of shape (2, # of cells)
maccell_labeled = np.vstack([is_healthy, is_diabetes])

# disease_matrix now has:
# - Row 0: 1 if healthy, 0 otherwise
# - Row 1: 1 if diabetes, 0 otherwise
print(maccell_labeled.shape)  # Should be (2, number of cells)

disease_umap = sc.pl.umap(maccell_adata, color=['disease'],
    show=False, palette=sns.color_palette("husl", 2),
legend_fontsize=6, frameon=True, title='Monocyte UMAP')

#lgd = one_col_lgd(disease_umap)

fig = disease_umap.get_figure()
fig.set_size_inches(5, 5)
#fig.savefig(str(sc.settings.figdir) + '/umap_lgd_leiden',
    #dpi=400, bbox_extra_artists=(lgd,), bbox_inches='tight')

tcell_matrix = tcell_adata.X.transpose()
bcell_matrix = bcell_adata.X.transpose()
gdtcell_matrix = gdtcell_adata.X.transpose()
maccell_matrix = maccell_adata.X.transpose()

from scipy.sparse import csr_matrix, coo_matrix

# Convert to scipy sparse format
tcell_matrix = tcell_matrix.tocoo()
bcell_matrix = bcell_matrix.tocoo()
gdtcell_matrix = gdtcell_matrix.tocoo()
maccell_matrix = maccell_matrix.tocoo()

# Extract row, col, and data for tcells, save csv
rows = tcell_matrix.row
cols = tcell_matrix.col
data = tcell_matrix.data

# Create DataFrame and save to CSV
df = pd.DataFrame({'row': rows, 'col': cols, 'data': data})
df.to_csv('tcell_matrix.csv', index=False)

# Extract row, col, and data for bcells, save csv
rows = bcell_matrix.row
cols = bcell_matrix.col
data = bcell_matrix.data

# Create DataFrame and save to CSV
df = pd.DataFrame({'row': rows, 'col': cols, 'data': data})
df.to_csv('bcell_matrix.csv', index=False)

# Extract row, col, and data for gdtcells, save csv
rows = gdtcell_matrix.row
cols = gdtcell_matrix.col
data = gdtcell_matrix.data

# Create DataFrame and save to CSV
df = pd.DataFrame({'row': rows, 'col': cols, 'data': data})
df.to_csv('gdtcell_matrix.csv', index=False)

# Extract row, col, and data for maccells, save csv
rows = maccell_matrix.row
cols = maccell_matrix.col
data = maccell_matrix.data

# Create DataFrame and save to CSV
df = pd.DataFrame({'row': rows, 'col': cols, 'data': data})
df.to_csv('maccell_matrix.csv', index=False)

# save labeled tcell csv
df = pd.DataFrame(tcell_labeled)
df.to_csv("tcell_labeled.csv", index=False, header=False)

# save labeled bcell csv
df = pd.DataFrame(bcell_labeled)
df.to_csv("bcell_labeled.csv", index=False, header=False)

# save labeled gdtcell csv
df = pd.DataFrame(gdtcell_labeled)
df.to_csv("gdtcell_labeled.csv", index=False, header=False)

# save labeled maccell csv
df = pd.DataFrame(maccell_labeled)
df.to_csv("maccell_labeled.csv", index=False, header=False)
