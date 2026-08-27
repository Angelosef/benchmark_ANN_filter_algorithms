import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import src.datasets.all_datasets
from src.datasets.base_dataset import Dataset
from src.analysis.utils import find_selectivity_path


class DatasetAnalyser:
    def __init__(self, output_dir='analysis'):
        self.output_folder = os.path.join(output_dir, 'dataset')
        self.plot_dir = os.path.join(self.output_folder, 'plots')
        self.table_dir = os.path.join(self.output_folder, 'tables')

        os.makedirs(self.plot_dir, exist_ok=True)
        os.makedirs(self.table_dir, exist_ok=True)

    def get_selectivity(self, ds_name, ds_subset_size, neighbors_retrieved=10, ds_query_param=None):
        cls_name = Dataset.get_dataset_class(ds_name)
        ds = cls_name(ds_subset_size, neighbors_retrieved)
        base_size = ds.get_base_count()
        sel_path = find_selectivity_path(ds_name, ds_subset_size, neighbors_retrieved, ds_query_param)
        selectivities = np.load(sel_path)
        # normalized to [0.0, 1.0]
        selectivities = selectivities / base_size

        return selectivities

    def plot_selectivity_pdf(self, ds_name, ds_subset_size, neighbors_retrieved=10, ds_query_param=None, bins=50):
        """Plots Probability Density Function (PDF) with a log-scaled selectivity axis."""
        selectivities = self.get_selectivity(ds_name, ds_subset_size, neighbors_retrieved, ds_query_param)

        # Shift zero-values slightly to avoid log(0) domain errors
        eps = 1e-6
        selectivities_log = np.where(selectivities == 0, eps, selectivities)

        fig, ax = plt.subplots(figsize=(8, 5))

        sns.histplot(
            selectivities_log, 
            kde=True, 
            bins=bins, 
            log_scale=True,  # Log scale for the selectivity axis
            stat='density', 
            color='royalblue',
            edgecolor='black',
            alpha=0.6,
            ax=ax
        )

        ax.set_xlabel('Normalized Selectivity (Log Scale)')
        ax.set_ylabel('Density')
        ax.set_title(f'Selectivity PDF (Log Scale): {ds_name} (subset={ds_subset_size}, q_param={ds_query_param})')
        ax.grid(True, which='both', linestyle='--', alpha=0.5)

        filename = f"{ds_name}_{ds_subset_size}_{ds_query_param}_selectivity_pdf.png"
        save_path = os.path.join(self.plot_dir, filename)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()

    def plot_selectivity_cummulative(self, ds_name, ds_subset_size, neighbors_retrieved=10, ds_query_param=None):
        """Plots Cumulative Distribution Function (CDF) with a log-scaled selectivity axis."""
        selectivities = self.get_selectivity(ds_name, ds_subset_size, neighbors_retrieved, ds_query_param)

        eps = 1e-6
        selectivities_log = np.where(selectivities == 0, eps, selectivities)

        fig, ax = plt.subplots(figsize=(8, 5))

        sns.ecdfplot(
            data=selectivities_log, 
            color='crimson', 
            linewidth=2, 
            log_scale=True,  # Log scale for the selectivity axis
            ax=ax
        )

        ax.set_xlabel('Normalized Selectivity (Log Scale)')
        ax.set_ylabel('Cumulative Probability (ECDF)')
        ax.set_title(f'Selectivity CDF (Log Scale): {ds_name} (subset={ds_subset_size}, q_param={ds_query_param})')
        ax.set_ylim(0.0, 1.05)
        ax.grid(True, which='both', linestyle='--', alpha=0.5)

        filename = f"{ds_name}_{ds_subset_size}_{ds_query_param}_selectivity_cdf.png"
        save_path = os.path.join(self.plot_dir, filename)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()

    def calculate_cooccurance_matrix(self, ds_name, ds_subset_size, neighbors_retrieved=10, ds_query_param=None, min_cooccur=5):
        """Calculates co-occurrence matrix and Lift values for all co-occurring tag pairs.
        
        Args:
            min_cooccur (int): Minimum joint frequency required to include a pair 
                               (filters out noise from rare tag pairs).
        Returns:
            lift_values (np.ndarray): Array of Lift ratios for valid co-occurring pairs.
            C (sp.csr_matrix): Sparse co-occurrence matrix (Count(Tag i AND Tag j)).
        """
        cls_name = Dataset.get_dataset_class(ds_name)
        ds = cls_name(ds_subset_size, neighbors_retrieved)

        # shape: (N_data_points x N_distinct_tags), binary CSR matrix
        tag_csr = ds.get_base_attributes()
        N = tag_csr.shape[0]

        # 1. Sparse matrix multiplication: C[i, j] = count(Tag i AND Tag j)
        C = tag_csr.T.dot(tag_csr).tocsr()
        C.setdiag(0)  # Ignore self-co-occurrence (diagonal)
        C.eliminate_zeros()

        # 2. Extract marginal counts for each tag
        tag_counts = np.asarray(tag_csr.sum(axis=0)).flatten()

        # 3. Extract non-zero co-occurrences
        row_indices, col_indices = C.nonzero()
        co_occurrences = C.data

        # Filter out rare co-occurrences to avoid extreme lift values from tiny sample sizes
        valid_mask = co_occurrences >= min_cooccur
        rows = row_indices[valid_mask]
        cols = col_indices[valid_mask]
        counts = co_occurrences[valid_mask]

        # 4. Compute Lift: N * Count(X, Y) / (Count(X) * Count(Y))
        N_float = float(N)
        counts_float = counts.astype(np.float64)
        tag_counts_float = tag_counts.astype(np.float64)

        lift_values = (N_float * counts_float) / (tag_counts_float[rows] * tag_counts_float[cols])

        return lift_values, C

    def plot_lift_distribution(self, ds_name, ds_subset_size, neighbors_retrieved=10, ds_query_param=None, min_cooccur=5):
        """Plots the PDF and CDF of tag Lift values to visualize global tag correlation."""
        lift_values, _ = self.calculate_cooccurance_matrix(
            ds_name, ds_subset_size, neighbors_retrieved, ds_query_param, min_cooccur=min_cooccur
        )

        if len(lift_values) == 0:
            print(f"No tag pairs found with co-occurrence >= {min_cooccur}.")
            return

        fig, (ax_pdf, ax_cdf) = plt.subplots(1, 2, figsize=(14, 5))

        # 1. PDF of Lift (Log Scale)
        sns.histplot(
            lift_values,
            kde=True,
            log_scale=True,
            bins=50,
            stat='density',
            color='teal',
            edgecolor='black',
            alpha=0.6,
            ax=ax_pdf
        )
        ax_pdf.axvline(1.0, color='red', linestyle='--', linewidth=1.5, label='Independence (Lift = 1)')
        ax_pdf.set_xlabel('Lift Ratio: P(X|Y) / P(X) [Log Scale]')
        ax_pdf.set_ylabel('Density')
        ax_pdf.set_title(f'Tag Correlation PDF: {ds_name}')
        ax_pdf.grid(True, which='both', linestyle='--', alpha=0.5)
        ax_pdf.legend()

        # 2. CDF of Lift (Log Scale)
        sns.ecdfplot(
            data=lift_values,
            log_scale=True,
            color='crimson',
            linewidth=2,
            ax=ax_cdf
        )
        ax_cdf.axvline(1.0, color='black', linestyle='--', linewidth=1.5, label='Independence (Lift = 1)')
        ax_cdf.set_xlabel('Lift Ratio: P(X|Y) / P(X) [Log Scale]')
        ax_cdf.set_ylabel('Cumulative Probability (ECDF)')
        ax_cdf.set_title(f'Tag Correlation CDF: {ds_name}')
        ax_cdf.set_ylim(0.0, 1.05)
        ax_cdf.grid(True, which='both', linestyle='--', alpha=0.5)
        ax_cdf.legend()

        plt.suptitle(f'Tag Co-occurrence Analysis for {ds_name} (min_cooccur={min_cooccur})', y=1.02, fontsize=14)
        plt.tight_layout()

        filename = f"{ds_name}_{ds_subset_size}_{ds_query_param}_tag_lift_distribution.png"
        save_path = os.path.join(self.plot_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Lift distribution plot saved to: {save_path}")

    def table_selectivity_stats(self, ds_name, ds_subset_size, neighbors_retrieved=10, ds_query_param=None):
        """Generates statistical summary table for query selectivities and exports to CSV & LaTeX."""
        selectivities = self.get_selectivity(ds_name, ds_subset_size, neighbors_retrieved, ds_query_param)

        percentiles = [1, 5, 25, 50, 75, 95, 99]
        perc_values = np.percentile(selectivities, percentiles)

        stats_data = {
            'Metric': ['Count', 'Mean', 'Std Dev', 'Min', 'Max'] + [f'P{p}' for p in percentiles],
            'Value': [
                len(selectivities),
                np.mean(selectivities),
                np.std(selectivities),
                np.min(selectivities),
                np.max(selectivities),
                *perc_values
            ]
        }

        df_stats = pd.DataFrame(stats_data)

        # Export CSV
        csv_filename = f"{ds_name}_{ds_subset_size}_{ds_query_param}_selectivity_stats.csv"
        csv_path = os.path.join(self.table_dir, csv_filename)
        df_stats.to_csv(csv_path, index=False)
        
        return df_stats

    def full_table_selectivity_stats(self, ds_list):
        """Generates a combined selectivity statistics table for a list of dataset configs."""
        all_stats = []

        percentiles = [1, 5, 25, 50, 75, 95, 99]

        for item in ds_list:
            # Flexible handling for key names ('ds_name' or 'index_name', 'query_param' or 'ds_query_param')
            ds_name = item.get('ds_name') or item.get('index_name')
            subset_size = item.get('subset_size', 1.0)
            k = item.get('neighbors_retrieved', 10)
            q_param = item.get('query_param') if 'query_param' in item else item.get('ds_query_param')

            selectivities = self.get_selectivity(ds_name, subset_size, k, q_param)
            perc_values = np.percentile(selectivities, percentiles)

            row = {
                'Dataset': ds_name,
                'Subset': subset_size,
                'Query Param': str(q_param) if q_param is not None else 'None',
                'Queries': len(selectivities),
                'Mean': np.mean(selectivities),
                'Std': np.std(selectivities),
                'Min': np.min(selectivities),
                'P1': perc_values[0],
                'P5': perc_values[1],
                'P25': perc_values[2],
                'P50 (Median)': perc_values[3],
                'P75': perc_values[4],
                'P95': perc_values[5],
                'P99': perc_values[6],
                'Max': np.max(selectivities),
            }
            all_stats.append(row)

        df_combined = pd.DataFrame(all_stats)

        # Export CSV
        csv_filename = "full_selectivity_stats.csv"
        csv_path = os.path.join(self.table_dir, csv_filename)
        df_combined.to_csv(csv_path, index=False)
        """
        # Export LaTeX
        tex_filename = "full_selectivity_stats.tex"
        tex_path = os.path.join(self.table_dir, tex_filename)
        df_combined.to_latex(tex_path, index=False, float_format="%.4f")
        print(f"Combined selectivity stats saved to:\n  • {csv_path}\n  • {tex_path}")
        """

        return df_combined

if __name__=='__main__':
    ds_analyser = DatasetAnalyser()
    ds_analyser.plot_selectivity_pdf('GIST', 1.0)
    ds_analyser.plot_selectivity_cummulative('GIST', 1.0)
    ds_analyser.plot_lift_distribution('GIST', 1.0)

    ds_analyser.plot_selectivity_pdf('YFCC', 0.1)
    ds_analyser.plot_selectivity_cummulative('YFCC', 0.1)
    ds_analyser.plot_lift_distribution('YFCC', 0.1)

    datasets = [
        {'index_name': 'SIFT', 'subset_size':1.0, 'neighbors_retrieved':10, 'query_param':1},
        {'index_name': 'SIFT', 'subset_size':1.0, 'neighbors_retrieved':10, 'query_param':2},
        {'index_name': 'SIFT', 'subset_size':1.0, 'neighbors_retrieved':10, 'query_param':3},
        {'index_name': 'GLOVE', 'subset_size':1.0, 'neighbors_retrieved':10, 'query_param':None},
        {'index_name': 'YFCC', 'subset_size':0.1, 'neighbors_retrieved':10, 'query_param':None},
        {'index_name': 'GIST', 'subset_size':1.0, 'neighbors_retrieved':10, 'query_param':None}
    ]
    ds_analyser.full_table_selectivity_stats(datasets)
