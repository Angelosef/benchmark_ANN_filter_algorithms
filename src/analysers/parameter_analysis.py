import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from src.analysers.utils import load_run_data


def clean_param_name(p):
    return p.replace('build_param_', '').replace('query_param_', '')


class ParameterAnalyser:

    def __init__(self, log_root_dir="logs", output_dir="analysis"):
        self.log_root = log_root_dir
        self.plot_dir = os.path.join(output_dir, 'hyperparameters', 'plots')
        os.makedirs(self.plot_dir, exist_ok=True)
        self.registry_path = os.path.join(self.log_root, "master_registry.csv")

        # Persistent palette mapping across method calls
        self.palette = (
            sns.color_palette("tab10")
            + sns.color_palette("Set2")
            + sns.color_palette("Dark2")
        )
        self.PARAM_COLOR_MAP = {}
        self._color_idx = 0

    def _get_param_color(self, param_name):
        """Returns a consistent, distinctive color for a given parameter name across all runs."""
        if param_name not in self.PARAM_COLOR_MAP:
            self.PARAM_COLOR_MAP[param_name] = self.palette[
                self._color_idx % len(self.palette)
            ]
            self._color_idx += 1
        return self.PARAM_COLOR_MAP[param_name]

    @staticmethod
    def _clean_val(val):
        """Unwraps numpy scalars/types into native python primitives for clean printing."""
        if hasattr(val, 'item'):
            val = val.item()
        return val

    def prepare_dataframe(
        self, index_name, dataset_name, ds_subset_size, ds_query_param
    ):
        registry = pd.read_csv(self.registry_path)

        mask = (
            (registry['dataset'] == dataset_name)
            & (registry['index'] == index_name)
            & (registry['subset_size'] == ds_subset_size)
        )

        if ds_query_param is not None:
            mask &= registry['ds_query_param'] == ds_query_param

        filtered = registry[mask]

        run_details = []
        for run_id in filtered['run_id']:
            metadata_path = os.path.join(self.log_root, run_id, 'metadata.json')
            details = load_run_data(metadata_path)
            if details:
                if (
                    'index_file_size' in details
                    and details['index_file_size'] is not None
                ):
                    details['index_file_size'] = details['index_file_size'] / (
                        1024**3
                    )

                # Flatten nested parameter dicts into top-level columns with distinct prefixes
                if 'build_params' in details and isinstance(
                    details['build_params'], dict
                ):
                    for k, v in details['build_params'].items():
                        details[f'build_param_{k}'] = v

                if 'query_params' in details and isinstance(
                    details['query_params'], dict
                ):
                    for k, v in details['query_params'].items():
                        details[f'query_param_{k}'] = v

                run_details.append(details)

        return pd.DataFrame(run_details)

    def plot_query_param_metric_correlation(
        self,
        index_name,
        dataset_name,
        ds_subset_size,
        ds_query_param=None,
        method="spearman",
    ):
        """Plots correlation heatmap showing how BOTH query and build parameters impact query metrics (Recall & Latency)."""
        df = self.prepare_dataframe(
            index_name, dataset_name, ds_subset_size, ds_query_param
        )

        if df.empty:
            print(f"No data found for {index_name} on {dataset_name}.")
            return

        query_metrics = [
            c for c in ['avg_recall', 'avg_latency'] if c in df.columns
        ]

        # Include BOTH query and build parameters that vary across runs
        param_cols = [
            c
            for c in df.columns
            if c.startswith(('query_param_', 'build_param_'))
            and df[c].nunique() > 1
        ]

        # Select only numeric columns
        df_numeric = df.select_dtypes(include=[np.number])
        param_cols = [c for c in param_cols if c in df_numeric.columns]

        if not param_cols or not query_metrics:
            print(
                f"Insufficient varying parameters or metrics for query correlation on {index_name} ({dataset_name})."
            )
            return

        fig, ax = plt.subplots(
            figsize=(6, max(4, len(param_cols) * 0.5 + 1.5))
        )

        self._draw_corr_heatmap(
            df_numeric,
            param_cols,
            query_metrics,
            f"Query Performance Correlation\n({index_name} on {dataset_name})",
            ax,
            method,
        )

        filename = f"{index_name}_{dataset_name}_{ds_subset_size}_{ds_query_param}_query_correlation.png"
        save_path = os.path.join(self.plot_dir, filename)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

    def plot_build_param_metric_correlation(
        self,
        index_name,
        dataset_name,
        ds_subset_size,
        ds_query_param=None,
        method="spearman",
    ):
        """Plots correlation heatmap showing how build parameters impact indexing metrics (TTI & Index File Size)."""
        df = self.prepare_dataframe(
            index_name, dataset_name, ds_subset_size, ds_query_param
        )

        if df.empty:
            print(f"No data found for {index_name} on {dataset_name}.")
            return

        # Build metrics are only driven by build parameters that vary
        build_params = [
            c
            for c in df.columns
            if c.startswith('build_param_') and df[c].nunique() > 1
        ]

        df_numeric = df.select_dtypes(include=[np.number])
        build_params = [c for c in build_params if c in df_numeric.columns]

        # Rename metric columns for cleaner plot display
        rename_map = {
            'build_time': 'TTI',
            'index_file_size': 'Index File Size'
        }
        df_numeric = df_numeric.rename(columns=rename_map)

        # Map build metrics to their new display names
        build_metrics = [
            rename_map[c] for c in ['build_time', 'index_file_size'] if c in df.columns
        ]

        if not build_params or not build_metrics:
            print(
                f"Insufficient varying build parameters or metrics for correlation on {index_name} ({dataset_name})."
            )
            return

        fig, ax = plt.subplots(
            figsize=(6, max(4, len(build_params) * 0.5 + 1.5))
        )

        self._draw_corr_heatmap(
            df_numeric,
            build_params,
            build_metrics,
            f"Construction Resource Correlation\n({index_name} on {dataset_name})",
            ax,
            method,
        )

        filename = f"{index_name}_{dataset_name}_{ds_subset_size}_{ds_query_param}_build_correlation.png"
        save_path = os.path.join(self.plot_dir, filename)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

    def _draw_corr_heatmap(self, df, params, metrics, title, ax, method):
        """Helper to draw filtered correlation heatmap onto a specific plot axis."""
        full_corr = df[params + metrics].corr(method=method)
        filtered_corr = full_corr.loc[params, metrics]

        # Use full prefixed names or clean names for index display
        filtered_corr.index = [clean_param_name(p) for p in filtered_corr.index]

        sns.heatmap(
            filtered_corr,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
            cbar_kws={"label": f"{method.capitalize()} Correlation"},
            ax=ax,
            linewidths=0.5,
        )
        ax.set_title(title, fontsize=11, pad=10)
        ax.set_xlabel("Metrics")
        ax.set_ylabel("Parameters")

    def plot_build_parameters(
        self, index_name, dataset_name, ds_subset_size, ds_query_param=None
    ):
        """Plots (index_file_size, build_time) across build hyperparameter variations."""
        df = self.prepare_dataframe(
            index_name, dataset_name, ds_subset_size, ds_query_param
        )

        if df.empty:
            print(f"No data found for {index_name} on {dataset_name}.")
            return

        param_cols = [c for c in df.columns if c.startswith('build_param_')]

        if not param_cols:
            print(
                f"No build parameters found for {index_name} on {dataset_name}."
            )
            return

        # Aggregate runs across query params since query params don't affect build metrics
        agg_dict = {'index_file_size': 'mean', 'build_time': 'mean'}
        df = df.groupby(param_cols, as_index=False).agg(agg_dict)

        varying_params = [c for c in param_cols if df[c].nunique() > 1]
        constant_params = [c for c in param_cols if df[c].nunique() == 1]

        varying_params.sort(key=lambda c: df[c].nunique(), reverse=True)

        fig, ax = plt.subplots(figsize=(10, 6.5))

        param_colors = {
            p_col: self._get_param_color(clean_param_name(p_col))
            for p_col in varying_params
        }

        unique_vals = {
            p_col: [self._clean_val(v) for v in sorted(df[p_col].unique())]
            for p_col in varying_params
        }

        root_query = {p_col: unique_vals[p_col][0] for p_col in varying_params}

        root_mask = pd.Series(True, index=df.index)
        for p_col, val in root_query.items():
            root_mask &= df[p_col] == val

        root_df = df[root_mask]

        ax.scatter(
            df['index_file_size'], df['build_time'], color='black', zorder=5, s=35
        )

        if not root_df.empty:
            root_row = root_df.iloc[0]
            ax.scatter(
                root_row['index_file_size'],
                root_row['build_time'],
                color='gold',
                edgecolor='black',
                zorder=6,
                s=130,
            )

        def draw_tree_level(current_fixed_params, level_idx):
            if level_idx >= len(varying_params):
                return

            curr_param = varying_params[level_idx]
            vals = unique_vals[curr_param]
            color = param_colors[curr_param]

            for i in range(len(vals) - 1):
                val_curr, val_next = vals[i], vals[i + 1]

                start_mask = pd.Series(True, index=df.index)
                for p, v in current_fixed_params.items():
                    start_mask &= df[p] == v
                start_mask &= df[curr_param] == val_curr

                end_mask = pd.Series(True, index=df.index)
                for p, v in current_fixed_params.items():
                    end_mask &= df[p] == v
                end_mask &= df[curr_param] == val_next

                start_df = df[start_mask]
                end_df = df[end_mask]

                if not start_df.empty and not end_df.empty:
                    s_row, e_row = start_df.iloc[0], end_df.iloc[0]
                    ax.annotate(
                        '',
                        xy=(e_row['index_file_size'], e_row['build_time']),
                        xytext=(s_row['index_file_size'], s_row['build_time']),
                        arrowprops=dict(
                            arrowstyle="->", color=color, lw=1.8, alpha=0.85
                        ),
                    )

            if level_idx + 1 < len(varying_params):
                for v in vals:
                    next_fixed = current_fixed_params.copy()
                    next_fixed[curr_param] = v
                    draw_tree_level(next_fixed, level_idx + 1)

        if varying_params:
            draw_tree_level({}, 0)

        legend_handles = []

        root_str = ", ".join(
            [
                f"{clean_param_name(p)}={unique_vals[p][0]}"
                for p in varying_params
            ]
        )
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color='gold',
                marker='o',
                markeredgecolor='black',
                linestyle='None',
                markersize=9,
                label=f'Root ({root_str})',
            )
        )

        for p_col in varying_params:
            p_name = p_col.replace('build_param_', '')
            vals_str = str(unique_vals[p_col])
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    color=param_colors[p_col],
                    lw=2,
                    label=f'Increment {p_name}: {vals_str}',
                )
            )

        ax.legend(
            handles=legend_handles, loc='upper left', frameon=True, fontsize=9
        )

        if constant_params:
            const_info = "Fixed Parameters:\n" + "\n".join(
                [
                    f"• {clean_param_name(p)}: {self._clean_val(df[p].iloc[0])}"
                    for p in constant_params
                ]
            )
            ax.text(
                0.97,
                0.03,
                const_info,
                transform=ax.transAxes,
                fontsize=9,
                verticalalignment='bottom',
                horizontalalignment='right',
                bbox=dict(
                    boxstyle='round,pad=0.5',
                    facecolor='white',
                    alpha=0.8,
                    edgecolor='lightgray',
                ),
            )

        ax.set_xlabel('Index File Memory (GB)')
        ax.set_ylabel('TTI (s)')
        ax.set_title(
            f'Construction Parameter Tradeoff: {index_name} on {dataset_name}'
        )

        filename = f"{index_name}_{dataset_name}_{ds_subset_size}_{ds_query_param}_build.png"
        save_path = os.path.join(self.plot_dir, filename)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()

    def plot_query_parameters(
        self, index_name, dataset_name, ds_subset_size, ds_query_param=None
    ):
        """Plots (avg_recall, avg_latency) with multi-dimensional hyperparameter tree topology"""
        df = self.prepare_dataframe(
            index_name, dataset_name, ds_subset_size, ds_query_param
        )

        if df.empty:
            print(f"No data found for {index_name} on {dataset_name}.")
            return

        param_cols = [
            c
            for c in df.columns
            if c.startswith(('build_param_', 'query_param_'))
        ]

        if not param_cols:
            print(
                f"No parameter columns found for {index_name} on {dataset_name}."
            )
            return

        varying_params = [c for c in param_cols if df[c].nunique() > 1]
        constant_params = [c for c in param_cols if df[c].nunique() == 1]

        varying_params.sort(key=lambda c: df[c].nunique(), reverse=True)

        fig, ax = plt.subplots(figsize=(10, 6.5))

        param_colors = {
            p_col: self._get_param_color(clean_param_name(p_col))
            for p_col in varying_params
        }

        unique_vals = {
            p_col: [self._clean_val(v) for v in sorted(df[p_col].unique())]
            for p_col in varying_params
        }

        root_query = {p_col: unique_vals[p_col][0] for p_col in varying_params}

        root_mask = pd.Series(True, index=df.index)
        for p_col, val in root_query.items():
            root_mask &= df[p_col] == val

        root_df = df[root_mask]

        ax.scatter(
            df['avg_recall'], df['avg_latency'], color='black', zorder=5, s=35
        )

        if not root_df.empty:
            root_row = root_df.iloc[0]
            ax.scatter(
                root_row['avg_recall'],
                root_row['avg_latency'],
                color='gold',
                edgecolor='black',
                zorder=6,
                s=130,
            )

        def draw_tree_level(current_fixed_params, level_idx):
            if level_idx >= len(varying_params):
                return

            curr_param = varying_params[level_idx]
            vals = unique_vals[curr_param]
            color = param_colors[curr_param]

            for i in range(len(vals) - 1):
                val_curr, val_next = vals[i], vals[i + 1]

                start_mask = pd.Series(True, index=df.index)
                for p, v in current_fixed_params.items():
                    start_mask &= df[p] == v
                start_mask &= df[curr_param] == val_curr

                end_mask = pd.Series(True, index=df.index)
                for p, v in current_fixed_params.items():
                    end_mask &= df[p] == v
                end_mask &= df[curr_param] == val_next

                start_df = df[start_mask]
                end_df = df[end_mask]

                if not start_df.empty and not end_df.empty:
                    s_row, e_row = start_df.iloc[0], end_df.iloc[0]
                    ax.annotate(
                        '',
                        xy=(e_row['avg_recall'], e_row['avg_latency']),
                        xytext=(s_row['avg_recall'], s_row['avg_latency']),
                        arrowprops=dict(
                            arrowstyle="->", color=color, lw=1.8, alpha=0.85
                        ),
                    )

            if level_idx + 1 < len(varying_params):
                for v in vals:
                    next_fixed = current_fixed_params.copy()
                    next_fixed[curr_param] = v
                    draw_tree_level(next_fixed, level_idx + 1)

        if varying_params:
            draw_tree_level({}, 0)

        legend_handles = []

        root_str = ", ".join(
            [
                f"{clean_param_name(p)}={unique_vals[p][0]}"
                for p in varying_params
            ]
        )
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color='gold',
                marker='o',
                markeredgecolor='black',
                linestyle='None',
                markersize=9,
                label=f'Root ({root_str})',
            )
        )

        for p_col in varying_params:
            p_name = clean_param_name(p_col)
            vals_str = str(unique_vals[p_col])
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    color=param_colors[p_col],
                    lw=2,
                    label=f'Increment {p_name}: {vals_str}',
                )
            )

        ax.legend(
            handles=legend_handles, loc='upper left', frameon=True, fontsize=9
        )

        if constant_params:
            const_info = "Fixed Parameters:\n" + "\n".join(
                [
                    f"• {clean_param_name(p)}: {self._clean_val(df[p].iloc[0])}"
                    for p in constant_params
                ]
            )
            ax.text(
                0.97,
                0.03,
                const_info,
                transform=ax.transAxes,
                fontsize=9,
                verticalalignment='bottom',
                horizontalalignment='right',
                bbox=dict(
                    boxstyle='round,pad=0.5',
                    facecolor='white',
                    alpha=0.8,
                    edgecolor='lightgray',
                ),
            )

        ax.set_xlabel('Average Recall')
        ax.set_ylabel('Average Latency (s)')
        ax.set_title(
            f'Hyperparameter Tree Tradeoff: {index_name} on {dataset_name}'
        )

        filename = f"{index_name}_{dataset_name}_{ds_subset_size}_{ds_query_param}_query.png"
        save_path = os.path.join(self.plot_dir, filename)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()

    def plot_query_parameters_DAG(
        self, index_name, dataset_name, ds_subset_size, ds_query_param=None
    ):
        """Plots (avg_recall, avg_latency) with multi-dimensional hyperparameter DAG topology"""
        df = self.prepare_dataframe(
            index_name, dataset_name, ds_subset_size, ds_query_param
        )

        if df.empty:
            print(f"No data found for {index_name} on {dataset_name}.")
            return

        param_cols = [
            c
            for c in df.columns
            if c.startswith(('build_param_', 'query_param_'))
        ]

        if not param_cols:
            print(
                f"No parameter columns found for {index_name} on {dataset_name}."
            )
            return

        varying_params = [c for c in param_cols if df[c].nunique() > 1]
        constant_params = [c for c in param_cols if df[c].nunique() == 1]

        varying_params.sort(key=lambda c: df[c].nunique(), reverse=True)

        fig, ax = plt.subplots(figsize=(10, 6.5))

        param_colors = {
            p_col: self._get_param_color(clean_param_name(p_col))
            for p_col in varying_params
        }

        unique_vals = {
            p_col: [self._clean_val(v) for v in sorted(df[p_col].unique())]
            for p_col in varying_params
        }

        val_to_index = {
            p_col: {val: idx for idx, val in enumerate(vals)}
            for p_col, vals in unique_vals.items()
        }

        root_query = {p_col: unique_vals[p_col][0] for p_col in varying_params}

        root_mask = pd.Series(True, index=df.index)
        for p_col, val in root_query.items():
            root_mask &= df[p_col] == val

        root_df = df[root_mask]

        ax.scatter(
            df['avg_recall'], df['avg_latency'], color='black', zorder=5, s=35
        )

        if not root_df.empty:
            root_row = root_df.iloc[0]
            ax.scatter(
                root_row['avg_recall'],
                root_row['avg_latency'],
                color='gold',
                edgecolor='black',
                zorder=6,
                s=130,
            )

        for curr_param in varying_params:
            color = param_colors[curr_param]
            other_params = [p for p in varying_params if p != curr_param]

            if other_params:
                grouped = df.groupby(other_params, dropna=False)
            else:
                grouped = [('all', df)]

            for _, group in grouped:
                if len(group) < 2:
                    continue

                sorted_group = group.sort_values(by=curr_param)

                for i in range(len(sorted_group) - 1):
                    start_row = sorted_group.iloc[i]
                    end_row = sorted_group.iloc[i + 1]

                    start_val = self._clean_val(start_row[curr_param])
                    end_val = self._clean_val(end_row[curr_param])

                    start_idx = val_to_index[curr_param][start_val]
                    end_idx = val_to_index[curr_param][end_val]

                    if end_idx == start_idx + 1:
                        ax.annotate(
                            '',
                            xy=(end_row['avg_recall'], end_row['avg_latency']),
                            xytext=(
                                start_row['avg_recall'],
                                start_row['avg_latency'],
                            ),
                            arrowprops=dict(
                                arrowstyle="->", color=color, lw=1.8, alpha=0.85
                            ),
                        )

        legend_handles = []

        root_str = ", ".join(
            [
                f"{clean_param_name(p)}={unique_vals[p][0]}"
                for p in varying_params
            ]
        )
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color='gold',
                marker='o',
                markeredgecolor='black',
                linestyle='None',
                markersize=9,
                label=f'Root ({root_str})',
            )
        )

        for p_col in varying_params:
            p_name = clean_param_name(p_col)
            vals_str = str(unique_vals[p_col])
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    color=param_colors[p_col],
                    lw=2,
                    label=f'Increment {p_name}: {vals_str}',
                )
            )

        ax.legend(
            handles=legend_handles, loc='upper left', frameon=True, fontsize=9
        )

        if constant_params:
            const_info = "Fixed Parameters:\n" + "\n".join(
                [
                    f"• {clean_param_name(p)}: {self._clean_val(df[p].iloc[0])}"
                    for p in constant_params
                ]
            )
            ax.text(
                0.97,
                0.03,
                const_info,
                transform=ax.transAxes,
                fontsize=9,
                verticalalignment='bottom',
                horizontalalignment='right',
                bbox=dict(
                    boxstyle='round,pad=0.5',
                    facecolor='white',
                    alpha=0.8,
                    edgecolor='lightgray',
                ),
            )

        ax.set_xlabel('Average Recall')
        ax.set_ylabel('Average Latency (s)')
        ax.set_title(
            f'Hyperparameter DAG Tradeoff: {index_name} on {dataset_name}'
        )

        filename = f"{index_name}_{dataset_name}_{ds_subset_size}_{ds_query_param}_query_DAG.png"
        save_path = os.path.join(self.plot_dir, filename)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()


def plot_helper(
    param_analyser, index, dataset, subset_size=1.0, query_param=None
):
    param_analyser.plot_query_parameters(index, dataset, subset_size, query_param)
    param_analyser.plot_build_parameters(index, dataset, subset_size, query_param)
    param_analyser.plot_query_param_metric_correlation(
        index, dataset, subset_size, query_param
    )
    param_analyser.plot_build_param_metric_correlation(
        index, dataset, subset_size, query_param
    )


if __name__ == '__main__':
    acorn_analyser = ParameterAnalyser()
    ivf_analyser = ParameterAnalyser()
    ivf2_analyser = ParameterAnalyser()
    hnsw_analyser = ParameterAnalyser()

    for nr in range(1, 4):
        plot_helper(acorn_analyser, 'AcornFlat', 'SIFT', 1.0, nr)
    plot_helper(acorn_analyser, 'AcornFlat', 'GLOVE')
    plot_helper(acorn_analyser, 'AcornFlat', 'YFCC', 0.1)
    plot_helper(acorn_analyser, 'AcornFlat', 'GIST')

    for nr in range(1, 4):
        plot_helper(hnsw_analyser, 'HNSWPostfilter', 'SIFT', 1.0, nr)
    plot_helper(hnsw_analyser, 'HNSWPostfilter', 'GLOVE')
    plot_helper(hnsw_analyser, 'HNSWPostfilter', 'YFCC', 0.1)
    plot_helper(hnsw_analyser, 'HNSWPostfilter', 'GIST')

    for nr in range(1, 4):
        plot_helper(ivf_analyser, 'IVFIdFilter', 'SIFT', 1.0, nr)
    plot_helper(ivf_analyser, 'IVFIdFilter', 'GLOVE')
    plot_helper(ivf_analyser, 'IVFIdFilter', 'YFCC', 0.1)
    plot_helper(ivf_analyser, 'IVFIdFilter', 'GIST')

    plot_helper(ivf2_analyser, 'IVFSquaredFaiss', 'YFCC', 0.1)
    plot_helper(ivf2_analyser, 'IVFSquaredFaiss', 'GIST')