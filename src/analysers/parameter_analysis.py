import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from src.analysis.utils import load_run_data
from matplotlib.lines import Line2D

def clean_param_name(p):
    return p.replace('build_param_', '').replace('query_param_', '')

class ParameterAnalyser:
    def __init__(self, log_root_dir="logs", output_dir="analysis"):
        self.log_root = log_root_dir
        self.plot_dir = os.path.join(output_dir, 'hyperparameters', 'plots')
        os.makedirs(self.plot_dir, exist_ok=True)
        self.registry_path = os.path.join(self.log_root, "master_registry.csv")

        # Persistent palette mapping across method calls
        self.palette = plt.cm.get_cmap('Dark2').colors
        self.PARAM_COLOR_MAP = {}
        self._color_idx = 0

    def _get_param_color(self, param_name):
        """Returns a consistent, distinctive color for a given parameter name across all runs."""
        if param_name not in self.PARAM_COLOR_MAP:
            self.PARAM_COLOR_MAP[param_name] = self.palette[self._color_idx % len(self.palette)]
            self._color_idx += 1
        return self.PARAM_COLOR_MAP[param_name]

    @staticmethod
    def _clean_val(val):
        """Unwraps numpy scalars/types into native python primitives for clean printing."""
        if hasattr(val, 'item'):
            val = val.item()
        return val

    def prepare_dataframe(self, index_name, dataset_name, ds_subset_size, ds_query_param):
        registry = pd.read_csv(self.registry_path)
        
        mask = (registry['dataset'] == dataset_name) & \
               (registry['index'] == index_name) & \
               (registry['subset_size'] == ds_subset_size)
    
        if ds_query_param is not None:
            mask &= (registry['ds_query_param'] == ds_query_param)
        
        filtered = registry[mask]

        run_details = []
        for run_id in filtered['run_id']:
            metadata_path = os.path.join(self.log_root, run_id, 'metadata.json')
            details = load_run_data(metadata_path)
            if details:
                details['memory_gb'] = details['index_memory'] / (1024**3)
                
                # Flatten nested parameter dicts into top-level columns
                # Flatten nested parameter dicts into top-level columns with distinct prefixes
                if 'build_params' in details and isinstance(details['build_params'], dict):
                    for k, v in details['build_params'].items():
                        details[f'build_param_{k}'] = v
                        
                if 'query_params' in details and isinstance(details['query_params'], dict):
                    for k, v in details['query_params'].items():
                        details[f'query_param_{k}'] = v
                        
                run_details.append(details)
        
        return pd.DataFrame(run_details)

    def plot_build_parameters(self, index_name, dataset_name, ds_subset_size, ds_query_param=None):
        """Plots (memory_gb, build_time) across build hyperparameter variations."""
        df = self.prepare_dataframe(index_name, dataset_name, ds_subset_size, ds_query_param)

        if df.empty:
            print(f"No data found for {index_name} on {dataset_name}.")
            return

        param_cols = [c for c in df.columns if c.startswith('build_param_')]
        
        if not param_cols:
            print(f"No build parameters found for {index_name} on {dataset_name}.")
            return

        # Aggregate runs across query params since query params don't affect build metrics
        agg_dict = {'memory_gb': 'mean', 'build_time': 'mean'}
        df = df.groupby(param_cols, as_index=False).agg(agg_dict)

        varying_params = [c for c in param_cols if df[c].nunique() > 1]
        constant_params = [c for c in param_cols if df[c].nunique() == 1]

        varying_params.sort(key=lambda c: df[c].nunique(), reverse=True)

        fig, ax = plt.subplots(figsize=(10, 6.5))

        # Assign persistent colors per build parameter
        param_colors = {p_col: self._get_param_color(clean_param_name(p_col)) for p_col in varying_params}

        # Extract values dictionary for building the tree and clean numpy scalar types
        unique_vals = {
            p_col: [self._clean_val(v) for v in sorted(df[p_col].unique())] 
            for p_col in varying_params
        }

        # Determine Root Configuration (minimum value of every varying build parameter)
        root_query = {p_col: unique_vals[p_col][0] for p_col in varying_params}
        
        root_mask = pd.Series(True, index=df.index)
        for p_col, val in root_query.items():
            root_mask &= (df[p_col] == val)

        root_df = df[root_mask]

        # 2. Draw Data Points
        ax.scatter(df['memory_gb'], df['build_time'], color='black', zorder=5, s=35)

        # Highlight Root Node
        if not root_df.empty:
            root_row = root_df.iloc[0]
            ax.scatter(root_row['memory_gb'], root_row['build_time'], 
                       color='gold', edgecolor='black', zorder=6, s=130)

        # 3. Build Multi-Dimensional Spanning Tree Edges
        def draw_tree_level(current_fixed_params, level_idx):
            if level_idx >= len(varying_params):
                return

            curr_param = varying_params[level_idx]
            vals = unique_vals[curr_param]
            color = param_colors[curr_param]

            for i in range(len(vals) - 1):
                val_curr, val_next = vals[i], vals[i+1]

                start_mask = pd.Series(True, index=df.index)
                for p, v in current_fixed_params.items():
                    start_mask &= (df[p] == v)
                start_mask &= (df[curr_param] == val_curr)

                end_mask = pd.Series(True, index=df.index)
                for p, v in current_fixed_params.items():
                    end_mask &= (df[p] == v)
                end_mask &= (df[curr_param] == val_next)

                start_df = df[start_mask]
                end_df = df[end_mask]

                if not start_df.empty and not end_df.empty:
                    s_row, e_row = start_df.iloc[0], end_df.iloc[0]
                    ax.annotate(
                        '', xy=(e_row['memory_gb'], e_row['build_time']),
                        xytext=(s_row['memory_gb'], s_row['build_time']),
                        arrowprops=dict(arrowstyle="->", color=color, lw=1.8, alpha=0.85)
                    )

            if level_idx + 1 < len(varying_params):
                for v in vals:
                    next_fixed = current_fixed_params.copy()
                    next_fixed[curr_param] = v
                    draw_tree_level(next_fixed, level_idx + 1)

        if varying_params:
            draw_tree_level({}, 0)

        # 4. Legend
        legend_handles = []
        
        root_str = ", ".join([f"{clean_param_name(p)}={unique_vals[p][0]}" for p in varying_params])
        legend_handles.append(
            Line2D([0], [0], color='gold', marker='o', markeredgecolor='black', 
                   linestyle='None', markersize=9, label=f'Root ({root_str})')
        )

        for p_col in varying_params:
            p_name = p_col.replace('build_param_', '')
            vals_str = str(unique_vals[p_col])
            legend_handles.append(
                Line2D([0], [0], color=param_colors[p_col], lw=2, label=f'Increment {p_name}: {vals_str}')
            )

        ax.legend(handles=legend_handles, loc='upper left', frameon=True, fontsize=9)

        # 5. Fixed Build Parameters Box
        if constant_params:
            const_info = "Fixed Parameters:\n" + "\n".join(
                [f"• {clean_param_name(p)}: {self._clean_val(df[p].iloc[0])}" for p in constant_params]
            )
            ax.text(
                0.97, 0.03, const_info,
                transform=ax.transAxes,
                fontsize=9,
                verticalalignment='bottom',
                horizontalalignment='right',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='lightgray')
            )

        # 6. Formatting & Saving
        ax.set_xlabel('Index Memory (GB)')
        ax.set_ylabel('Build Time (s)')
        ax.set_title(f'Build Parameter Tradeoff: {index_name} on {dataset_name}')

        filename = f"{index_name}_{dataset_name}_{ds_subset_size}_{ds_query_param}_build.png"
        save_path = os.path.join(self.plot_dir, filename)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()

    def plot_query_parameter(self, index_name, dataset_name, ds_subset_size, ds_query_param=None):
        """Plots (avg_recall, avg_latency) with multi-dimensional hyperparameter tree topology"""
        df = self.prepare_dataframe(index_name, dataset_name, ds_subset_size, ds_query_param)

        if df.empty:
            print(f"No data found for {index_name} on {dataset_name}.")
            return

        # 1. Separate varying parameters from constant parameters
        param_cols = [c for c in df.columns if c.startswith(('build_param_', 'query_param_'))]
        
        if not param_cols:
            print(f"No parameter columns found for {index_name} on {dataset_name}.")
            return

        varying_params = [c for c in param_cols if df[c].nunique() > 1]
        constant_params = [c for c in param_cols if df[c].nunique() == 1]

        # Sort varying params by grid breadth
        varying_params.sort(key=lambda c: df[c].nunique(), reverse=True)

        fig, ax = plt.subplots(figsize=(10, 6.5))

        # Assign consistent colors per parameter
        param_colors = {p_col: self._get_param_color(clean_param_name(p_col)) for p_col in varying_params}

        # Extract values dictionary for building the tree and clean numpy scalar types
        unique_vals = {
            p_col: [self._clean_val(v) for v in sorted(df[p_col].unique())] 
            for p_col in varying_params
        }

        # Determine Root Configuration (minimum value of every varying parameter)
        root_query = {p_col: unique_vals[p_col][0] for p_col in varying_params}
        
        # Build Pandas query string for root
        root_mask = pd.Series(True, index=df.index)
        for p_col, val in root_query.items():
            root_mask &= (df[p_col] == val)

        root_df = df[root_mask]

        # 2. Draw Points
        # All data points in black (no text annotations)
        ax.scatter(df['avg_recall'], df['avg_latency'], color='black', zorder=5, s=35)

        # Highlight Root Node distinctly
        if not root_df.empty:
            root_row = root_df.iloc[0]
            ax.scatter(root_row['avg_recall'], root_row['avg_latency'], 
                       color='gold', edgecolor='black', zorder=6, s=130)

        # 3. Build Multi-Dimensional Spanning Tree Edges
        def draw_tree_level(current_fixed_params, level_idx):
            if level_idx >= len(varying_params):
                return

            curr_param = varying_params[level_idx]
            vals = unique_vals[curr_param]
            color = param_colors[curr_param]

            # Trace along the current parameter
            for i in range(len(vals) - 1):
                val_curr, val_next = vals[i], vals[i+1]

                # Match start configuration
                start_mask = pd.Series(True, index=df.index)
                for p, v in current_fixed_params.items():
                    start_mask &= (df[p] == v)
                start_mask &= (df[curr_param] == val_curr)

                # Match end configuration
                end_mask = pd.Series(True, index=df.index)
                for p, v in current_fixed_params.items():
                    end_mask &= (df[p] == v)
                end_mask &= (df[curr_param] == val_next)

                start_df = df[start_mask]
                end_df = df[end_mask]

                if not start_df.empty and not end_df.empty:
                    s_row, e_row = start_df.iloc[0], end_df.iloc[0]
                    ax.annotate(
                        '', xy=(e_row['avg_recall'], e_row['avg_latency']),
                        xytext=(s_row['avg_recall'], s_row['avg_latency']),
                        arrowprops=dict(arrowstyle="->", color=color, lw=1.8, alpha=0.85)
                    )

            # Recurse down to child sub-branches for the next parameters
            if level_idx + 1 < len(varying_params):
                for v in vals:
                    next_fixed = current_fixed_params.copy()
                    next_fixed[curr_param] = v
                    draw_tree_level(next_fixed, level_idx + 1)

        # Start tree generation from root level
        if varying_params:
            draw_tree_level({}, 0)

        # 4. Construct Legend for All Parameters
        legend_handles = []
        
        # Root node in legend
        root_str = ", ".join([f"{clean_param_name(p)}={unique_vals[p][0]}" for p in varying_params])
        legend_handles.append(
            Line2D([0], [0], color='gold', marker='o', markeredgecolor='black', 
                   linestyle='None', markersize=9, label=f'Root ({root_str})')
        )

        # Varying parameters lines in legend
        for p_col in varying_params:
            p_name = clean_param_name(p_col)
            vals_str = str(unique_vals[p_col])
            legend_handles.append(
                Line2D([0], [0], color=param_colors[p_col], lw=2, label=f'Increment {p_name}: {vals_str}')
            )

        ax.legend(handles=legend_handles, loc='upper left', frameon=True, fontsize=9)

        # 5. Add Box with Fixed/Constant Parameters
        if constant_params:
            const_info = "Fixed Parameters:\n" + "\n".join(
                [f"• {clean_param_name(p)}: {self._clean_val(df[p].iloc[0])}" for p in constant_params]
            )
            ax.text(
                0.97, 0.03, const_info,
                transform=ax.transAxes,
                fontsize=9,
                verticalalignment='bottom',
                horizontalalignment='right',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='lightgray')
            )

        # 6. Formatting & Saving (Using Linear Axes)
        ax.set_xlabel('Average Recall')
        ax.set_ylabel('Average Latency (s)')
        ax.set_title(f'Hyperparameter Tree Tradeoff: {index_name} on {dataset_name}')

        filename = f"{index_name}_{dataset_name}_{ds_subset_size}_{ds_query_param}_query.png"
        save_path = os.path.join(self.plot_dir, filename)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()

    def plot_query_parameter_DAG(self, index_name, dataset_name, ds_subset_size, ds_query_param=None):
        """Plots (avg_recall, avg_latency) with multi-dimensional hyperparameter DAG topology"""
        df = self.prepare_dataframe(index_name, dataset_name, ds_subset_size, ds_query_param)

        if df.empty:
            print(f"No data found for {index_name} on {dataset_name}.")
            return

        # 1. Separate varying parameters from constant parameters
        param_cols = [c for c in df.columns if c.startswith(('build_param_', 'query_param_'))]
        
        if not param_cols:
            print(f"No parameter columns found for {index_name} on {dataset_name}.")
            return

        varying_params = [c for c in param_cols if df[c].nunique() > 1]
        constant_params = [c for c in param_cols if df[c].nunique() == 1]

        # Sort varying params by grid breadth
        varying_params.sort(key=lambda c: df[c].nunique(), reverse=True)

        fig, ax = plt.subplots(figsize=(10, 6.5))

        # Assign consistent colors per parameter
        param_colors = {p_col: self._get_param_color(clean_param_name(p_col)) for p_col in varying_params}

        # Extract sorted unique values per parameter
        unique_vals = {
            p_col: [self._clean_val(v) for v in sorted(df[p_col].unique())] 
            for p_col in varying_params
        }

        # Map each value to its index position for fast sequential lookups
        val_to_index = {
            p_col: {val: idx for idx, val in enumerate(vals)}
            for p_col, vals in unique_vals.items()
        }

        # Determine Root Configuration (minimum value of every varying parameter)
        root_query = {p_col: unique_vals[p_col][0] for p_col in varying_params}
        
        root_mask = pd.Series(True, index=df.index)
        for p_col, val in root_query.items():
            root_mask &= (df[p_col] == val)

        root_df = df[root_mask]

        # 2. Draw Points
        ax.scatter(df['avg_recall'], df['avg_latency'], color='black', zorder=5, s=35)

        # Highlight Root Node distinctly
        if not root_df.empty:
            root_row = root_df.iloc[0]
            ax.scatter(root_row['avg_recall'], root_row['avg_latency'], 
                    color='gold', edgecolor='black', zorder=6, s=130)

        # 3. Construct DAG Edges
        # Iterate through each parameter to evaluate single-step transitions
        for curr_param in varying_params:
            color = param_colors[curr_param]
            other_params = [p for p in varying_params if p != curr_param]

            # Group rows by all parameters EXCEPT the target parameter
            if other_params:
                grouped = df.groupby(other_params, dropna=False)
            else:
                grouped = [('all', df)]

            for _, group in grouped:
                if len(group) < 2:
                    continue

                # Sort group by current parameter's sequential values
                sorted_group = group.sort_values(by=curr_param)

                # Draw directed arrows between consecutive parameter steps
                for i in range(len(sorted_group) - 1):
                    start_row = sorted_group.iloc[i]
                    end_row = sorted_group.iloc[i + 1]

                    start_val = self._clean_val(start_row[curr_param])
                    end_val = self._clean_val(end_row[curr_param])

                    start_idx = val_to_index[curr_param][start_val]
                    end_idx = val_to_index[curr_param][end_val]

                    # Draw edge only if the step is a direct single-step increment
                    if end_idx == start_idx + 1:
                        ax.annotate(
                            '', 
                            xy=(end_row['avg_recall'], end_row['avg_latency']),
                            xytext=(start_row['avg_recall'], start_row['avg_latency']),
                            arrowprops=dict(arrowstyle="->", color=color, lw=1.8, alpha=0.85)
                        )

        # 4. Construct Legend for All Parameters
        legend_handles = []
        
        # Root node in legend
        root_str = ", ".join([f"{clean_param_name(p)}={unique_vals[p][0]}" for p in varying_params])
        legend_handles.append(
            Line2D([0], [0], color='gold', marker='o', markeredgecolor='black', 
                linestyle='None', markersize=9, label=f'Root ({root_str})')
        )

        # Varying parameters lines in legend
        for p_col in varying_params:
            p_name = clean_param_name(p_col)
            vals_str = str(unique_vals[p_col])
            legend_handles.append(
                Line2D([0], [0], color=param_colors[p_col], lw=2, label=f'Increment {p_name}: {vals_str}')
            )

        ax.legend(handles=legend_handles, loc='upper left', frameon=True, fontsize=9)

        # 5. Add Box with Fixed/Constant Parameters
        if constant_params:
            const_info = "Fixed Parameters:\n" + "\n".join(
                [f"• {clean_param_name(p)}: {self._clean_val(df[p].iloc[0])}" for p in constant_params]
            )
            ax.text(
                0.97, 0.03, const_info,
                transform=ax.transAxes,
                fontsize=9,
                verticalalignment='bottom',
                horizontalalignment='right',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='lightgray')
            )

        # 6. Formatting & Saving
        ax.set_xlabel('Average Recall')
        ax.set_ylabel('Average Latency (s)')
        ax.set_title(f'Hyperparameter DAG Tradeoff: {index_name} on {dataset_name}')

        filename = f"{index_name}_{dataset_name}_{ds_subset_size}_{ds_query_param}_query_DAG.png"
        save_path = os.path.join(self.plot_dir, filename)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()

if __name__=='__main__':
    analyser = ParameterAnalyser()
    analyser.plot_query_parameter('Acorn', 'GIST', 1.0, None)
    analyser.plot_query_parameter_DAG('Acorn', 'GIST', 1.0, None)
    analyser.plot_build_parameters('Acorn', 'GIST', 1.0, None)
    
    analyser.plot_query_parameter('IVFSquaredFaiss', 'GIST', 1.0, None)
    analyser.plot_query_parameter_DAG('IVFSquaredFaiss', 'GIST', 1.0, None)
    analyser.plot_build_parameters('IVFSquaredFaiss', 'GIST', 1.0, None)

    analyser.plot_query_parameter('IVFSquaredFaiss', 'YFCC', 0.1, None)
    analyser.plot_query_parameter_DAG('IVFSquaredFaiss', 'YFCC', 0.1, None)
    analyser.plot_build_parameters('IVFSquaredFaiss', 'YFCC', 0.1, None)

    analyser.plot_query_parameter('HNSWPostfilter', 'GIST', 1.0, None)
    analyser.plot_query_parameter_DAG('HNSWPostfilter', 'GIST', 1.0, None)
    analyser.plot_build_parameters('HNSWPostfilter', 'GIST', 1.0, None)

    analyser.plot_query_parameter('IVFIdFilter', 'GIST', 1.0, None)
    analyser.plot_query_parameter_DAG('IVFIdFilter', 'GIST', 1.0, None)
    analyser.plot_build_parameters('IVFIdFilter', 'GIST', 1.0, None)

    
