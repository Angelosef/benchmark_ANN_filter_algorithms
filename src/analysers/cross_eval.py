import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
from src.analysers.utils import find_selectivity_path, load_run_data

class CrossEvaluator:
    def __init__(self, log_root_dir="logs", output_dir="analysis"):
        self.log_root = Path(log_root_dir)
        self.plot_dir = os.path.join(output_dir, 'cross_eval', 'plots')
        self.table_dir = os.path.join(output_dir, 'cross_eval', 'tables')
        os.makedirs(self.plot_dir, exist_ok=True)
        os.makedirs(self.table_dir, exist_ok=True)
        self.registry_path = self.log_root / "master_registry.csv"
        
        # Plotting aesthetics
        sns.set_theme(style="whitegrid")
        
        # Pre-build a high-contrast qualitative color pool (26+ distinct colors)
        self._palette_pool = (
            sns.color_palette("tab10") + 
            sns.color_palette("Set2") + 
            sns.color_palette("Dark2")
        )
        self.color_map = {}

    def _prepare_dataframe(self, dataset_name, subset_size, ds_query_param):
        """Joins registry with JSON details into a single DataFrame."""
        registry = pd.read_csv(self.registry_path)
        
        mask = (registry['dataset'] == dataset_name) & \
               (registry['subset_size'] == subset_size)
    
        if ds_query_param is not None:
            mask &= (registry['ds_query_param'] == ds_query_param)
        
        filtered = registry[mask]
        
        run_details = []
        for run_id in filtered['run_id']:
            metadata_path = os.path.join(self.log_root, run_id, 'metadata.json')
            details = load_run_data(metadata_path)
            if details:
                run_details.append(details)
        
        return pd.DataFrame(run_details)

    def _get_color_map(self, algorithms):
        """Assigns high-contrast, distinct colors to algorithms and preserves them across plots."""
        unique_algos = sorted(list(set(algorithms)))
        unseen = [algo for algo in unique_algos if algo not in self.color_map]
        
        for algo in unseen:
            idx = len(self.color_map)
            if idx < len(self._palette_pool):
                # Use distinct categorical hues from the qualitative palette pool
                self.color_map[algo] = self._palette_pool[idx]
            else:
                # Dynamically sample HUSL color space for 27+ algorithms (evenly spaced hues)
                self.color_map[algo] = sns.color_palette("husl", idx + 1)[-1]
                
        return {algo: self.color_map[algo] for algo in unique_algos}

    
    def _prepare_selectivity_df(self, dataset_name, subset_size, neighbors_retrieved, ds_query_param, num_buckets=4):
        """Joins registry with JSON details into a single DataFrame."""
        registry = pd.read_csv(self.registry_path)
        sel_path = find_selectivity_path(dataset_name, subset_size, neighbors_retrieved, ds_query_param)
        selectivities = np.load(sel_path)

        mask = (registry['dataset'] == dataset_name) & \
               (registry['subset_size'] == subset_size)
    
        if ds_query_param is not None:
            mask &= (registry['ds_query_param'] == ds_query_param)
        
        filtered = registry[mask]

        for run_id in filtered['run_id']:
            details = load_run_data(os.path.join(self.log_root, run_id, 'metadata.json'))
            if details is None:
                filtered = filtered[filtered['run_id'] != run_id]

        if filtered.empty:
            return filtered

        run_id = filtered['run_id'].iloc[0]
        details = load_run_data(os.path.join(self.log_root, run_id, 'metadata.json'))

        selectivities = selectivities / details['base_count']
        sel_bucket_points = np.geomspace(np.min(selectivities), np.max(selectivities), num_buckets+1)
        
        run_details = []
        for run_id in filtered['run_id']:
            for i in range(len(sel_bucket_points)-1):
                min_sel = sel_bucket_points[i]
                max_sel = sel_bucket_points[i+1]
                query_indexes = np.where((selectivities >= min_sel) & (selectivities <= max_sel))[0]
                
                recalls = np.load(os.path.join(self.log_root, run_id, "recalls.npy"))
                recalls = recalls[query_indexes]

                latencies = np.load(os.path.join(self.log_root, run_id, "latencies.npy"))
                latencies = latencies[query_indexes]
                
                details = load_run_data(os.path.join(self.log_root, run_id, 'metadata.json'))
                details["min_selectivity"] = min_sel
                details["max_selectivity"] = max_sel

                details["avg_recall"] = np.mean(recalls)
                details["std_recall"] = np.std(recalls)
                details["p2_recall"] = np.percentile(recalls, 2)
                details["p5_recall"] = np.percentile(recalls, 5)
                details["p25_recall"] = np.percentile(recalls, 25)
                details["p50_recall"] = np.percentile(recalls, 50)
                details["p75_recall"] = np.percentile(recalls, 75)
                details["p95_recall"] = np.percentile(recalls, 95)
                details["p98_recall"] = np.percentile(recalls, 98)

                details["avg_latency"] = np.mean(latencies)
                details["p2_latency"] = np.percentile(latencies, 2)
                details["p5_latency"] = np.percentile(latencies, 5)
                details["p25_latency"] = np.percentile(latencies, 25)
                details["p50_latency"] = np.percentile(latencies, 50)
                details["p75_latency"] = np.percentile(latencies, 75)
                details["p95_latency"] = np.percentile(latencies, 95)
                details["p98_latency"] = np.percentile(latencies, 98)

                run_details.append(details)
        
        return pd.DataFrame(run_details)

    @staticmethod
    def extract_pareto_frontier(recalls, latencies):
        """Extracts strictly non-dominated points (higher recall, lower latency) 
        and returns them sorted by recall."""
        sorted_indices = np.argsort(recalls)
        r_sorted = np.asarray(recalls)[sorted_indices]
        l_sorted = np.asarray(latencies)[sorted_indices]

        pareto_r, pareto_l = [], []
        min_latency = float('inf')

        # Traverse backwards from highest recall
        for r, l in zip(reversed(r_sorted), reversed(l_sorted)):
            if l < min_latency:
                pareto_r.append(r)
                pareto_l.append(l)
                min_latency = l

        return np.array(list(reversed(pareto_r))), np.array(list(reversed(pareto_l)))

    def interpolate_latency(self, pareto_r, pareto_l, target_recalls):
        """Linearly interpolates latency at target recall levels.
        
        Behavior for out-of-bounds target values:
        - If target_recall < min(pareto_r): snaps to pareto_l[0] (lowest recall's 
          latency) to play it safe and avoid NaNs.
        - If target_recall > max(pareto_r): returns NaN (cannot achieve target recall).
        """
        if len(pareto_r) == 0:
            return np.full_like(target_recalls, np.nan, dtype=np.float64)

        target_recalls = np.asarray(target_recalls, dtype=np.float64)

        # Standard linear interpolation for values within pareto_r bounds
        # 'left' sets values < min(pareto_r), 'right' sets values > max(pareto_r)
        interp_latencies = np.interp(
            target_recalls, 
            pareto_r, 
            pareto_l, 
            left=pareto_l[0],   # Snap to latency of lowest achieved recall (v2, y2)
            right=np.nan        # Keep NaN if target recall exceeds max achieved recall
        )

        return interp_latencies

    def generate_speed_up_table_selectivity(self, ds_name, ds_subset_size, ds_query_param, baseline_index, target_recalls, neighbors_retrieved=10):
        save_folder = os.path.join(self.table_dir, f'speed_up_{ds_name}_{ds_subset_size}_{ds_query_param}_{baseline_index}')
        os.makedirs(save_folder, exist_ok=True)

        full_df = self._prepare_selectivity_df(ds_name, ds_subset_size, neighbors_retrieved, ds_query_param)
        for (min_sel, max_sel), group in full_df.groupby(["min_selectivity", "max_selectivity"]):    
            filename = f'{min_sel:.4f}_{max_sel:.4f},csv'
            save_path = os.path.join(save_folder, filename)
            self.save_speedup_table(group, baseline_index, target_recalls, save_path, index_col='index_name', recall_col='avg_recall', latency_col='avg_latency')
        
    def generate_speedup_table(self, ds_name, ds_subset_size, ds_query_param, baseline_index, target_recalls):
        dataframe = self._prepare_dataframe(ds_name, ds_subset_size, ds_query_param)
        filename = f'speed_up_{ds_name}_{ds_subset_size}_{ds_query_param}_{baseline_index}.csv'
        save_path = os.path.join(self.table_dir, filename)
        speedup_table = self.save_speedup_table(dataframe, baseline_index, target_recalls, save_path, index_col='index_name', recall_col='avg_recall', latency_col='avg_latency')
        return speedup_table

    def save_speedup_table(
        self,
        df,
        baseline_index,
        target_recalls,
        save_path,
        index_col='index_name',
        recall_col='avg_recall',
        latency_col='avg_latency',
    ):
        if baseline_index not in df[index_col].unique():
            raise ValueError(
                f"Baseline index '{baseline_index}' not found in column"
                f" '{index_col}'."
            )

        # 1. Extract Pareto Frontiers for each index group from the DataFrame
        pareto_frontiers = {}
        for name, group in df.groupby(index_col):
            recalls = group[recall_col].to_numpy()
            latencies = group[latency_col].to_numpy()
            pr, pl = self.extract_pareto_frontier(recalls, latencies)
            pareto_frontiers[name] = (pr, pl)

        # 2. Get baseline latencies
        base_r, base_l = pareto_frontiers[baseline_index]
        base_interp_l = self.interpolate_latency(base_r, base_l, target_recalls)

        rows = []
        for name, (pr, pl) in pareto_frontiers.items():
            interp_l = self.interpolate_latency(pr, pl, target_recalls)

            row = {'Index': name}
            for i, r_target in enumerate(target_recalls):
                l_cand = interp_l[i]
                l_base = base_interp_l[i]

                if np.isnan(l_cand) or np.isnan(l_base):
                    row[f'R={r_target} (s)'] = 'N/A'
                    row[f'Speedup @ {r_target}'] = 'N/A'
                else:
                    speedup = l_base / l_cand
                    row[f'R={r_target} (s)'] = f'{l_cand:.5f}'
                    row[f'Speedup @ {r_target}'] = f'{speedup:.2f}x'

            rows.append(row)

        df_summary = pd.DataFrame(rows)

        # Export CSV & LaTeX
        csv_path = os.path.join(save_path)

        df_summary.to_csv(csv_path, index=False)

        return df_summary
    
    def plot_selectivity_based_performance(self, dataset_name, subset_size, neighbors_retrieved, ds_query_param):
        df = self._prepare_selectivity_df(dataset_name, subset_size, neighbors_retrieved, ds_query_param)

        if df.empty:
            print(f"No data found for {dataset_name} with subset {subset_size} and param {ds_query_param}.")
            return

        p_suffix = f"_p{ds_query_param}" if ds_query_param is not None else ""

        for (min_sel, max_sel), group in df.groupby(["min_selectivity", "max_selectivity"]):
            self.plot_recall_vs_latency(group, dataset_name, subset_size, p_suffix, min_sel, max_sel)
        
        print(f"Done! Plots saved in: {self.plot_dir}")

    def plot_recall_vs_latency(self, df, dataset_name, subset_size, ds_query_param, min_sel=0.0, max_sel=1.0):
        """Recall vs latency Pareto frontier with percentile crosses."""
        plt.figure(figsize=(10, 6))

        df = df.sort_values(by=["index_name", "p50_latency"])

        pareto_frames = []
        for algo in df["index_name"].unique():
            algo_data = df[df["index_name"] == algo]
            max_recall_so_far = -1.0
            frontier_indices = []

            for idx, row in algo_data.iterrows():
                if row["avg_recall"] > max_recall_so_far:
                    frontier_indices.append(idx)
                    max_recall_so_far = row["avg_recall"]

            pareto_frames.append(algo_data.loc[frontier_indices])

        frontier_df = pd.concat(pareto_frames)
        colors = self._get_color_map(frontier_df["index_name"])

        ax = sns.lineplot(
            data=frontier_df,
            x="avg_recall",
            y="p50_latency",
            hue="index_name",
            palette=colors,
            marker="o",
            sort=True,
        )

        # Draw percentile crosses using direct lookup from self.color_map
        for _, row in frontier_df.iterrows():
            color = colors[row["index_name"]]

            x = row["avg_recall"]
            y = row["p50_latency"]

            plt.hlines(
                y=y,
                xmin=x - row["std_recall"],
                xmax=x + row["std_recall"],
                color=color,
                linewidth=1,
                alpha=0.5,
                zorder=2,
            )

            plt.vlines(
                x=x,
                ymin=row["p25_latency"],
                ymax=row["p75_latency"],
                color=color,
                linewidth=1,
                alpha=0.5,
                zorder=2,
            )

        plt.yscale("log")
        title_sel_part = 'all queries'
        if min_sel != 0.0 or max_sel != 1.0:
            title_sel_part = f'selectivity range: ({min_sel:.4f}, {max_sel:.4f})'
            
        plt.title(
            f"Recall vs Query Latency "
            f"({dataset_name} -  {ds_query_param}) - {title_sel_part}"
        )

        plt.xlabel("Recall")
        plt.ylabel("Latency Per Query (s)")
        plt.grid(True, which="both", alpha=0.4)

        fname = os.path.join(
            self.plot_dir,
            f"{dataset_name}_{subset_size}_{ds_query_param}_{min_sel:.4f}_{max_sel:.4f}_recall_latency.png"
        )

        plt.savefig(fname, bbox_inches="tight")
        plt.close()
    
    def plot_recall_vs_qps(self, df, dataset_name, subset_size, ds_query_param):
        """Creates the Recall/qps trade-off curve with lightweight percentile intervals."""
        plt.figure(figsize=(10, 6))
        sns.set_theme(style="whitegrid")
        
        df['qps'] = df['query_count'] / df['total_query_time']
        df = df.sort_values(by=['index_name', 'qps'], ascending=False)

        pareto_frames = []
        for algo in df['index_name'].unique():
            algo_data = df[df['index_name'] == algo]
            max_avg_recall_so_far = -1.0
            frontier_indices = []
            
            for i, row in algo_data.iterrows():
                if row['avg_recall'] > max_avg_recall_so_far:
                    frontier_indices.append(i)
                    max_avg_recall_so_far = row['avg_recall']
            
            pareto_frames.append(algo_data.loc[frontier_indices])

        frontier_df = pd.concat(pareto_frames)
        colors = self._get_color_map(frontier_df['index_name'])

        ax = sns.lineplot(
            data=frontier_df, 
            x='avg_recall',
            y='qps', 
            hue='index_name', 
            palette=colors,
            marker='o',
            sort=True
        )
        
        plt.yscale('log')
        plt.title(f"Recall vs qps ({dataset_name} - {subset_size} - {ds_query_param})", fontsize=14, pad=15)
        plt.xlabel("Recall", fontsize=12)
        plt.ylabel("Queries per Second", fontsize=12)
        plt.grid(True, which="both", ls="-", alpha=0.3)
        plt.xlim(-0.02, 1.02)
        
        os.makedirs(self.plot_dir, exist_ok=True)
        fname = os.path.join(self.plot_dir, f"{dataset_name}_{subset_size}_{ds_query_param}_recall_qps.png")
        plt.savefig(fname, bbox_inches='tight', dpi=300)
        plt.close()

    def plot_memory_usage(self, df, dataset_name, subset_size):
        """
        Creates a grouped bar plot comparing index_memory, build_memory_peak, 
        and index_file_size_bytes (converted to GB) across different index_names.
        """
        # 1. Filter DataFrame for the target dataset and subset size (if needed)
        filtered_df = df[
            (df['dataset_name'] == dataset_name) & 
            (df['subset_size'] == subset_size)
        ].copy()

        if filtered_df.empty:
            print(f"No data available for {dataset_name} ({subset_size})")
            return

        BYTES_TO_GB = 1024 ** 3

        # 2. Convert memory metrics from bytes to GB
        filtered_df['Index Memory (GB)'] = filtered_df['index_memory'] / BYTES_TO_GB
        filtered_df['Peak Build Memory (GB)'] = filtered_df['build_memory_peak'] / BYTES_TO_GB
        
        # Handle index_file_size column name fallback (e.g., index_file_size vs index_file_size_bytes)
        file_size_col = 'index_file_size_bytes' if 'index_file_size_bytes' in filtered_df.columns else 'index_file_size'
        filtered_df['Index File Size (GB)'] = filtered_df[file_size_col] / BYTES_TO_GB

        # 3. Reshape DataFrame to long format for Seaborn grouped bar plotting
        metric_cols = ['Index Memory (GB)', 'Peak Build Memory (GB)', 'Index File Size (GB)']
        
        melted_df = filtered_df.melt(
            id_vars=['index_name'],
            value_vars=metric_cols,
            var_name='Memory Metric',
            value_name='Gigabytes (GB)'
        )

        # 4. Create Plot
        plt.figure(figsize=(10, 6))
        
        colors = self._get_color_map(df['index_name'])

        ax = sns.barplot(
            data=melted_df,
            x='Memory Metric',
            y='Gigabytes (GB)',
            hue='index_name',
            palette=colors,
            errorbar='sd'
        )

        # Formatting
        plt.title(f"Memory Usage Overview - {dataset_name} ({subset_size})", fontsize=13, fontweight='bold')
        plt.ylabel("Memory (GB)", fontsize=11)
        plt.xlabel("")
        plt.grid(True, axis='y', linestyle='--', alpha=0.5)
        plt.legend(title="Index Type", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()

        # 5. Save figure
        os.makedirs(self.plot_dir, exist_ok=True)
        filename = f"{dataset_name}_{subset_size}_memory.png"
        save_path = os.path.join(self.plot_dir, filename)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Plot saved to: {save_path}")
        return save_path

    def plot_tti_memory(self, df, dataset_name, subset_size):
        """
        Plots Build Time vs. Index Memory in a 2D scatter plot with points colored by index_name.
        """
        # 1. Filter DataFrame for target dataset and subset size
        filtered_df = df[
            (df['dataset_name'] == dataset_name) & 
            (df['subset_size'] == subset_size)
        ].copy()

        if filtered_df.empty:
            print(f"No data available for {dataset_name} ({subset_size})")
            return

        BYTES_TO_GB = 1024 ** 3

        # 2. Convert Index Memory to GB
        filtered_df['index_memory_gb'] = filtered_df['index_memory'] / BYTES_TO_GB

        # 3. Create Plot
        plt.figure(figsize=(9, 6))

        colors = self._get_color_map(df['index_name'])

        # Plot 2D scatter points
        ax = sns.scatterplot(
            data=filtered_df,
            x='build_time',
            y='index_memory_gb',
            hue='index_name',
            style='index_name',  # Uses distinct marker shapes per index_name
            palette=colors,
            s=120,               # Marker size
            alpha=0.85
        )

        # Formatting
        plt.title(f"Build Time vs. Index Memory - {dataset_name} ({subset_size})", fontsize=13, fontweight='bold')
        plt.xlabel("Build Time (Seconds)", fontsize=11)
        plt.ylabel("Index Memory (GB)", fontsize=11)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend(title="Index Type", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()

        # 4. Save figure
        os.makedirs(self.plot_dir, exist_ok=True)
        filename = f"{dataset_name}_{subset_size}_tti_memory.png"
        save_path = os.path.join(self.plot_dir, filename)

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Plot saved to: {save_path}")
        return save_path
        

    def plot(self, dataset_name, subset_size, ds_query_param=None):
        """Main interface to generate all plots for a specific benchmark."""
        df = self._prepare_dataframe(dataset_name, subset_size, ds_query_param)
        
        if df.empty:
            print(f"No data found for {dataset_name} with subset {subset_size} and param {ds_query_param}.")
            return

        print(f"Generating plots for {dataset_name}...")
        p_suffix = f"_p{ds_query_param}" if ds_query_param is not None else ""
        self.plot_recall_vs_latency(df, dataset_name, subset_size, p_suffix)
        self.plot_recall_vs_qps(df, dataset_name, subset_size, p_suffix)
        self.plot_tti_memory(df, dataset_name, subset_size)
        self.plot_memory_usage(df, dataset_name, subset_size)
        print(f"Done! Plots saved in: {self.plot_dir}")
        return

def generate_cross_eval_plots(test_sift=False, test_glove=True, test_yfcc=True, test_gist=False):
    evaluator = CrossEvaluator()
    
    if test_sift:
        for num_restrictions in range(1, 4):
            evaluator.plot(dataset_name="SIFT", subset_size=1.0, ds_query_param=num_restrictions)

    if test_glove:
        evaluator.plot(dataset_name="GLOVE", subset_size=0.1, ds_query_param=None)

    if test_yfcc:
        evaluator.plot(dataset_name="YFCC", subset_size=0.1, ds_query_param=None)
        evaluator.plot_selectivity_based_performance(dataset_name="YFCC", subset_size=0.1, neighbors_retrieved=10, ds_query_param=None)
    
    if test_gist:
        evaluator.plot(dataset_name="GIST", subset_size=1.0, ds_query_param=None)
        evaluator.plot_selectivity_based_performance(dataset_name="GIST", subset_size=1.0, neighbors_retrieved=10, ds_query_param=None)

def generate_cross_eval_tables(test_sift=True, test_glove=True, test_yfcc=True, test_gist=True):
    evaluator = CrossEvaluator()
    target_recalls = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    baseline_index = 'IVFIdFilter'
    evaluator.generate_speedup_table(ds_name='GIST', ds_subset_size=1.0, ds_query_param=None, baseline_index=baseline_index, target_recalls=target_recalls)
    evaluator.generate_speed_up_table_selectivity(ds_name='GIST', ds_subset_size=1.0, ds_query_param=None, baseline_index=baseline_index, target_recalls=target_recalls)

    if test_sift:
        for num_restrictions in range(1, 4):
            evaluator.generate_speedup_table(ds_name='SIFT', ds_subset_size=1.0, ds_query_param=num_restrictions, baseline_index=baseline_index, target_recalls=target_recalls)

    if test_glove:
        evaluator.generate_speedup_table(ds_name='GLOVE', ds_subset_size=1.0, ds_query_param=None, baseline_index=baseline_index, target_recalls=target_recalls)

    if test_yfcc:
        evaluator.generate_speedup_table(ds_name='YFCC', ds_subset_size=0.1, ds_query_param=None, baseline_index=baseline_index, target_recalls=target_recalls)
        evaluator.generate_speed_up_table_selectivity(ds_name='YFCC', ds_subset_size=0.1, ds_query_param=None, baseline_index=baseline_index, target_recalls=target_recalls)
    
    if test_gist:
        evaluator.generate_speedup_table(ds_name='GIST', ds_subset_size=1.0, ds_query_param=None, baseline_index=baseline_index, target_recalls=target_recalls)
        evaluator.generate_speed_up_table_selectivity(ds_name='GIST', ds_subset_size=1.0, ds_query_param=None, baseline_index=baseline_index, target_recalls=target_recalls)

if __name__ == '__main__':
    generate_cross_eval_plots()
    # generate_cross_eval_tables()