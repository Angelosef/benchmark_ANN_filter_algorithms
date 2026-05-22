import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

class ANNBenchmarkPlotter:
    def __init__(self, log_root_dir, output_dir="plots"):
        self.log_root = Path(log_root_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.registry_path = self.log_root / "master_registry.csv"
        
        # Plotting aesthetics
        sns.set_theme(style="whitegrid")
        self.palette = "viridis"

    def _load_run_data(self, run_id):
        """Reads the individual JSON metadata for a specific run."""
        json_path = self.log_root / run_id / "metadata.json"
        if not json_path.exists():
            print(f"Warning: Metadata for {run_id} not found.")
            return None
        with open(json_path, 'r') as f:
            return json.load(f)

    def _prepare_dataframe(self, dataset_name, subset_size, ds_query_param):
        """Joins registry with JSON details into a single DataFrame."""
        registry = pd.read_csv(self.registry_path)
        
        # Filter for the specific dataset and subset
        mask = (registry['dataset'] == dataset_name) & \
           (registry['subset_size'] == subset_size)
    
        if ds_query_param is not None:
            mask &= (registry['ds_query_param'] == ds_query_param)
        
        filtered = registry[mask]
        
        run_details = []
        for run_id in filtered['run_id']:
            details = self._load_run_data(run_id)
            if details:
                # Convert memory from bytes to GB for readability
                details['memory_gb'] = details['index_memory'] / (1024**3)
                run_details.append(details)
        
        return pd.DataFrame(run_details)

    def plot_recall_vs_latency(self, df, dataset_name, subset_size, ds_query_param):
        """Creates the Recall/Latency trade-off curve."""
        plt.figure(figsize=(10, 6))
        
        df = df.sort_values(by=['index_name', 'query_time'])
    
        pareto_frames = []
        
        for algo in df['index_name'].unique():
            algo_data = df[df['index_name'] == algo]
            
            max_recall_so_far = -1.0
            frontier_indices = []
            
            for i, row in algo_data.iterrows():
                if row['recall'] > max_recall_so_far:
                    frontier_indices.append(i)
                    max_recall_so_far = row['recall']
            
            pareto_frames.append(algo_data.loc[frontier_indices])

        frontier_df = pd.concat(pareto_frames)

        sns.lineplot(
            data=frontier_df, 
            x='recall', 
            y='query_time', 
            hue='index_name', 
            marker='o',
            sort=True # Ensure points are connected in order of x-axis
        )
        
        plt.yscale('log')
        plt.title(f"Recall vs Query Latency ({dataset_name} - {subset_size} - {ds_query_param})")
        plt.xlabel("Recall (Higher is better)")
        plt.ylabel("Query Time (seconds, Log Scale)")
        plt.grid(True, which="both", ls="-", alpha=0.5)
        
        fname = self.output_dir / f"{dataset_name}_{subset_size}_{ds_query_param}_recall_latency.png"
        plt.savefig(fname, bbox_inches='tight')
        plt.close()
    
    def plot_recall_vs_qps(self, df, dataset_name, subset_size, ds_query_param):
        """Creates the Recall/qps trade-off curve with lightweight percentile intervals."""
        plt.figure(figsize=(10, 6))
        sns.set_theme(style="whitegrid")
        
        df['qps'] = df['query_count'] / df['query_time']
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

        ax = sns.lineplot(
            data=frontier_df, 
            x='avg_recall', # Changed from 'recall' to match your column logic
            y='qps', 
            hue='index_name', 
            marker='o',
            sort=True
        )
        
        lines = ax.get_lines()
        unique_algos = frontier_df['index_name'].unique()
        algo_color_map = {algo: lines[i].get_color() for i, algo in enumerate(unique_algos)}

        for algo, color in algo_color_map.items():
            algo_frontier = frontier_df[frontier_df['index_name'] == algo]
            
            left_err = np.maximum(algo_frontier['avg_recall'] - algo_frontier['p5_recall'], 0.0)
            right_err = np.maximum(algo_frontier['p95_recall'] - algo_frontier['avg_recall'], 0.0)
            x_errors = [left_err, right_err]

            plt.errorbar(
                x=algo_frontier['avg_recall'],
                y=algo_frontier['qps'],
                xerr=x_errors,
                fmt='none',          # 'none' means do not plot markers again (handled by lineplot)
                ecolor=color,        # Match the exact color of the line
                alpha=0.25,          # Very transparent to keep it lightweight and clean
                elinewidth=1.5,      # Thin line to prevent cluttering
                capsize=0            # No ugly vertical ticks/caps at the ends of the bars
            )
        
        plt.yscale('log')
        plt.title(f"Recall vs qps ({dataset_name} - {subset_size} - {ds_query_param})", fontsize=14, pad=15)
        plt.xlabel("Recall (Higher is better)", fontsize=12)
        plt.ylabel("Queries per Second (Log Scale)", fontsize=12)
        plt.grid(True, which="both", ls="-", alpha=0.3)
        
        plt.xlim(-0.02, 1.02)
        
        os.makedirs(self.output_dir, exist_ok=True)
        fname = self.output_dir / f"{dataset_name}_{subset_size}_{ds_query_param}_recall_qps.png"
        plt.savefig(fname, bbox_inches='tight', dpi=300)
        plt.close()

    def plot_resource_usage(self, df, dataset_name, subset_size, ds_query_param):
        """Creates bar charts for Build Time and Memory usage."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Build Time Bar Chart
        sns.barplot(
            data=df, 
            x='index_name', 
            y='build_time', 
            hue='index_name',
            errorbar='sd',
            legend=False,
            ax=ax1, 
            palette=self.palette
        )
        ax1.set_title("Average Build Time")
        ax1.set_ylabel("Seconds")
        ax1.tick_params(axis='x', rotation=45)

        # Memory Usage Bar Chart
        sns.barplot(
            data=df, 
            x='index_name', 
            y='memory_gb', 
            hue='index_name',
            errorbar='sd',
            legend=False,
            ax=ax2, 
            palette=self.palette
        )
        ax2.set_title("Average Index Memory")
        ax2.set_ylabel("GB")
        ax2.tick_params(axis='x', rotation=45)

        plt.suptitle(f"Resource Consumption: {dataset_name} (Subset: {subset_size}) (parameter: {ds_query_param})")
        plt.tight_layout()
        
        fname = self.output_dir / f"{dataset_name}_{subset_size}_{ds_query_param}_resources.png"
        plt.savefig(fname, bbox_inches='tight')
        plt.close()

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
        self.plot_resource_usage(df, dataset_name, subset_size, p_suffix)
        print(f"Done! Plots saved in: {self.output_dir}")
    
    def load_and_plot_recall(self, run_dir):
        recalls = np.load(os.path.join(run_dir, 'recalls.npy'))
        self.plot_recall_histogram(recalls, os.path.join(run_dir, 'recall_histogram.png'))
    
    def plot_recall_histogram(self, recalls, save_path):
        plt.figure(figsize=(10, 6))
        sns.set_theme(style="whitegrid")
        
        sns.histplot(
            recalls, 
            bins=30, 
            kde=False, 
            binrange=(0.0, 1.0),
            color="skyblue", 
            edgecolor="white",
            stat="percent"
        )
        
        plt.title("Distribution of Query Recalls", fontsize=14, pad=15)
        plt.xlabel("Recall Score (0.0 to 1.0)", fontsize=12)
        plt.ylabel("Percentage of Total Queries (%)", fontsize=12)
        
        plt.xlim(-0.02, 1.02)
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        
        plt.close()
