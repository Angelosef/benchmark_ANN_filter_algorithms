import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

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
        self.plot_resource_usage(df, dataset_name, subset_size, p_suffix)
        print(f"Done! Plots saved in: {self.output_dir}")
        