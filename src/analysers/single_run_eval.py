import os
import matplotlib.pyplot as plt
import numpy as np
from src.analysis.utils import load_run_data, find_selectivity_path
from src.logger import BenchmarkLogger
import json

class SingleRunEvaluator:
    def __init__(self, plot_dir='logs'):
        self.output_dir = plot_dir
        os.makedirs(self.output_dir, exist_ok=True)
        return

    def load_and_plot_recall_latency(self, run_dir):
        recalls = np.load(os.path.join(run_dir, 'recalls.npy'))
        latencies = np.load(os.path.join(run_dir, 'latencies.npy'))
        metadata = load_run_data(os.path.join(run_dir, 'metadata.json'))
        if metadata is None:
            return

        parts = run_dir.split('/')
        parts.pop(0)
        run_dir = '/'.join(parts)
        
        os.makedirs(os.path.join(self.output_dir, run_dir), exist_ok=True)
        save_path = os.path.join(self.output_dir, run_dir, 'recall_latency_hexbin.png')
        self.plot_recall_latency_hexbin(recalls, latencies, save_path, metadata)

    def plot_recall_latency_hexbin(self, recalls, latencies, save_path, metadata):
        fig, ax = plt.subplots(figsize=(8, 6))
        
        hb = ax.hexbin(
            recalls, 
            latencies, 
            gridsize=10, 
            cmap='YlGnBu', 
            mincnt=1
        )

        index_name = metadata['index_name']
        ds_name = metadata['dataset_name']

        info_text = f"Dataset: {ds_name}\nIndex: {index_name}"
        
        ax.text(
            0.05, 0.95, info_text, 
            transform=ax.transAxes, 
            fontsize=10,
            verticalalignment='top', 
            horizontalalignment='left',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='lightgray')
        )
        
        ax.set_title('Recall vs Latency Density', fontsize=14, pad=15)
        ax.set_xlabel('Recall', fontsize=12)
        ax.set_ylabel('Latency (s)', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.5)

        cb = fig.colorbar(hb, ax=ax)
        cb.set_label('Number of Queries (Density)', fontsize=12)

        output_dir = os.path.dirname(save_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()

    def _compute_binned_averages(self, selectivities, values, num_points=200, window_ratio=0.15):
        """
        Calculates moving window averages using overlapping rectangular windows in log-space.
        
        Parameters:
        - num_points: Number of log-spaced evaluation points (higher = smoother curve).
        - window_ratio: Width of the window in log10 space (e.g., 0.15 means the window spans 
                        [center / 10^0.15, center * 10^0.15] in linear space).
        """
        # Ensure positive values for log conversion
        min_sel = max(selectivities.min(), 1e-9)
        max_sel = selectivities.max()

        # Generate finely spaced evaluation points across the log range
        grid_x = np.geomspace(min_sel, max_sel, num_points)

        # Convert data and grid points to log10 space for uniform windowing
        log_selectivities = np.log10(selectivities)
        log_grid = np.log10(grid_x)

        half_window = window_ratio / 2.0

        binned_x = []
        binned_y = []

        for x_val, log_x in zip(grid_x, log_grid):
            # Mask points that fall inside the rectangular window in log-space
            mask = (log_selectivities >= log_x - half_window) & (log_selectivities <= log_x + half_window)
            
            if np.any(mask):
                binned_x.append(x_val)
                binned_y.append(values[mask].mean())

        return np.array(binned_x), np.array(binned_y)

    def load_and_plot_selectivity_avg_recall(self, run_dir, selectivity_path):
        recalls = np.load(os.path.join(run_dir, 'recalls.npy'))
        selectivities = np.load(selectivity_path)
        metadata = load_run_data(os.path.join(run_dir, 'metadata.json'))

        if metadata is None:
            return

        parts = run_dir.split('/')
        parts.pop(0)
        run_dir = '/'.join(parts)
        
        os.makedirs(os.path.join(self.output_dir, run_dir), exist_ok=True)
        selectivities = selectivities / metadata['base_count']
        save_path = os.path.join(self.output_dir, run_dir, 'selectivity_avg_recall.png')
        self.plot_selectivity_avg_recall(selectivities, recalls, save_path, metadata)

    def plot_selectivity_avg_recall(self, selectivities, recalls, save_path, metadata, num_points=200, window_ratio=0.15):
        grid_x, grid_y = self._compute_binned_averages(
            selectivities, recalls, num_points=num_points, window_ratio=window_ratio
        )

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(selectivities, recalls, alpha=0.05, color='gray', s=10)
        ax.plot(grid_x, grid_y, color='crimson', linewidth=3)
        
        ax.set_xscale('log')
        ax.set_title('Selectivity vs Recall')
        ax.set_xlabel('Selectivity (log scale)')
        ax.set_ylabel('Recall')
        ax.set_ylim(-0.05, 1.05)

        info_text = f"Dataset: {metadata['dataset_name']}\nIndex: {metadata['index_name']}"
        ax.text(
            0.05, 0.95, info_text, 
            transform=ax.transAxes, 
            fontsize=10,
            verticalalignment='top', 
            horizontalalignment='left',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='lightgray')
        )

        output_dir = os.path.dirname(save_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()

    def load_and_plot_selectivity_avg_latency(self, run_dir, selectivity_path):
        latencies = np.load(os.path.join(run_dir, 'latencies.npy'))
        selectivities = np.load(selectivity_path)
        metadata = load_run_data(os.path.join(run_dir, 'metadata.json'))

        if metadata is None:
            return

        parts = run_dir.split('/')
        parts.pop(0)
        run_dir = '/'.join(parts)

        os.makedirs(os.path.join(self.output_dir, run_dir), exist_ok=True)
        selectivities = selectivities / metadata['base_count']
        save_path = os.path.join(self.output_dir, run_dir, 'selectivity_avg_latency.png')
        self.plot_selectivity_avg_latency(selectivities, latencies, save_path, metadata)

    def plot_selectivity_avg_latency(self, selectivities, latencies, save_path, metadata, num_points=200, window_ratio=0.15):
        grid_x, grid_y = self._compute_binned_averages(
            selectivities, latencies, num_points=num_points, window_ratio=window_ratio
        )

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(selectivities, latencies, alpha=0.05, color='gray', s=10)
        ax.plot(grid_x, grid_y, color='crimson', linewidth=3)
        
        ax.set_xscale('log')
        ax.set_title('Selectivity vs Latency')
        ax.set_xlabel('Selectivity (log scale)')
        ax.set_ylabel('Latency (s)')

        info_text = f"Dataset: {metadata['dataset_name']}\nIndex: {metadata['index_name']}"
        ax.text(
            0.05, 0.95, info_text, 
            transform=ax.transAxes, 
            fontsize=10,
            verticalalignment='top', 
            horizontalalignment='left',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='lightgray')
        )

        output_dir = os.path.dirname(save_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()


if __name__=='__main__':
    logger = BenchmarkLogger()
    plotter = SingleRunEvaluator()
    
    target_dir = logger.get_log_dir()
    run_dirs = [
        os.path.join(target_dir, name) 
        for name in os.listdir(target_dir) 
            if os.path.isdir(os.path.join(target_dir, name))
    ]

    # run_dirs = ['logs/IVFSquaredFaiss_GIST_20260809-003122']

    for run_directory in run_dirs:
        
        plotter.load_and_plot_recall_latency(run_directory)

        with open(os.path.join(run_directory, "metadata.json"), "r") as f:
            metadata = json.load(f)
        
        sel_path = find_selectivity_path(
            metadata.get('dataset_name'),
            metadata.get('subset_size'),
            metadata.get('neighbors_retrieved'),
            metadata.get('ds_query_param')
        )

        plotter.load_and_plot_selectivity_avg_recall(run_directory, sel_path)
        plotter.load_and_plot_selectivity_avg_latency(run_directory, sel_path)