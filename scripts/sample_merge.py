"""
Sample Merge Module

This module provides functionality to merge multiple spatial transcriptomics datasets
based on a user-defined layout, preserving spatial relationships between samples.
"""

import os
import glob
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict


def read_layout(layout_file: str) -> List[List[str]]:
    """
    Read and parse the sample layout file.
    
    Parameters:
    -----------
    layout_file : str
        Path to the tab-separated layout file
        
    Returns:
    --------
    List[List[str]]
        2D list representing the sample layout
    """
    with open(layout_file, 'r') as f:
        lines = f.readlines()
    
    sample_layout = []
    for line in lines:
        row = [cell.strip() for cell in line.strip().split('\t')]
        sample_layout.append(row)
    
    return sample_layout


def get_available_samples(data_location: str) -> Dict[str, str]:
    """
    Get all available h5ad files in the data location.
    
    Parameters:
    -----------
    data_location : str
        Path to the folder containing h5ad files
        
    Returns:
    --------
    Dict[str, str]
        Dictionary mapping sample IDs to file paths
    """
    h5ad_files = glob.glob(os.path.join(data_location, "*.h5ad"))
    
    sample_files = {}
    for file_path in h5ad_files:
        filename = os.path.basename(file_path)
        sample_id = filename.replace('.h5ad', '')
        sample_files[sample_id] = file_path
    
    return sample_files


def plot_layout(sample_layout: List[List[str]], sample_files: Dict[str, str], 
                output_file: str = None) -> plt.Figure:
    """
    Plot all samples according to the layout.
    
    Parameters:
    -----------
    sample_layout : List[List[str]]
        2D list representing the sample layout
    sample_files : Dict[str, str]
        Dictionary mapping sample IDs to file paths
    output_file : str, optional
        Path to save the figure
        
    Returns:
    --------
    plt.Figure
        The generated figure
    """
    n_rows = len(sample_layout)
    n_cols = max(len(row) for row in sample_layout)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4))
    
    # Ensure axes is 2D array
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    if n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Plot each sample in its layout position
    for i, row in enumerate(sample_layout):
        for j, sample_id in enumerate(row):
            ax = axes[i, j]
            
            # Skip empty cells
            if sample_id == '-' or sample_id == '':
                ax.axis('off')
                continue
            
            # Check if sample file exists
            if sample_id in sample_files:
                # Load the data
                adata_sample = sc.read_h5ad(sample_files[sample_id])
                
                # Plot spatial data
                sc.pl.embedding(adata_sample, basis='spatial', 
                              title=sample_id, size=10, ax=ax, show=False)
                print(f"Plotted {sample_id}: {adata_sample.n_obs} cells")
            else:
                # Sample file not found
                ax.text(0.5, 0.5, f'{sample_id}\n(not found)', 
                       ha='center', va='center', fontsize=10, color='red')
                ax.axis('off')
                print(f"Warning: {sample_id} not found in data folder")
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Layout plot saved to {output_file}")
    
    return fig


def calculate_layout_positions(sample_layout: List[List[str]], 
                               sample_files: Dict[str, str],
                               spacing: float = 100.0,
                               horizontal_spacing_multiplier: float = 1.5,
                               vertical_spacing_multiplier: float = 1.5) -> Dict[str, Tuple[float, float]]:
    """
    Calculate the offset positions for each sample in the layout.
    Samples are aligned on rows and columns based on maximum dimensions.
    
    Parameters:
    -----------
    sample_layout : List[List[str]]
        2D list representing the sample layout
    sample_files : Dict[str, str]
        Dictionary mapping sample IDs to file paths
    spacing : float
        Base spacing between samples in spatial coordinates
    horizontal_spacing_multiplier : float
        Multiplier for horizontal spacing (default 1.5)
    vertical_spacing_multiplier : float
        Multiplier for vertical spacing (default 1.5 to prevent overlap)
        
    Returns:
    --------
    Dict[str, Tuple[float, float]]
        Dictionary mapping sample IDs to (x_offset, y_offset) tuples
    """
    # First pass: get dimensions of each sample
    sample_dimensions = {}
    
    for row in sample_layout:
        for sample_id in row:
            if sample_id and sample_id != '-' and sample_id in sample_files:
                adata = sc.read_h5ad(sample_files[sample_id])
                spatial_coords = adata.obsm['spatial']
                #width = spatial_coords[:, 0].max() - spatial_coords[:, 0].min()
                #height = spatial_coords[:, 1].max() - spatial_coords[:, 1].min()
                width = spatial_coords[:, 0].max() - 0
                height = spatial_coords[:, 1].max() - 0              
                sample_dimensions[sample_id] = (width, height)
    
    # Second pass: calculate maximum width for each column
    n_cols = max(len(row) for row in sample_layout)
    col_widths = [0.0] * n_cols
    
    for row in sample_layout:
        for j, sample_id in enumerate(row):
            if sample_id and sample_id != '-' and sample_id in sample_files:
                width, _ = sample_dimensions[sample_id]
                col_widths[j] = max(col_widths[j], width)
    
    # Third pass: calculate maximum height for each row
    row_heights = []
    for row in sample_layout:
        max_height = 0.0
        for sample_id in row:
            if sample_id and sample_id != '-' and sample_id in sample_files:
                _, height = sample_dimensions[sample_id]
                max_height = max(max_height, height)
        row_heights.append(max_height)
    
    # Fourth pass: calculate offsets using aligned grid
    sample_offsets = {}
    current_y = 0.0
    
    for i, row in enumerate(sample_layout):
        current_x = 0.0
        
        for j, sample_id in enumerate(row):
            if sample_id and sample_id != '-' and sample_id in sample_files:
                sample_offsets[sample_id] = (current_x, current_y)
            
            # Move to next column position (even if cell is empty)
            current_x += col_widths[j] + (spacing * horizontal_spacing_multiplier)
        
        # Move to next row position (negative to match user layout order)
        if row_heights[i] > 0:
            current_y -= row_heights[i] + (spacing * vertical_spacing_multiplier)
    
    return sample_offsets


def merge_samples(sample_layout: List[List[str]], 
                 sample_files: Dict[str, str],
                 spacing: float = 100.0,
                 horizontal_spacing_multiplier: float = 1.5,
                 vertical_spacing_multiplier: float = 1.5) -> sc.AnnData:
    """
    Merge all samples according to the layout with appropriate spatial offsets.
    
    Parameters:
    -----------
    sample_layout : List[List[str]]
        2D list representing the sample layout
    sample_files : Dict[str, str]
        Dictionary mapping sample IDs to file paths
    spacing : float
        Base spacing between samples in spatial coordinates
    horizontal_spacing_multiplier : float
        Multiplier for horizontal spacing
    vertical_spacing_multiplier : float
        Multiplier for vertical spacing to prevent overlap
        
    Returns:
    --------
    sc.AnnData
        Merged AnnData object with adjusted spatial coordinates
    """
    # Calculate offsets for each sample
    sample_offsets = calculate_layout_positions(sample_layout, sample_files, 
                                                spacing, horizontal_spacing_multiplier,
                                                vertical_spacing_multiplier)
    
    # Collect all adata objects with offset coordinates
    adata_list = []
    
    for sample_id, (x_offset, y_offset) in sample_offsets.items():
        if sample_id in sample_files:
            print(f"Processing {sample_id}: offset ({x_offset:.1f}, {y_offset:.1f})")
            
            # Load the data
            adata = sc.read_h5ad(sample_files[sample_id])
            
            # Add sample_id to obs
            adata.obs['sample_id'] = sample_id
            
            # Get spatial coordinates and apply offset
            spatial_coords = adata.obsm['spatial'].copy()
            spatial_coords[:, 0] += x_offset
            spatial_coords[:, 1] += y_offset
            
            # Update spatial coordinates
            adata.obsm['spatial'] = spatial_coords
            
            # Keep track of original coordinates if they exist
            if 'spatial_original' in adata.obsm:
                # Already has original coordinates, keep them
                pass
            else:
                # Save pre-offset coordinates as spatial_original
                adata.obsm['spatial_original'] = adata.obsm['spatial'].copy()
            
            adata_list.append(adata)
            print(f"  Added {adata.n_obs} cells")
    
    # Concatenate all samples
    print(f"\nMerging {len(adata_list)} samples...")
    merged_adata = sc.concat(adata_list, join='outer', label='batch', 
                             keys=[adata.obs['sample_id'][0] for adata in adata_list])
    
    print(f"Merged dataset: {merged_adata.n_obs} cells, {merged_adata.n_vars} genes")
    
    return merged_adata


def plot_merged_data(merged_adata: sc.AnnData, 
                    color_by: str = 'sample_id',
                    output_file: str = None) -> plt.Figure:
    """
    Plot the merged spatial data.
    
    Parameters:
    -----------
    merged_adata : sc.AnnData
        Merged AnnData object
    color_by : str
        Column name to color the plot by
    output_file : str, optional
        Path to save the figure
        
    Returns:
    --------
    plt.Figure
        The generated figure
    """
    # Calculate figure size based on spatial extent to maintain aspect ratio
    spatial_coords = merged_adata.obsm['spatial']
    x_range = spatial_coords[:, 0].max() - spatial_coords[:, 0].min()
    y_range = spatial_coords[:, 1].max() - spatial_coords[:, 1].min()
    
    # Set base size and calculate dimensions maintaining aspect ratio
    base_size = 12
    aspect_ratio = x_range / y_range if y_range > 0 else 1.0
    
    if aspect_ratio > 1:
        # Wider than tall
        fig_width = base_size
        fig_height = base_size / aspect_ratio
    else:
        # Taller than wide
        fig_width = base_size * aspect_ratio
        fig_height = base_size
    
    fig, ax = plt.subplots(figsize=(fig_width*1.2, fig_height))
    
    sc.pl.embedding(merged_adata, basis='spatial', color=color_by,
                   title='Merged Spatial Data', size=5, ax=ax, show=False,
                   legend_loc='right margin', legend_fontoutline=2)
    
    # Modify legend to use single column
    legend = ax.get_legend()
    if legend is not None:
        legend._ncol = 1
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Merged plot saved to {output_file}")
    
    return fig


def sample_merge(layout_file: str, 
                data_location: str,
                output_file: str = "merged_adata.h5ad",
                spacing: float = 100.0,
                horizontal_spacing_multiplier: float = 1.5,
                vertical_spacing_multiplier: float = 1.5,
                plot_layout_file: str = None,
                plot_merged_file: str = None) -> sc.AnnData:
    """
    Main function to merge samples based on user-defined layout.
    
    Parameters:
    -----------
    layout_file : str
        Path to the Sample_layout_input.txt file
    data_location : str
        Path to the folder containing individual h5ad files
    output_file : str
        Path to save the merged h5ad file
    spacing : float
        Base spacing between samples in spatial coordinates
    horizontal_spacing_multiplier : float
        Multiplier for horizontal spacing (default 1.5)
    vertical_spacing_multiplier : float
        Multiplier for vertical spacing (default 1.5 to prevent overlap)
    plot_layout_file : str, optional
        Path to save the layout plot
    plot_merged_file : str, optional
        Path to save the merged data plot
        
    Returns:
    --------
    sc.AnnData
        Merged AnnData object
    """
    print("=" * 80)
    print("SAMPLE MERGE: Starting...")
    print("=" * 80)
    
    # Read layout
    print(f"\nReading layout from: {layout_file}")
    sample_layout = read_layout(layout_file)
    print(f"Layout dimensions: {len(sample_layout)} rows x {max(len(row) for row in sample_layout)} columns")
    
    # Get available samples
    print(f"\nScanning data location: {data_location}")
    sample_files = get_available_samples(data_location)
    print(f"Found {len(sample_files)} h5ad files")
    
    # Check for missing samples
    layout_samples = set()
    for row in sample_layout:
        for sample_id in row:
            if sample_id and sample_id != '-' and sample_id.strip():
                layout_samples.add(sample_id)
    
    missing_samples = layout_samples - set(sample_files.keys())
    if missing_samples:
        print(f"\n⚠️  WARNING: Missing samples: {sorted(missing_samples)}")
    else:
        print("\n✓ All samples found!")
    
    # Plot layout
    print("\n" + "=" * 80)
    print("STEP 1: Plotting individual samples in layout...")
    print("=" * 80)
    plot_layout(sample_layout, sample_files, output_file=plot_layout_file)
    plt.show()
    
    # Merge samples
    print("\n" + "=" * 80)
    print("STEP 2: Merging samples with spatial offsets...")
    print("=" * 80)
    merged_adata = merge_samples(sample_layout, sample_files, spacing=spacing,
                                horizontal_spacing_multiplier=horizontal_spacing_multiplier,
                                vertical_spacing_multiplier=vertical_spacing_multiplier)
    
    # Save merged data
    print(f"\n" + "=" * 80)
    print(f"STEP 3: Saving merged data to {output_file}...")
    print("=" * 80)
    
    # Save output file in the same directory as data_location
    output_path = os.path.join(data_location, output_file)
    merged_adata.write_h5ad(output_path)
    print(f"✓ Saved: {output_path}")
    
    # Plot merged data
    print("\n" + "=" * 80)
    print("STEP 4: Plotting merged data...")
    print("=" * 80)
    plot_merged_data(merged_adata, output_file=plot_merged_file)
    plt.show()
    
    print("\n" + "=" * 80)
    print("SAMPLE MERGE: Complete!")
    print("=" * 80)
    
    return merged_adata


# Example usage
if __name__ == "__main__":
    # Example parameters
    layout_file = "Sample_layout_input.txt"
    data_location = "/path/to/refined_cores/"
    
    # Run the merge
    merged_adata = sample_merge(
        layout_file=layout_file,
        data_location=data_location,
        output_file="merged_adata.h5ad",
        spacing=100.0,
        plot_layout_file="layout_plot.png",
        plot_merged_file="merged_plot.png"
    )
    
    print(f"\nFinal merged dataset:")
    print(f"  Cells: {merged_adata.n_obs}")
    print(f"  Genes: {merged_adata.n_vars}")
    print(f"  Samples: {merged_adata.obs['sample_id'].nunique()}")
