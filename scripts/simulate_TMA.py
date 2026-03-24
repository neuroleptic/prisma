#!/usr/bin/env python3
"""
Simulate TMA (Tissue Microarray) array for spatial transcriptomics.

This script generates simulated spatial transcriptomics data that mimics real TMA arrays
with round or square core shapes, variable cell densities, and realistic edge effects.
Output is compatible with anndata/scanpy workflows.
"""

import numpy as np
import pandas as pd
import anndata as ad
from scipy import ndimage
from scipy.spatial import ConvexHull
from scipy.ndimage import binary_erosion, binary_dilation
import matplotlib.pyplot as plt
from matplotlib.path import Path
from typing import Tuple, List, Optional, Literal
import argparse
import json


def generate_organic_shape(center: Tuple[float, float], 
                           base_size: float,
                           shape_type: str = 'square',
                           irregularity: float = 0.3,
                           jaggedness: float = 0.15,
                           num_vertices: int = 40,
                           missing_chunk_prob: float = 0.3,
                           edge_cut: Optional[str] = None,
                           edge_cut_amount: float = 0.3) -> np.ndarray:
    """
    Generate a highly irregular, organic tissue shape.
    
    Parameters
    ----------
    center : tuple
        (x, y) center coordinates
    base_size : float
        Base size of the core
    shape_type : str
        'square' or 'round' - base shape before adding irregularity
    irregularity : float
        Overall shape distortion (0-1)
    jaggedness : float
        High-frequency edge noise for jagged boundaries (0-1)
    num_vertices : int
        Number of vertices for the boundary
    missing_chunk_prob : float
        Probability of having a missing chunk/indent
    edge_cut : str or None
        Which edge to cut ('top', 'bottom', 'left', 'right', or None)
    edge_cut_amount : float
        How much of the edge to cut (0-1)
    
    Returns
    -------
    np.ndarray
        Array of (x, y) vertex coordinates
    """
    cx, cy = center
    
    # Generate angles
    angles = np.linspace(0, 2 * np.pi, num_vertices, endpoint=False)
    
    # Base radius - keep consistent size (punch tool is uniform)
    # Irregularity comes from shape, not overall size
    base_radius = base_size / 2
    
    # For square-ish shapes, modify the base radius based on angle
    if shape_type == 'square':
        # Create a rounded square using superellipse formula
        # Higher n = more square, lower n = more round
        n = np.random.uniform(2.5, 4.0)  # Vary squareness
        cos_a = np.cos(angles)
        sin_a = np.sin(angles)
        # Superellipse radius
        base_radii = base_radius / (np.abs(cos_a)**n + np.abs(sin_a)**n)**(1/n)
        # Add rotation to break perfect alignment
        rotation = np.random.uniform(-0.15, 0.15)  # Small random rotation
        angles = angles + rotation
    else:
        base_radii = np.full(num_vertices, base_radius)
    
    # Add smooth low-frequency variation (overall shape distortion)
    n_waves = np.random.randint(2, 5)
    smooth_variation = np.zeros(num_vertices)
    for _ in range(n_waves):
        phase = np.random.uniform(0, 2 * np.pi)
        freq = np.random.randint(1, 4)
        amplitude = np.random.uniform(0.05, irregularity / n_waves)
        smooth_variation += amplitude * np.sin(freq * angles + phase)
    
    # Add high-frequency jagged noise
    jagged_noise = np.random.uniform(-jaggedness, jaggedness, num_vertices)
    # Smooth the jagged noise slightly to avoid too sharp transitions
    jagged_noise = np.convolve(jagged_noise, [0.2, 0.6, 0.2], mode='same')
    
    # Combine all variations
    radii = base_radii * (1 + smooth_variation + jagged_noise)
    
    # Add missing chunks (indentations)
    if np.random.random() < missing_chunk_prob:
        n_chunks = np.random.randint(1, 3)
        for _ in range(n_chunks):
            # Random position for the chunk
            chunk_center = np.random.randint(0, num_vertices)
            chunk_width = np.random.randint(3, 8)
            chunk_depth = np.random.uniform(0.2, 0.5)
            
            # Create indent
            for i in range(-chunk_width, chunk_width + 1):
                idx = (chunk_center + i) % num_vertices
                # Gaussian-like falloff
                falloff = np.exp(-0.5 * (i / (chunk_width / 2))**2)
                radii[idx] *= (1 - chunk_depth * falloff)
    
    # Add occasional protrusions (bumps)
    if np.random.random() < 0.4:
        n_bumps = np.random.randint(1, 3)
        for _ in range(n_bumps):
            bump_center = np.random.randint(0, num_vertices)
            bump_width = np.random.randint(2, 5)
            bump_height = np.random.uniform(0.1, 0.25)
            
            for i in range(-bump_width, bump_width + 1):
                idx = (bump_center + i) % num_vertices
                falloff = np.exp(-0.5 * (i / (bump_width / 2))**2)
                radii[idx] *= (1 + bump_height * falloff)
    
    # Convert to cartesian coordinates
    vertices = np.column_stack([
        cx + radii * np.cos(angles),
        cy + radii * np.sin(angles)
    ])
    
    # Apply edge cutting to simulate tissue at image boundary
    if edge_cut is not None:
        cut_boundary = base_size * edge_cut_amount
        if edge_cut == 'left':
            mask = vertices[:, 0] < cx - base_size/2 + cut_boundary
            vertices[mask, 0] = cx - base_size/2 + cut_boundary + np.random.uniform(-2, 2, mask.sum())
        elif edge_cut == 'right':
            mask = vertices[:, 0] > cx + base_size/2 - cut_boundary
            vertices[mask, 0] = cx + base_size/2 - cut_boundary + np.random.uniform(-2, 2, mask.sum())
        elif edge_cut == 'top':
            mask = vertices[:, 1] < cy - base_size/2 + cut_boundary
            vertices[mask, 1] = cy - base_size/2 + cut_boundary + np.random.uniform(-2, 2, mask.sum())
        elif edge_cut == 'bottom':
            mask = vertices[:, 1] > cy + base_size/2 - cut_boundary
            vertices[mask, 1] = cy + base_size/2 - cut_boundary + np.random.uniform(-2, 2, mask.sum())
    
    return vertices


def generate_irregular_square(center: Tuple[float, float], 
                               base_size: float,
                               irregularity: float = 0.3,
                               corner_noise: float = 0.15,
                               edge_cut: Optional[str] = None,
                               edge_cut_amount: float = 0.3) -> np.ndarray:
    """Generate irregular square-like shape. Wrapper for generate_organic_shape."""
    return generate_organic_shape(
        center=center,
        base_size=base_size,
        shape_type='square',
        irregularity=irregularity,
        jaggedness=corner_noise,
        num_vertices=np.random.randint(35, 50),
        missing_chunk_prob=0.4,
        edge_cut=edge_cut,
        edge_cut_amount=edge_cut_amount
    )


def generate_irregular_circle(center: Tuple[float, float], 
                               base_radius: float,
                               irregularity: float = 0.25,
                               num_vertices: int = 40,
                               edge_cut: Optional[str] = None,
                               edge_cut_amount: float = 0.3) -> np.ndarray:
    """Generate irregular circular shape. Wrapper for generate_organic_shape."""
    return generate_organic_shape(
        center=center,
        base_size=base_radius * 2,
        shape_type='round',
        irregularity=irregularity,
        jaggedness=irregularity * 0.5,
        num_vertices=num_vertices,
        missing_chunk_prob=0.35,
        edge_cut=edge_cut,
        edge_cut_amount=edge_cut_amount
    )


def generate_cell_positions(polygon_vertices: np.ndarray,
                           cell_density: float = 0.02,
                           density_variation: float = 0.5,
                           min_cell_distance: float = 3.0) -> np.ndarray:
    """
    Generate cell positions within a polygon with realistic clustering.
    
    Parameters
    ----------
    polygon_vertices : np.ndarray
        Vertices defining the core boundary
    cell_density : float
        Base density of cells per unit area
    density_variation : float
        How much density varies across the tissue (0-1)
    min_cell_distance : float
        Minimum distance between cells
    
    Returns
    -------
    np.ndarray
        Array of (x, y) cell positions
    """
    # Get bounding box
    min_x, min_y = polygon_vertices.min(axis=0)
    max_x, max_y = polygon_vertices.max(axis=0)
    
    # Calculate area and expected number of cells
    try:
        hull = ConvexHull(polygon_vertices)
        area = hull.volume  # In 2D, volume gives area
    except:
        area = (max_x - min_x) * (max_y - min_y) * 0.7
    
    # Vary density per core
    actual_density = cell_density * (1 + np.random.uniform(-density_variation, density_variation))
    expected_cells = int(area * actual_density)
    
    if expected_cells <= 0:
        return np.array([]).reshape(0, 2)
    
    # Generate candidate cells using rejection sampling
    cells = []
    attempts = 0
    max_attempts = expected_cells * 100
    
    # Create a path for point-in-polygon test
    polygon_path = Path(polygon_vertices)
    
    while len(cells) < expected_cells and attempts < max_attempts:
        # Generate random point
        x = np.random.uniform(min_x, max_x)
        y = np.random.uniform(min_y, max_y)
        
        # Check if inside polygon
        if polygon_path.contains_point([x, y]):
            # Check minimum distance to existing cells
            if len(cells) == 0 or min(np.sqrt((np.array(cells)[:, 0] - x)**2 + 
                                               (np.array(cells)[:, 1] - y)**2)) >= min_cell_distance:
                cells.append([x, y])
        
        attempts += 1
    
    return np.array(cells) if cells else np.array([]).reshape(0, 2)


def add_cell_clusters(cells: np.ndarray,
                     polygon_vertices: np.ndarray,
                     num_clusters: int = 3,
                     cluster_cells: int = 50,
                     cluster_spread: float = 15.0) -> np.ndarray:
    """
    Add clustered regions of cells to simulate tissue heterogeneity.
    
    Parameters
    ----------
    cells : np.ndarray
        Existing cell positions
    polygon_vertices : np.ndarray
        Vertices defining the core boundary
    num_clusters : int
        Number of dense clusters to add
    cluster_cells : int
        Approximate number of cells per cluster
    cluster_spread : float
        Standard deviation of cluster spread
    
    Returns
    -------
    np.ndarray
        Updated array of cell positions including clusters
    """
    polygon_path = Path(polygon_vertices)
    
    # Get bounding box
    min_x, min_y = polygon_vertices.min(axis=0)
    max_x, max_y = polygon_vertices.max(axis=0)
    
    new_cells = list(cells)
    
    for _ in range(num_clusters):
        # Random cluster center inside polygon
        attempts = 0
        while attempts < 100:
            cx = np.random.uniform(min_x, max_x)
            cy = np.random.uniform(min_y, max_y)
            if polygon_path.contains_point([cx, cy]):
                break
            attempts += 1
        
        if attempts >= 100:
            continue
        
        # Generate clustered cells
        n_cells = int(cluster_cells * np.random.uniform(0.5, 1.5))
        for _ in range(n_cells):
            x = cx + np.random.normal(0, cluster_spread)
            y = cy + np.random.normal(0, cluster_spread)
            if polygon_path.contains_point([x, y]):
                new_cells.append([x, y])
    
    return np.array(new_cells)


def simulate_tma_array(n_rows: int = 6,
                       n_cols: int = 3,
                       core_shape: Literal['round', 'square'] = 'square',
                       core_size: float = 150.0,
                       spacing: float = 30.0,
                       cell_density: float = 0.015,
                       irregularity: float = 0.3,
                       missing_cores: float = 0.1,
                       edge_core_prob: float = 0.2,
                       position_jitter: float = 0.3,
                       overlap_prob: float = 0.2,
                       n_genes: int = 100,
                       random_seed: Optional[int] = None) -> ad.AnnData:
    """
    Simulate a complete TMA array with multiple cores.
    
    Parameters
    ----------
    n_rows : int
        Number of rows in the TMA grid
    n_cols : int
        Number of columns in the TMA grid
    core_shape : str
        Shape of cores: 'round' or 'square'
    core_size : float
        Base size of each core (diameter for round, side length for square)
    spacing : float
        Spacing between cores
    cell_density : float
        Base cell density
    irregularity : float
        How irregular the core shapes are (0-1). 
        Recommended: 0.2-0.4 for realistic tissue appearance
    missing_cores : float
        Fraction of cores that are missing/empty
    edge_core_prob : float
        Probability that a core touches an edge
    position_jitter : float
        How much to offset cores from perfect grid (0-1, as fraction of spacing)
    overlap_prob : float
        Probability that adjacent cores will touch/overlap
    n_genes : int
        Number of genes to simulate
    random_seed : int or None
        Random seed for reproducibility
    
    Returns
    -------
    ad.AnnData
        AnnData object with simulated spatial transcriptomics data
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    all_cells = []
    all_core_ids = []
    all_core_polygons = []
    
    core_id = 0
    
    # Pre-generate which adjacent pairs will overlap
    overlap_pairs = set()
    for row in range(n_rows):
        for col in range(n_cols):
            # Check right neighbor
            if col < n_cols - 1 and np.random.random() < overlap_prob:
                overlap_pairs.add(((row, col), (row, col + 1)))
            # Check bottom neighbor
            if row < n_rows - 1 and np.random.random() < overlap_prob:
                overlap_pairs.add(((row, col), (row + 1, col)))
    
    for row in range(n_rows):
        for col in range(n_cols):
            # Randomly skip some cores
            if np.random.random() < missing_cores:
                continue
            
            # Calculate base center position
            cx = col * (core_size + spacing) + core_size / 2 + spacing
            cy = row * (core_size + spacing) + core_size / 2 + spacing
            
            # Add position jitter (random offset from perfect grid)
            jitter_amount = spacing * position_jitter
            cx += np.random.uniform(-jitter_amount, jitter_amount)
            cy += np.random.uniform(-jitter_amount, jitter_amount)
            
            # Check if this core should extend toward a neighbor (overlap)
            size_multiplier = 1.0
            extend_direction = None
            
            # Check for overlap with right neighbor - shift position, not size
            if ((row, col), (row, col + 1)) in overlap_pairs:
                extend_direction = 'right'
                cx += spacing * 0.25  # Shift toward neighbor for overlap
            # Check for overlap with left neighbor (from their perspective)
            elif ((row, col - 1), (row, col)) in overlap_pairs and col > 0:
                extend_direction = 'left'
                cx -= spacing * 0.25
            # Check for overlap with bottom neighbor
            elif ((row, col), (row + 1, col)) in overlap_pairs:
                extend_direction = 'bottom'
                cy += spacing * 0.25
            # Check for overlap with top neighbor
            elif ((row - 1, col), (row, col)) in overlap_pairs and row > 0:
                extend_direction = 'top'
                cy -= spacing * 0.25
            
            # Determine if this core should touch an edge
            edge_cut = None
            if np.random.random() < edge_core_prob:
                edge_options = []
                if col == 0:
                    edge_options.append('left')
                if col == n_cols - 1:
                    edge_options.append('right')
                if row == 0:
                    edge_options.append('top')
                if row == n_rows - 1:
                    edge_options.append('bottom')
                
                # Also allow random edge cuts for interior cores occasionally
                if not edge_options and np.random.random() < 0.4:
                    edge_options = ['left', 'right', 'top', 'bottom']
                
                if edge_options:
                    edge_cut = np.random.choice(edge_options)
            
            # Generate core shape based on type
            # Slight size variation (punch tool is mostly consistent)
            current_size = core_size * np.random.uniform(0.95, 1.05) * size_multiplier
            
            if core_shape == 'square':
                polygon_vertices = generate_irregular_square(
                    center=(cx, cy),
                    base_size=current_size,
                    irregularity=irregularity,
                    corner_noise=irregularity * 0.6,
                    edge_cut=edge_cut,
                    edge_cut_amount=np.random.uniform(0.15, 0.5) if edge_cut else 0
                )
            else:  # round
                polygon_vertices = generate_irregular_circle(
                    center=(cx, cy),
                    base_radius=current_size / 2,
                    irregularity=irregularity,
                    num_vertices=np.random.randint(35, 50),
                    edge_cut=edge_cut,
                    edge_cut_amount=np.random.uniform(0.15, 0.5) if edge_cut else 0
                )
            
            all_core_polygons.append(polygon_vertices)
            
            # Generate cells within core
            cells = generate_cell_positions(
                polygon_vertices,
                cell_density=cell_density * np.random.uniform(0.5, 1.5),
                density_variation=0.3,
                min_cell_distance=2.0
            )
            
            # Add some clustered regions
            if len(cells) > 0 and np.random.random() < 0.7:
                cells = add_cell_clusters(
                    cells, 
                    polygon_vertices,
                    num_clusters=np.random.randint(1, 5),
                    cluster_cells=np.random.randint(20, 80),
                    cluster_spread=core_size * 0.1
                )
            
            if len(cells) > 0:
                all_cells.append(cells)
                all_core_ids.extend([core_id] * len(cells))
            
            core_id += 1
    
    if not all_cells:
        print("Warning: No cells generated!")
        return None
    
    # Combine all cells
    all_cells = np.vstack(all_cells)
    
    # Create pseudo transcript counts
    # 1 for cells (presence), can add variation
    n_cells = len(all_cells)
    
    # Generate gene expression matrix
    # Some genes expressed in all cells (housekeeping), others variable
    X = np.zeros((n_cells, n_genes), dtype=np.float32)
    
    # Housekeeping genes (always expressed)
    n_housekeeping = n_genes // 10
    X[:, :n_housekeeping] = 1
    
    # Variable genes with some spatial pattern
    for gene_idx in range(n_housekeeping, n_genes):
        # Random expression probability per gene
        expr_prob = np.random.uniform(0.1, 0.8)
        X[:, gene_idx] = (np.random.random(n_cells) < expr_prob).astype(np.float32)
    
    # Create gene names
    gene_names = [f"Gene_{i:04d}" for i in range(n_genes)]
    
    # Create cell IDs
    cell_ids = [f"Cell_{i:06d}" for i in range(n_cells)]
    
    # Create AnnData object
    adata = ad.AnnData(X=X)
    adata.obs_names = cell_ids
    adata.var_names = gene_names
    
    # Add spatial coordinates
    adata.obsm['spatial'] = all_cells
    
    # Add metadata
    adata.obs['core_id'] = all_core_ids
    adata.obs['x'] = all_cells[:, 0]
    adata.obs['y'] = all_cells[:, 1]
    adata.obs['n_counts'] = X.sum(axis=1)
    
    # Store core polygons for visualization (as JSON string for h5ad compatibility)
    adata.uns['core_polygons_json'] = json.dumps([p.tolist() for p in all_core_polygons])
    adata.uns['tma_params'] = {
        'n_rows': n_rows,
        'n_cols': n_cols,
        'core_shape': core_shape,
        'core_size': core_size,
        'spacing': spacing,
        'irregularity': irregularity,
        'position_jitter': position_jitter,
        'overlap_prob': overlap_prob,
        'n_cores_generated': core_id,
        'n_cells_total': n_cells
    }
    
    return adata


def plot_tma(adata: ad.AnnData, 
             color_by: str = 'core_id',
             figsize: Tuple[int, int] = (10, 15),
             point_size: float = 1.0,
             show_polygons: bool = True,
             save_path: Optional[str] = None):
    """
    Plot the simulated TMA array.
    
    Parameters
    ----------
    adata : ad.AnnData
        AnnData object with spatial data
    color_by : str
        Column in obs to color by
    figsize : tuple
        Figure size
    point_size : float
        Size of cell points
    show_polygons : bool
        Whether to show core boundaries
    save_path : str or None
        Path to save figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot cells
    coords = adata.obsm['spatial']
    
    # Get unique categories and assign colors
    categories = adata.obs[color_by] if color_by in adata.obs else np.zeros(len(adata))
    unique_cats = np.unique(categories)
    n_cats = len(unique_cats)
    
    # Use tab20 colormap for up to 20 categories, otherwise use a continuous colormap
    if n_cats <= 20:
        cmap = plt.cm.tab20
        colors_map = {cat: cmap(i % 20) for i, cat in enumerate(unique_cats)}
    else:
        cmap = plt.cm.nipy_spectral
        colors_map = {cat: cmap(i / n_cats) for i, cat in enumerate(unique_cats)}
    
    # Plot each category separately for proper legend
    for cat in unique_cats:
        mask = categories == cat
        ax.scatter(coords[mask, 0], coords[mask, 1], 
                  c=[colors_map[cat]], s=point_size, alpha=0.8,
                  label=f'Core {int(cat)}')
    
    # Plot core boundaries
    if show_polygons and 'core_polygons_json' in adata.uns:
        polygons = json.loads(adata.uns['core_polygons_json'])
        for poly in polygons:
            poly = np.array(poly)
            poly_closed = np.vstack([poly, poly[0]])  # Close the polygon
            ax.plot(poly_closed[:, 0], poly_closed[:, 1], 'gray', alpha=0.3, linewidth=0.5)
    
    ax.set_aspect('equal')
    ax.set_facecolor('white')
    ax.invert_yaxis()
    ax.set_xlabel('X coordinate')
    ax.set_ylabel('Y coordinate')
    
    core_shape = adata.uns.get('tma_params', {}).get('core_shape', 'unknown')
    ax.set_title(f'Simulated TMA Array - {core_shape} cores ({len(adata)} cells)')
    
    # Categorical legend
    legend = ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), 
                       title=color_by, fontsize='small', 
                       markerscale=3, frameon=True, facecolor='white')
    legend.get_title().set_fontsize('medium')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='black')
        print(f"Figure saved to {save_path}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Simulate TMA array for spatial transcriptomics')
    parser.add_argument('--rows', type=int, default=6, help='Number of rows in TMA grid')
    parser.add_argument('--cols', type=int, default=3, help='Number of columns in TMA grid')
    parser.add_argument('--shape', type=str, default='square', choices=['round', 'square'],
                        help='Core shape: round or square')
    parser.add_argument('--core-size', type=float, default=150.0, help='Base core size')
    parser.add_argument('--spacing', type=float, default=30.0, help='Spacing between cores')
    parser.add_argument('--cell-density', type=float, default=0.015, help='Cell density per area')
    parser.add_argument('--irregularity', type=float, default=0.3, help='Shape irregularity (0-1), higher=messier')
    parser.add_argument('--missing', type=float, default=0.1, help='Fraction of missing cores')
    parser.add_argument('--edge-prob', type=float, default=0.2, help='Probability of edge-cut core')
    parser.add_argument('--jitter', type=float, default=0.3, help='Position jitter (0-1), how much to offset from grid')
    parser.add_argument('--overlap', type=float, default=0.2, help='Overlap probability (0-1), chance of touching neighbors')
    parser.add_argument('--n-genes', type=int, default=100, help='Number of genes to simulate')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    parser.add_argument('--output', type=str, default='simulated_tma.h5ad', help='Output file path')
    parser.add_argument('--plot', action='store_true', help='Generate visualization')
    parser.add_argument('--plot-output', type=str, default=None, help='Save plot to file')
    
    args = parser.parse_args()
    
    print(f"Simulating TMA array: {args.rows} rows x {args.cols} cols")
    print(f"Core shape: {args.shape}, Size: {args.core_size}, Spacing: {args.spacing}")
    print(f"Irregularity: {args.irregularity}, Jitter: {args.jitter}, Overlap: {args.overlap}")
    
    # Generate simulated TMA
    adata = simulate_tma_array(
        n_rows=args.rows,
        n_cols=args.cols,
        core_shape=args.shape,
        core_size=args.core_size,
        spacing=args.spacing,
        cell_density=args.cell_density,
        irregularity=args.irregularity,
        missing_cores=args.missing,
        edge_core_prob=args.edge_prob,
        position_jitter=args.jitter,
        overlap_prob=args.overlap,
        n_genes=args.n_genes,
        random_seed=args.seed
    )
    
    if adata is None:
        print("Error: Failed to generate TMA array")
        return
    
    print(f"\nGenerated TMA array:")
    print(f"  Total cells: {len(adata)}")
    print(f"  Genes: {adata.n_vars}")
    print(f"  Corees: {len(np.unique(adata.obs['core_id']))}")
    
    # Save to h5ad
    adata.write_h5ad(args.output)
    print(f"\nSaved to {args.output}")
    
    # Plot if requested
    if args.plot:
        plot_tma(adata, color_by='core_id', save_path=args.plot_output)


if __name__ == '__main__':
    main()


# Example usage in interactive mode:
# ================================
# 
# from simulate_TMA import simulate_tma_array, plot_tma
#
# # Generate realistic TMA with SQUARE cores (messy, overlapping)
# adata_square = simulate_tma_array(
#     n_rows=6,
#     n_cols=3,
#     core_shape='square',
#     core_size=150,
#     spacing=20,  # Reduced spacing for closer samples
#     cell_density=0.015,
#     irregularity=0.35,  # Higher for organic, messy shapes
#     missing_cores=0.05,
#     edge_core_prob=0.3,
#     position_jitter=0.35,  # Offset from perfect grid
#     overlap_prob=0.25,  # Some samples touch/overlap
#     n_genes=100,
#     random_seed=42
# )
# plot_tma(adata_square, color_by='core_id')
# adata_square.write_h5ad('square_tma.h5ad')
#
# # Generate TMA with ROUND cores
# adata_round = simulate_tma_array(
#     n_rows=6,
#     n_cols=3,
#     core_shape='round',
#     core_size=150,
#     spacing=20,
#     cell_density=0.015,
#     irregularity=0.3,
#     missing_cores=0.05,
#     edge_core_prob=0.3,
#     position_jitter=0.3,
#     overlap_prob=0.2,
#     n_genes=100,
#     random_seed=42
# )
# plot_tma(adata_round, color_by='core_id')
# adata_round.write_h5ad('round_tma.h5ad')
