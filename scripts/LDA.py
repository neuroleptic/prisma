"""
LDA.py - Advanced Boundary Estimation for Core Separation

This script performs sophisticated separation of spatial transcriptomics cores
using multiple methods:
1. Gaussian Mixture Model (GMM) - fits 2D Gaussian distributions to each core
2. Linear Discriminant Analysis (LDA) - optimal linear boundary
3. Bayesian Decision Boundary - where P(core1|x,y) = P(core2|x,y)

The best boundary is estimated by finding the equiprobability contour where
the posterior probabilities of belonging to either core are equal.

Usage:
    python LDA.py cores_6_7.h5ad
    
Or import as module:
    import LDA
    results = LDA.separate_cores(adata)

Author: TMA Spatial Analysis
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import scanpy as sc
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from scipy.stats import multivariate_normal
from scipy.optimize import minimize_scalar
from typing import Tuple, Dict, Optional, List
import anndata
import warnings
warnings.filterwarnings('ignore')


class CoreSeparator:
    """
    Advanced core separation using multiple methods.
    
    Fits 2D Gaussian distributions to each core and finds the optimal
    decision boundary where posterior probabilities are equal.
    """
    
    def __init__(self, adata: anndata.AnnData, spatial_key: str = 'spatial'):
        """
        Initialize with AnnData object containing two cores.
        
        Parameters
        ----------
        adata : anndata.AnnData
            AnnData with 'core_id' in obs and spatial coordinates in obsm
        spatial_key : str
            Key for spatial coordinates in adata.obsm
        """
        self.adata = adata
        self.spatial_key = spatial_key
        self.coords = adata.obsm[spatial_key]
        
        # Get unique core IDs
        self.core_ids = sorted(adata.obs['core_id'].unique())
        if len(self.core_ids) != 2:
            raise ValueError(f"Expected 2 cores, found {len(self.core_ids)}: {self.core_ids}")
        
        self.core1_id, self.core2_id = self.core_ids
        
        # Extract coordinates for each core
        mask1 = adata.obs['core_id'] == self.core1_id
        mask2 = adata.obs['core_id'] == self.core2_id
        
        self.coords1 = self.coords[mask1]
        self.coords2 = self.coords[mask2]
        
        # Results storage
        self.results = {}
        
    def fit_gaussians(self) -> Dict:
        """
        Fit 2D Gaussian distributions to each core.
        
        Returns
        -------
        params : dict
            Fitted parameters for each core (mean, covariance)
        """
        # Compute parameters for Core 1
        mean1 = np.mean(self.coords1, axis=0)
        cov1 = np.cov(self.coords1.T)
        
        # Compute parameters for Core 2
        mean2 = np.mean(self.coords2, axis=0)
        cov2 = np.cov(self.coords2.T)
        
        # Create multivariate normal distributions
        self.dist1 = multivariate_normal(mean=mean1, cov=cov1)
        self.dist2 = multivariate_normal(mean=mean2, cov=cov2)
        
        # Store priors (proportion of cells in each core)
        n_total = len(self.coords1) + len(self.coords2)
        self.prior1 = len(self.coords1) / n_total
        self.prior2 = len(self.coords2) / n_total
        
        self.results['gaussian'] = {
            'core1': {
                'mean': mean1,
                'cov': cov1,
                'n_cells': len(self.coords1),
                'prior': self.prior1
            },
            'core2': {
                'mean': mean2,
                'cov': cov2,
                'n_cells': len(self.coords2),
                'prior': self.prior2
            }
        }
        
        return self.results['gaussian']
    
    def fit_gmm(self, n_components_per_core: int = 1) -> Dict:
        """
        Fit Gaussian Mixture Model to better capture complex distributions.
        
        Parameters
        ----------
        n_components_per_core : int
            Number of Gaussian components per core (for complex shapes)
            
        Returns
        -------
        params : dict
            GMM parameters
        """
        # Fit GMM to each core separately
        self.gmm1 = GaussianMixture(n_components=n_components_per_core, 
                                     covariance_type='full', 
                                     random_state=42)
        self.gmm1.fit(self.coords1)
        
        self.gmm2 = GaussianMixture(n_components=n_components_per_core,
                                     covariance_type='full',
                                     random_state=42)
        self.gmm2.fit(self.coords2)
        
        self.results['gmm'] = {
            'core1': {
                'means': self.gmm1.means_,
                'covariances': self.gmm1.covariances_,
                'weights': self.gmm1.weights_,
                'bic': self.gmm1.bic(self.coords1),
                'aic': self.gmm1.aic(self.coords1)
            },
            'core2': {
                'means': self.gmm2.means_,
                'covariances': self.gmm2.covariances_,
                'weights': self.gmm2.weights_,
                'bic': self.gmm2.bic(self.coords2),
                'aic': self.gmm2.aic(self.coords2)
            }
        }
        
        return self.results['gmm']
    
    def fit_lda(self) -> Dict:
        """
        Fit Linear Discriminant Analysis for optimal linear boundary, using spatial coordinates and local density as features.
        """
        # Combine data
        X_spatial = np.vstack([self.coords1, self.coords2])
        y = np.array([0] * len(self.coords1) + [1] * len(self.coords2))

        # Compute local density for each cell using KernelDensity
        from sklearn.neighbors import KernelDensity
        kde = KernelDensity(bandwidth=20).fit(X_spatial)
        density = np.exp(kde.score_samples(X_spatial)).reshape(-1, 1)

        # Add density as a feature
        X = np.hstack([X_spatial, density])

        # Standardize
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Fit LDA
        self.lda = LinearDiscriminantAnalysis()
        self.lda.fit(X_scaled, y)

        # Get accuracy
        accuracy = self.lda.score(X_scaled, y)

        # Extract line coefficients in original space (for spatial only)
        # Only use the first two coefficients for spatial boundary
        w_scaled = self.lda.coef_[0][:2]
        w_original = w_scaled / self.scaler.scale_[:2]

        # Normalize
        norm = np.sqrt(np.sum(w_original**2))
        a, b = w_original / norm

        # Find intercept: line passes through midpoint of centroids
        centroid1 = np.mean(self.coords1, axis=0)
        centroid2 = np.mean(self.coords2, axis=0)
        midpoint = (centroid1 + centroid2) / 2
        c = -(a * midpoint[0] + b * midpoint[1])

        self.results['lda'] = {
            'a': a,
            'b': b,
            'c': c,
            'accuracy': accuracy,
            'centroid1': centroid1,
            'centroid2': centroid2,
            'midpoint': midpoint
        }

        return self.results['lda']
    
    def compute_posterior(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute posterior probabilities P(core|x,y) using Bayes' theorem.
        
        Parameters
        ----------
        points : np.ndarray
            Points to evaluate (n, 2)
            
        Returns
        -------
        p1, p2 : np.ndarray
            Posterior probabilities for each core
        """
        if not hasattr(self, 'dist1'):
            self.fit_gaussians()
        
        # Compute likelihoods
        likelihood1 = self.dist1.pdf(points)
        likelihood2 = self.dist2.pdf(points)
        
        # Compute posteriors using Bayes' theorem
        # P(core|x) = P(x|core) * P(core) / P(x)
        joint1 = likelihood1 * self.prior1
        joint2 = likelihood2 * self.prior2
        
        # Normalize (avoid division by zero)
        total = joint1 + joint2
        total = np.maximum(total, 1e-300)
        
        p1 = joint1 / total
        p2 = joint2 / total
        
        return p1, p2
    
    def compute_log_likelihood_ratio(self, points: np.ndarray) -> np.ndarray:
        """
        Compute log-likelihood ratio: log(P(x|core1)/P(x|core2)).
        
        Positive values favor core1, negative favor core2.
        Decision boundary is where this equals log(prior2/prior1).
        
        Parameters
        ----------
        points : np.ndarray
            Points to evaluate (n, 2)
            
        Returns
        -------
        llr : np.ndarray
            Log-likelihood ratios
        """
        if not hasattr(self, 'dist1'):
            self.fit_gaussians()
        
        ll1 = self.dist1.logpdf(points)
        ll2 = self.dist2.logpdf(points)
        
        # Log-likelihood ratio
        llr = ll1 - ll2
        
        # Include priors: classify as core1 if llr > log(prior2/prior1)
        prior_ratio = np.log(self.prior2 / self.prior1)
        
        return llr, prior_ratio
    
    def find_decision_boundary(self, n_points: int = 1000) -> np.ndarray:
        """
        Find the optimal decision boundary contour.
        
        The boundary is where P(core1|x,y) = P(core2|x,y) = 0.5.
        
        Parameters
        ----------
        n_points : int
            Number of points to sample for contour estimation
            
        Returns
        -------
        boundary_points : np.ndarray
            Points along the decision boundary
        """
        if not hasattr(self, 'dist1'):
            self.fit_gaussians()
        
        # Create grid
        x_min, x_max = self.coords[:, 0].min() - 10, self.coords[:, 0].max() + 10
        y_min, y_max = self.coords[:, 1].min() - 10, self.coords[:, 1].max() + 10
        
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, 200),
            np.linspace(y_min, y_max, 200)
        )
        grid_points = np.column_stack([xx.ravel(), yy.ravel()])
        
        # Compute posteriors
        p1, p2 = self.compute_posterior(grid_points)
        
        # Decision boundary: |p1 - 0.5| is minimized
        diff = np.abs(p1 - 0.5).reshape(xx.shape)
        
        # Store grid for visualization
        self.grid_x = xx
        self.grid_y = yy
        self.posterior_core1 = p1.reshape(xx.shape)
        self.posterior_core2 = p2.reshape(xx.shape)
        self.posterior_diff = (p1 - p2).reshape(xx.shape)
        
        return diff
    
    def classify_cells(self, method: str = 'bayesian') -> np.ndarray:
        """
        Classify cells to cores using specified method.
        
        Parameters
        ----------
        method : str
            'bayesian' - using posterior probabilities
            'lda' - using LDA decision boundary
            'gmm' - using GMM log-likelihood
            
        Returns
        -------
        predictions : np.ndarray
            Predicted core IDs
        """
        if method == 'bayesian':
            p1, p2 = self.compute_posterior(self.coords)
            predictions = np.where(p1 > p2, self.core1_id, self.core2_id)
            confidence = np.maximum(p1, p2)
            
        elif method == 'lda':
            if not hasattr(self, 'lda'):
                self.fit_lda()
            # Add density as a feature for prediction
            from sklearn.neighbors import KernelDensity
            coords = self.coords
            kde = KernelDensity(bandwidth=20).fit(coords)
            density = np.exp(kde.score_samples(coords)).reshape(-1, 1)
            X = np.hstack([coords, density])
            X_scaled = self.scaler.transform(X)
            pred_labels = self.lda.predict(X_scaled)
            predictions = np.where(pred_labels == 0, self.core1_id, self.core2_id)
            proba = self.lda.predict_proba(X_scaled)
            confidence = np.max(proba, axis=1)
            
        elif method == 'gmm':
            if not hasattr(self, 'gmm1'):
                self.fit_gmm()
            
            # Compare log-likelihoods
            ll1 = self.gmm1.score_samples(self.coords)
            ll2 = self.gmm2.score_samples(self.coords)
            
            # Include priors
            threshold = np.log(self.prior2 / self.prior1)
            predictions = np.where(ll1 - ll2 > threshold, self.core1_id, self.core2_id)
            
            # Confidence from probability
            diff = np.abs(ll1 - ll2)
            confidence = 1 / (1 + np.exp(-diff))
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Calculate accuracy against true labels
        true_labels = self.adata.obs['core_id'].values
        accuracy = np.mean(predictions == true_labels)
        
        self.results[f'{method}_classification'] = {
            'predictions': predictions,
            'confidence': confidence,
            'accuracy': accuracy
        }
        
        return predictions, confidence, accuracy
    
    def compute_separation_metrics(self) -> Dict:
        """
        Compute metrics quantifying how well the cores are separated.
        
        Returns
        -------
        metrics : dict
            Separation metrics including:
            - mahalanobis_distance: Mahalanobis distance between distributions
            - bhattacharyya_distance: Bhattacharyya distance
            - overlap_coefficient: Estimated distribution overlap
        """
        if not hasattr(self, 'dist1'):
            self.fit_gaussians()
        
        g = self.results['gaussian']
        mean1, cov1 = g['core1']['mean'], g['core1']['cov']
        mean2, cov2 = g['core2']['mean'], g['core2']['cov']
        
        # Mahalanobis distance (using pooled covariance)
        pooled_cov = (cov1 + cov2) / 2
        diff = mean1 - mean2
        
        try:
            inv_cov = np.linalg.inv(pooled_cov)
            mahal_dist = np.sqrt(diff @ inv_cov @ diff)
        except np.linalg.LinAlgError:
            mahal_dist = np.nan
        
        # Bhattacharyya distance
        try:
            avg_cov = (cov1 + cov2) / 2
            det1 = np.linalg.det(cov1)
            det2 = np.linalg.det(cov2)
            det_avg = np.linalg.det(avg_cov)
            
            term1 = 0.125 * diff @ np.linalg.inv(avg_cov) @ diff
            term2 = 0.5 * np.log(det_avg / np.sqrt(det1 * det2))
            bhattacharyya_dist = term1 + term2
        except (np.linalg.LinAlgError, ValueError):
            bhattacharyya_dist = np.nan
        
        # Overlap coefficient (approximation via Monte Carlo)
        n_samples = 10000
        samples1 = self.dist1.rvs(size=n_samples)
        samples2 = self.dist2.rvs(size=n_samples)
        
        # Compute PDFs at sampled points
        pdf1_at_1 = self.dist1.pdf(samples1)
        pdf2_at_1 = self.dist2.pdf(samples1)
        pdf1_at_2 = self.dist1.pdf(samples2)
        pdf2_at_2 = self.dist2.pdf(samples2)
        
        # Overlap via minimum of normalized PDFs
        overlap = 0.5 * (np.mean(np.minimum(pdf1_at_1, pdf2_at_1) / (pdf1_at_1 + 1e-300)) +
                        np.mean(np.minimum(pdf1_at_2, pdf2_at_2) / (pdf2_at_2 + 1e-300)))
        
        metrics = {
            'mahalanobis_distance': mahal_dist,
            'bhattacharyya_distance': bhattacharyya_dist,
            'overlap_coefficient': overlap,
            'centroid_distance': np.linalg.norm(mean1 - mean2)
        }
        
        self.results['separation_metrics'] = metrics
        return metrics
    
    def plot_distributions(self, save_path: str = 'core_distributions.png') -> None:
        """
        Plot the fitted Gaussian distributions for both cores.
        """
        if not hasattr(self, 'dist1'):
            self.fit_gaussians()
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        
        # Panel A: Scatter plot with centroids
        ax1 = axes[0, 0]
        ax1.scatter(self.coords1[:, 0], self.coords1[:, 1], 
                   c='#1f77b4', s=8, alpha=0.6, label=f'Core {self.core1_id}')
        ax1.scatter(self.coords2[:, 0], self.coords2[:, 1], 
                   c='#ff7f0e', s=8, alpha=0.6, label=f'Core {self.core2_id}')
        
        # Plot centroids and covariance ellipses
        g = self.results['gaussian']
        for i, (core_data, color) in enumerate([(g['core1'], '#1f77b4'), (g['core2'], '#ff7f0e')]):
            mean = core_data['mean']
            cov = core_data['cov']
            
            # Centroid
            ax1.scatter(mean[0], mean[1], c=color, s=200, marker='*', 
                       edgecolors='white', linewidths=2, zorder=10)
            
            # Covariance ellipse (2 std)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
            for n_std in [1, 2]:
                width, height = 2 * n_std * np.sqrt(eigenvalues)
                ellipse = plt.matplotlib.patches.Ellipse(
                    mean, width, height, angle=angle,
                    fill=False, color=color, linewidth=2, linestyle='--' if n_std == 2 else '-'
                )
                ax1.add_patch(ellipse)
        
        ax1.set_xlabel('X coordinate', fontsize=12)
        ax1.set_ylabel('Y coordinate', fontsize=12)
        ax1.set_title('A: Gaussian Fits (1σ and 2σ ellipses)', fontsize=14, fontweight='bold')
        ax1.legend(loc='best')
        ax1.set_aspect('equal')
        ax1.set_facecolor('#f0f0f0')

        
        # Panel B: Posterior probability heatmap
        ax2 = axes[0, 1]
        self.find_decision_boundary()
        
        # Plot posterior for core 1
        im = ax2.contourf(self.grid_x, self.grid_y, self.posterior_core1, 
                         levels=20, cmap='RdBu', vmin=0, vmax=1)
        plt.colorbar(im, ax=ax2, label=f'P(Core {self.core1_id} | x, y)')
        
        # Add decision boundary (0.5 contour)
        ax2.contour(self.grid_x, self.grid_y, self.posterior_core1, 
                   levels=[0.5], colors='black', linewidths=3)
        
        # Scatter points
        ax2.scatter(self.coords1[:, 0], self.coords1[:, 1], c='blue', s=3, alpha=0.3)
        ax2.scatter(self.coords2[:, 0], self.coords2[:, 1], c='red', s=3, alpha=0.3)
        
        ax2.set_xlabel('X coordinate', fontsize=12)
        ax2.set_ylabel('Y coordinate', fontsize=12)
        ax2.set_title('B: Bayesian Decision Boundary (P=0.5)', fontsize=14, fontweight='bold')
        ax2.set_aspect('equal')

        
        # Panel C: Log-likelihood ratio
        ax3 = axes[1, 0]
        llr, threshold = self.compute_log_likelihood_ratio(
            np.column_stack([self.grid_x.ravel(), self.grid_y.ravel()])
        )
        llr = llr.reshape(self.grid_x.shape)
        
        # Clip for visualization
        llr_clipped = np.clip(llr, -10, 10)
        
        im3 = ax3.contourf(self.grid_x, self.grid_y, llr_clipped, 
                          levels=20, cmap='coolwarm')
        plt.colorbar(im3, ax=ax3, label='Log-likelihood ratio')
        
        # Decision boundary (where llr = threshold)
        ax3.contour(self.grid_x, self.grid_y, llr, 
                   levels=[threshold], colors='black', linewidths=3)
        
        ax3.scatter(self.coords1[:, 0], self.coords1[:, 1], c='blue', s=3, alpha=0.3)
        ax3.scatter(self.coords2[:, 0], self.coords2[:, 1], c='red', s=3, alpha=0.3)
        
        ax3.set_xlabel('X coordinate', fontsize=12)
        ax3.set_ylabel('Y coordinate', fontsize=12)
        ax3.set_title('C: Log-Likelihood Ratio', fontsize=14, fontweight='bold')
        ax3.set_aspect('equal')

        
        # Panel D: LDA boundary comparison
        ax4 = axes[1, 1]
        self.fit_lda()

        ax4.scatter(self.coords1[:, 0], self.coords1[:, 1], 
                   c='#1f77b4', s=8, alpha=0.6, label=f'Core {self.core1_id}')
        ax4.scatter(self.coords2[:, 0], self.coords2[:, 1], 
                   c='#ff7f0e', s=8, alpha=0.6, label=f'Core {self.core2_id}')

        # LDA line
        a, b, c = self.results['lda']['a'], self.results['lda']['b'], self.results['lda']['c']
        # Use same xlim and ylim as Panel A
        xlim = ax1.get_xlim()
        ylim = ax1.get_ylim()
        ax4.set_xlim(xlim)
        ax4.set_ylim(ylim)
        x_range = np.array(xlim)

        if abs(b) > 1e-10:
            y_lda = (-a * x_range - c) / b
            ax4.plot(x_range, y_lda, 'g-', linewidth=3, label='LDA boundary')

        # Bayesian boundary (draw last for visibility)
        contour = ax4.contour(self.grid_x, self.grid_y, self.posterior_core1, 
               levels=[0.5], colors='black', linewidths=3, linestyles='--')
        from matplotlib.lines import Line2D
        handles, labels = ax4.get_legend_handles_labels()
        handles.append(Line2D([0], [0], color='black', linewidth=3, linestyle='--', label='Bayesian boundary'))
        ax4.set_xlabel('X coordinate', fontsize=12)
        ax4.set_ylabel('Y coordinate', fontsize=12)
        ax4.set_title(f'D: LDA vs Bayesian Boundary (LDA acc: {self.results["lda"]["accuracy"]:.1%})', 
                 fontsize=14, fontweight='bold')
        ax4.legend(handles=handles, loc='best')
        ax4.set_aspect('equal')
        ax4.set_facecolor('#f0f0f0')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✓ Saved distribution plot to {save_path}")
    
    def plot_classification_results(self, method: str = 'bayesian',
                                    save_path: str = 'classification_results.png') -> None:
        """
        Plot classification results with confidence scores.
        """
        predictions, confidence, accuracy = self.classify_cells(method)
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Panel A: True labels
        ax1 = axes[0]
        true_labels = self.adata.obs['core_id'].values
        colors_true = ['#1f77b4' if l == self.core1_id else '#ff7f0e' for l in true_labels]
        ax1.scatter(self.coords[:, 0], self.coords[:, 1], c=colors_true, s=8, alpha=0.6)
        ax1.set_title('A: True Core Labels', fontsize=14, fontweight='bold')
        ax1.set_xlabel('X coordinate')
        ax1.set_ylabel('Y coordinate')
        ax1.set_aspect('equal')
        
        # Add legend
        handles = [Patch(color='#1f77b4', label=f'Core {self.core1_id}'),
                  Patch(color='#ff7f0e', label=f'Core {self.core2_id}')]
        ax1.legend(handles=handles, loc='best')
        
        # Panel B: Predicted labels
        ax2 = axes[1]
        colors_pred = ['#1f77b4' if l == self.core1_id else '#ff7f0e' for l in predictions]
        ax2.scatter(self.coords[:, 0], self.coords[:, 1], c=colors_pred, s=8, alpha=0.6)
        ax2.set_title(f'B: Predicted Labels ({method.title()}, Acc: {accuracy:.1%})', 
                     fontsize=14, fontweight='bold')
        ax2.set_xlabel('X coordinate')
        ax2.set_ylabel('Y coordinate')
        ax2.set_aspect('equal')
        ax2.legend(handles=handles, loc='best')
        
        # Panel C: Confidence scores
        ax3 = axes[2]
        
        # Color by correctness
        correct = predictions == true_labels
        colors_conf = ['green' if c else 'red' for c in correct]
        sc = ax3.scatter(self.coords[:, 0], self.coords[:, 1], 
                        c=confidence, s=8, alpha=0.7, cmap='viridis')
        plt.colorbar(sc, ax=ax3, label='Classification Confidence')
        
        # Mark misclassified points
        misclassified = ~correct
        if np.any(misclassified):
            ax3.scatter(self.coords[misclassified, 0], self.coords[misclassified, 1],
                       facecolors='none', edgecolors='red', s=30, linewidths=1.5,
                       label=f'Misclassified (n={np.sum(misclassified)})')
            ax3.legend(loc='best')
        
        ax3.set_title('C: Classification Confidence', fontsize=14, fontweight='bold')
        ax3.set_xlabel('X coordinate')
        ax3.set_ylabel('Y coordinate')
        ax3.set_aspect('equal')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✓ Saved classification results to {save_path}")
        print(f"\n  Accuracy: {accuracy:.2%}")
        print(f"  Misclassified: {np.sum(~correct)} / {len(predictions)} cells")
    
    def get_boundary_equation(self) -> str:
        """
        Return the LDA boundary equation as a string.
        """
        if 'lda' not in self.results:
            self.fit_lda()
        
        a = self.results['lda']['a']
        b = self.results['lda']['b']
        c = self.results['lda']['c']
        
        return f"{a:.6f}x + {b:.6f}y + {c:.6f} = 0"
    
    def summary(self) -> None:
        """
        Print summary of all analyses.
        """
        print("\n" + "="*60)
        print("CORE SEPARATION ANALYSIS SUMMARY")
        print("="*60)
        
        print(f"\nCores: {self.core1_id} and {self.core2_id}")
        print(f"  Core {self.core1_id}: {len(self.coords1)} cells")
        print(f"  Core {self.core2_id}: {len(self.coords2)} cells")
        
        if 'gaussian' in self.results:
            g = self.results['gaussian']
            print(f"\nGaussian Distribution Fits:")
            print(f"  Core {self.core1_id} centroid: ({g['core1']['mean'][0]:.2f}, {g['core1']['mean'][1]:.2f})")
            print(f"  Core {self.core2_id} centroid: ({g['core2']['mean'][0]:.2f}, {g['core2']['mean'][1]:.2f})")
        
        if 'separation_metrics' in self.results:
            m = self.results['separation_metrics']
            print(f"\nSeparation Metrics:")
            print(f"  Centroid distance: {m['centroid_distance']:.2f} units")
            print(f"  Mahalanobis distance: {m['mahalanobis_distance']:.2f}")
            print(f"  Bhattacharyya distance: {m['bhattacharyya_distance']:.4f}")
            print(f"  Overlap coefficient: {m['overlap_coefficient']:.4f}")
        
        if 'lda' in self.results:
            print(f"\nLDA Boundary:")
            print(f"  Equation: {self.get_boundary_equation()}")
            print(f"  Accuracy: {self.results['lda']['accuracy']:.2%}")
        
        print("\n" + "="*60)


def separate_cores(
    adata: anndata.AnnData,
    plot: bool = True,
    save_prefix: str = 'core_separation'
) -> Dict:
    """
    Main function to separate two cores using advanced methods.
    
    Parameters
    ----------
    adata : anndata.AnnData
        AnnData with exactly 2 cores
    plot : bool
        Whether to generate plots
    save_prefix : str
        Prefix for saved plot files
        
    Returns
    -------
    results : dict
        All analysis results
    """
    separator = CoreSeparator(adata)
    
    # Fit distributions
    separator.fit_gaussians()
    separator.fit_gmm()
    separator.fit_lda()
    
    # Compute metrics
    separator.compute_separation_metrics()
    
    # Classify using all methods
    for method in ['bayesian', 'lda', 'gmm']:
        separator.classify_cells(method)
    
    # Summary
    separator.summary()
    
    # Generate plots
    if plot:
        separator.plot_distributions(f'{save_prefix}_distributions.png')
        separator.plot_classification_results('bayesian', f'{save_prefix}_bayesian.png')

    # --- Save separated cores based on LDA boundary ---
    # Classify cells using LDA
    lda_preds, _, _ = separator.classify_cells('lda')
    adata = separator.adata
    core1_id, core2_id = separator.core1_id, separator.core2_id
    mask1 = lda_preds == core1_id
    mask2 = lda_preds == core2_id
    adata_core1 = adata[mask1].copy()
    adata_core2 = adata[mask2].copy()
    adata_core1.write_h5ad(f'{save_prefix}_lda_core1_{core1_id}.h5ad')
    adata_core2.write_h5ad(f'{save_prefix}_lda_core2_{core2_id}.h5ad')
    print(f"✓ Saved LDA-separated core {core1_id} to {save_prefix}_lda_core1_{core1_id}.h5ad")
    print(f"✓ Saved LDA-separated core {core2_id} to {save_prefix}_lda_core2_{core2_id}.h5ad")
    
    return separator


if __name__ == "__main__":
    import sys
    
    print("="*60)
    print("Core Separation using LDA and Gaussian Fitting")
    print("="*60)
    
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        print(f"\nLoading: {file_path}")
        
        try:
            adata = sc.read_h5ad(file_path)
            print(f"Loaded {adata.n_obs} cells")
            
            separator = separate_cores(adata, plot=True, save_prefix='core_separation')
            
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    else:
        print("\nUsage:")
        print("  python LDA.py <path_to_h5ad_with_2_cores>")
        print("\nExample:")
        print("  python LDA.py cores_6_7.h5ad")
        print("\nOr import as module:")
        print("  from LDA import separate_cores")
        print("  separator = separate_cores(adata)")
