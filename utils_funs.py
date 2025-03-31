from sklearn.cluster import SpectralClustering
from sklearn.metrics import adjusted_rand_score, silhouette_score
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, confusion_matrix
from scipy.optimize import linear_sum_assignment
import numpy as np
import matplotlib.pyplot as plt
import torch
from typing import Union

def cluster_with_spectral(similarity_matrix, n_clusters=4):
    # Create a fully cleaned copy
    similarity_matrix_copy = similarity_matrix.copy()
    
    # Set diagonal to ones (or zeros)
    # np.fill_diagonal(similarity_matrix_copy, 1.0)
    
    # Replace any NaNs with zeros
    similarity_matrix_copy = np.nan_to_num(similarity_matrix_copy, nan=0.0)
    
    # Ensure matrix is symmetric
    similarity_matrix_copy = 0.5 * (similarity_matrix_copy + similarity_matrix_copy.T)
    
    # Ensure all values are in valid range (0-1 for similarity)
    similarity_matrix_copy = np.clip(similarity_matrix_copy, 0, 1)
    
    # Final check for NaNs
    assert not np.isnan(similarity_matrix_copy).any(), "Matrix still contains NaNs!"
    
    # Now try spectral clustering
    spectral = SpectralClustering(
        n_clusters=n_clusters,
        random_state=42,
        affinity='precomputed',
        assign_labels='discretize'
    )
    cluster_labels = spectral.fit_predict(similarity_matrix_copy)
    
    # Fill diagonal with zeros for silhouette calculation
    np.fill_diagonal(similarity_matrix_copy, 0)
    
    # Convert similarity to distance matrix (ensure non-negative values)
    # Method 1: Convert to distance by taking 1 - normalized similarity
    # First normalize to [0,1] range
    min_val = similarity_matrix_copy.min()
    max_val = similarity_matrix_copy.max()
    normalized = (similarity_matrix_copy - min_val) / (max_val - min_val + 1e-8)
    
    # Make sure diagonal remains zero after normalization
    np.fill_diagonal(normalized, 0)
    
    # Convert to distance (higher similarity = lower distance)
    distance_matrix = 1 - normalized
    
    # Ensure diagonal is still zero in the final distance matrix
    np.fill_diagonal(distance_matrix, 0)
    
    # Calculate silhouette score
    silhouette = silhouette_score(distance_matrix, cluster_labels, metric='precomputed')
    
    return cluster_labels, silhouette

def evaluate_clustering(predicted_clusters, true_labels=None):
    """
    Evaluate clustering results against true labels or expected grouping pattern.
    
    Args:
        predicted_clusters: Array of predicted cluster assignments
        true_labels: Array of true class labels. If None, will use the pattern [0,0,0,0,1,1,1,1,...] based on 4 samples per class
        
    Returns:
        Dictionary with evaluation metrics
    """
    # If no true labels provided, create them based on 4 samples per group pattern
    if true_labels is None:
        # Assume 4 samples per class in order
        return None
    
    # Standard clustering metrics (invariant to label permutation)
    ari = adjusted_rand_score(true_labels, predicted_clusters)
    nmi = normalized_mutual_info_score(true_labels, predicted_clusters)
    
    # Find optimal label mapping for accuracy calculation
    # This handles the case where cluster IDs are arbitrary
    cm = confusion_matrix(true_labels, predicted_clusters)
    row_ind, col_ind = linear_sum_assignment(-cm)  # Maximize overlap
    
    # Create remapped predictions using optimal assignment
    remapped_predictions = np.zeros_like(predicted_clusters)
    for i, j in zip(row_ind, col_ind):
        mask = (predicted_clusters == j)
        remapped_predictions[mask] = i
    
    # Calculate accuracy after optimal mapping
    accuracy = np.sum(remapped_predictions == true_labels) / len(true_labels)
    
    # Create confusion matrix for display
    relabeled_cm = confusion_matrix(true_labels, remapped_predictions)
    
    return {
        "ARI": ari,
        "NMI": nmi,
        "Remapped Accuracy": accuracy,
        "Remapped Predictions": remapped_predictions,
        "Confusion Matrix": relabeled_cm
    }
        


def plot_clustered_matrices(matrices, cluster_assignments, true_labels):
    """
    Three-part visualization:
    1. Each similarity matrix reordered by its own clustering method
    2. A single confusion matrix with metrics for each clustering method
    3. A summary metrics comparison chart
    
    Args:
        matrices: Dictionary of {name: matrix} pairs
        cluster_assignments: Dictionary of {method_name: cluster_labels} pairs
        true_labels: Array of ground truth labels
    """
    # Part 1: Plot each matrix reordered by its own clustering method
    fig, axes = plt.subplots(1, len(matrices), figsize=(6*len(matrices), 5))
    fig.suptitle(f"Similarity Matrices Reordered by Their Respective Clustering Methods", fontsize=16)
    
    if len(matrices) == 1:
        axes = [axes]  # Convert to list for consistent indexing
    
    # Match matrix names with clustering method names
    matrix_to_method = {
        "SAE Similarity": "SAE Spectral",
        "Activation Similarity": "Activation Spectral", 
        "NTK Similarity": "NTK Spectral"
    }
    
    # For each matrix
    for i, (matrix_name, matrix) in enumerate(matrices.items()):
        ax = axes[i]
        
        # Get corresponding clustering method
        method_name = matrix_to_method.get(matrix_name)
        if method_name not in cluster_assignments:
            # Fallback if no exact match
            print(f"Warning: No exact method match for {matrix_name}, using first available clustering")
            method_name = list(cluster_assignments.keys())[0]
            
        clusters = cluster_assignments[method_name]
        
        # Get indices that would sort by cluster assignment
        sorted_indices = np.argsort(clusters)
        
        # Get evaluation metrics
        eval_results = evaluate_clustering(clusters, true_labels)
        accuracy = eval_results['Remapped Accuracy']
        ari = eval_results['ARI']
        nmi = eval_results['NMI']
        
        # Reorder matrix by cluster assignments
        reordered_matrix = matrix[sorted_indices][:, sorted_indices]
        
        # Plot heatmap
        im = ax.imshow(reordered_matrix, cmap='viridis')
        ax.set_title(f"{matrix_name}\nClustered by {method_name}", fontsize=12)
        
        # Add colorbar
        plt.colorbar(im, ax=ax)
        
        # Add cluster boundaries
        cluster_bounds = []
        prev_cluster = clusters[sorted_indices[0]]
        for idx, cluster in enumerate(clusters[sorted_indices]):
            if cluster != prev_cluster:
                cluster_bounds.append(idx)
                prev_cluster = cluster
        
        # Draw lines to show cluster boundaries
        for bound in cluster_bounds:
            ax.axhline(y=bound-0.5, color='r', linestyle='-', linewidth=1)
            ax.axvline(x=bound-0.5, color='r', linestyle='-', linewidth=1)
        
        # Add metric summary under each plot
        ax.text(0.5, -0.15, 
              f"Accuracy: {accuracy:.3f}, ARI: {ari:.3f}, NMI: {nmi:.3f}",
              ha="center", transform=ax.transAxes, fontsize=10, 
              bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    # plt.savefig(f"clustering_results.png")
    plt.show()
    # Part 2: Plot all confusion matrices side by side
    n_methods = len(cluster_assignments)
    fig, axes = plt.subplots(1, n_methods, figsize=(5*n_methods, 5))
    fig.suptitle("Confusion Matrices for Different Clustering Methods", fontsize=16)
    
    if n_methods == 1:
        axes = [axes]  # Convert to list for consistent indexing
    
    for i, (method_name, clusters) in enumerate(cluster_assignments.items()):
        ax = axes[i]
        
        # Get evaluation metrics
        eval_results = evaluate_clustering(clusters, true_labels)
        accuracy = eval_results['Remapped Accuracy']
        ari = eval_results['ARI']
        nmi = eval_results['NMI']
        cm = eval_results['Confusion Matrix']
        
        # Plot confusion matrix
        im = ax.imshow(cm, cmap='Blues')
        
        # Set title with method name and metrics
        title = f"{method_name}\nAcc: {accuracy:.3f}, ARI: {ari:.3f}, NMI: {nmi:.3f}"
        ax.set_title(title, fontsize=12)
        
        # Add labels
        ax.set_xlabel("Predicted Class")
        ax.set_ylabel("True Class")
        
        # Configure ticks
        n_classes = len(np.unique(true_labels))
        ax.set_xticks(np.arange(n_classes))
        ax.set_yticks(np.arange(n_classes))
        ax.set_xticklabels(np.arange(n_classes))
        ax.set_yticklabels(np.arange(n_classes))
        
        # Add text annotations to confusion matrix
        thresh = cm.max() / 2
        for i_cm in range(cm.shape[0]):
            for j_cm in range(cm.shape[1]):
                # Calculate percentage of row total
                row_total = np.sum(cm[i_cm, :])
                percentage = cm[i_cm, j_cm] / row_total * 100 if row_total > 0 else 0
                
                # Set text color based on cell darkness
                color = "white" if cm[i_cm, j_cm] > thresh else "black"
                
                # Show count and percentage
                text = f"{cm[i_cm, j_cm]}\n({percentage:.1f}%)"
                ax.text(j_cm, i_cm, text,
                        ha="center", va="center", color=color, fontsize=10)
                        
        plt.colorbar(im, ax=ax)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
    # plt.savefig(f"confusion_matrices.png")


# --- Utility: Normalize a Matrix ---
def normalize_matrix(M: Union[torch.Tensor, np.ndarray]):
    if isinstance(M, np.ndarray):
        M_tensor = torch.tensor(M, device="cuda" if torch.cuda.is_available() else "cpu")
    else:
        M_tensor = M
    diag = torch.sqrt(torch.diag(M_tensor))
    norm_matrix = M_tensor / (diag.unsqueeze(1) * diag.unsqueeze(0) + 1e-8)
    return norm_matrix.detach().cpu().float().numpy()

def laplacian_normalize(M: Union[torch.Tensor, np.ndarray]):
    if isinstance(M, np.ndarray):
        M_tensor = torch.tensor(M, device="cuda" if torch.cuda.is_available() else "cpu")
    else:
        M_tensor = M
    
    # Ensure symmetry
    M_tensor = 0.5 * (M_tensor + M_tensor.T)
    
    # Compute inverse square root of degree matrix directly
    # (no need to store the full degree matrix since we only need its inverse square root)
    row_sums = torch.sum(M_tensor, dim=1)
    D_sqrt_inv = torch.diag(1.0 / torch.sqrt(row_sums + 1e-8))
    
    # Apply normalized Laplacian formula: S_norm = D^(-1/2) * S * D^(-1/2)
    norm_matrix = D_sqrt_inv @ M_tensor @ D_sqrt_inv
    diag = torch.sqrt(torch.diag(norm_matrix))
    norm_matrix = norm_matrix / (diag.unsqueeze(1) * diag.unsqueeze(0) + 1e-8)

    return norm_matrix.detach().cpu().float().numpy()

def affinity_normalize(M: Union[torch.Tensor, np.ndarray], preference_value=1.0):
    if isinstance(M, np.ndarray):
        M_numpy = M
    else:
        M_numpy = M.detach().cpu().numpy()
    
    # Ensure symmetry
    M_numpy = 0.5 * (M_numpy + M_numpy.T)
    
    # Set diagonal elements (preferences)
    if preference_value is None:
        # Use median similarity as the preference
        preference_value = np.median(M_numpy)
    
    np.fill_diagonal(M_numpy, preference_value)
    
    return M_numpy

def normalize_and_compute_clusters(sae_aggregation_matrix, activation_aggregation_matrix, ntk_matrix, true_labels, test_prompts):
    # Compute the SAE dependency graph (similarity matrix between prompts)
    sae_dependency_graph = (sae_aggregation_matrix @ sae_aggregation_matrix.T).cpu().float().numpy()
    activation_aggregation_dependency_graph = (activation_aggregation_matrix @ activation_aggregation_matrix.T).cpu().float().numpy()
    normalized_sae_graph = laplacian_normalize(sae_dependency_graph)
    normalized_ntk_matrix = laplacian_normalize(ntk_matrix)
    normalized_activation_aggregation_graph = laplacian_normalize(activation_aggregation_dependency_graph)


    # Compute statistics for each matrix
    matrices = {
        "SAE Similarity": normalized_sae_graph,
        "Activation Similarity": normalized_activation_aggregation_graph,
        "NTK Similarity": normalized_ntk_matrix
    }


    # Cluster using spectral clustering on the similarity matrices
    sae_spectral_clusters, sae_spectral_silhouette = cluster_with_spectral(
        normalized_sae_graph, n_clusters=4
    )

    act_spectral_clusters, act_spectral_silhouette = cluster_with_spectral(
        normalized_activation_aggregation_graph, n_clusters=4
    )

    ntk_spectral_clusters, ntk_spectral_silhouette = cluster_with_spectral(
        normalized_ntk_matrix, n_clusters=4
    )

    methods = {
        "SAE Spectral": (sae_spectral_clusters, sae_spectral_silhouette),
        "Activation Spectral": (act_spectral_clusters, act_spectral_silhouette),
        "NTK Spectral": (ntk_spectral_clusters, ntk_spectral_silhouette),
        "Random": (np.random.randint(0, 4, size=len(test_prompts)), 0)
    }
    # After running your clustering code:
    matrices = {
        "SAE Similarity": normalized_sae_graph,
        "Activation Similarity": normalized_activation_aggregation_graph,
        "NTK Similarity": normalized_ntk_matrix
    }

    cluster_assignments = {
        "SAE Spectral": sae_spectral_clusters,
        "Activation Spectral": act_spectral_clusters,
        "NTK Spectral": ntk_spectral_clusters
    }

    plot_clustered_matrices(matrices, cluster_assignments, true_labels)
