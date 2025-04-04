import torch as t
# from tqdm import tqdm
# import random
from model import MLP
from dataset import make_dataset, train_test_split, make_random_dataset
# from influence_functions_mlp import get_influences, get_query_grad
import os
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
import numpy as np
from jaxtyping import Float
from typing import Optional, List, Tuple, Union
import random
# from dynamics import (
#     ablate_other_modes_fourier_basis,
#     ablate_other_modes_embed_basis,
#     get_magnitude_modes,
# )
# from model_viz import viz_weights_modes, plot_mode_ablations, plot_magnitudes
# from movie import run_movie_cmd
# from dataclasses import dataclass, asdict
# import json
from helpers import eval_model
# from typing import Optional
# import os 
from train import ExperimentParams, test

seed_int = 422
t.manual_seed(seed_int)
np.random.seed(seed_int)
random.seed(seed_int)



def _find_param_name(
    model: t.nn.Module, parameter: t.nn.Parameter
) -> Optional[str]:
    for name, p in model.named_parameters():
        if p is parameter:
            return name
    return None

def compute_empirical_ntk(
    model: t.nn.Module,
    criterion: t.nn.modules.loss._Loss,
    data: List[Tuple[Tuple[t.Tensor, t.Tensor], t.Tensor]],
) -> Float[np.ndarray, "num_data num_data"]:
    # Prepare data
    # if batch_size is None:
    #     dataloader: DataLoader = DataLoader(data, batch_size=len(data))  # type: ignore
    #     X, y = next(iter(dataloader))
    # else:
    #     dataloader = DataLoader(data, batch_size=batch_size)
    #     X_batches, y_batches = [], []
    #     for X_batch, y_batch in dataloader:
    #         X_batches.append(X_batch)
    #         y_batches.append(y_batch)

    #     X = t.cat(X_batches, dim=0)
    #     y = t.cat(y_batches, dim=0)
    # Initialize Jacobian
    # n_samples = X.shape[0]
    n_samples = len(data)
    parameters = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    n_params = sum(p[1].numel() for p in parameters)
    device = t.device("cuda" if t.cuda.is_available() else "cpu")
    jacobian = t.zeros(n_samples, n_params, device=device)

    model.train()
    
    for i in range(n_samples):
        X_1 = t.stack([data[i][0][0]]).to(device)
        X_2 = t.stack([data[i][0][1]]).to(device)
        Y = t.stack([data[i][1]]).to(device)
        model.zero_grad()
        y_pred = model(X_1, X_2)
        loss = criterion(y_pred, Y)
        # loss = criterion(y_pred, y_i)
        loss.backward(retain_graph=True) # Hotfix. Prob should pivot towards using torch.autograd.grad as opposed to backward() in the long run

        # Extract gradients
        idx = 0
        for name, param in parameters:
            if param.grad is not None:
                grad_flatten = param.grad.flatten()
                jacobian[i, idx:idx+len(grad_flatten)] = grad_flatten
                idx += len(grad_flatten)
            else:
                if i == 0:
                    print(f"Warning: Parameter '{name}' has None gradient. it's not used in forward pass")
                idx += param.numel()
    # ENTK: K(x, x') = J(x) · J(x')^T
    entk_matrix = jacobian @ jacobian.T
    
    return entk_matrix, entk_matrix.detach().cpu().float().numpy(), jacobian


def find_recursive_ntk_neighbors(
    ntk_matrix: t.Tensor,
    start_idx: int, 
    neighbors_per_step: int = 5,
    total_neighbors: int = 20,
    strategy: str = "average"
) -> List[int]:
    """
    Recursively find neighbors with high NTK similarity.
    
    Args:
        ntk_matrix: The NTK matrix (n_samples × n_samples)
        start_idx: Index of the starting input
        neighbors_per_step: Number of new neighbors to add at each step
        total_neighbors: Maximum total neighbors to find
        strategy: How to select new neighbors - "average" (avg similarity to all selected) 
                  or "last" (similarity to most recently added)
    
    Returns:
        List of indices representing the selected inputs
    """
    device = ntk_matrix.device
    selected_indices = [start_idx]
    all_indices = set(range(ntk_matrix.shape[0]))
    remaining_indices = all_indices - set(selected_indices)
    
    while len(selected_indices) < total_neighbors and remaining_indices:
        # Compute similarities based on the chosen strategy
        if strategy == "average":
            # Average similarity to all already selected points
            similarities = t.mean(
                ntk_matrix[list(remaining_indices)][:, selected_indices], 
                dim=1
            )
        elif strategy == "last":
            # Similarity to the most recently added points
            recent_k = min(3, len(selected_indices))  # Use last k points
            recent_indices = selected_indices[-recent_k:]
            similarities = t.mean(
                ntk_matrix[list(remaining_indices)][:, recent_indices], 
                dim=1
            )
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        # Get top-k indices from the remaining set
        k = min(neighbors_per_step, total_neighbors - len(selected_indices))
        if k <= 0:
            break
            
        # Convert remaining_indices to list and get their positions for topk
        remaining_list = list(remaining_indices)
        _, top_positions = t.topk(similarities, k)
        
        # Convert positions back to original indices
        new_indices = [remaining_list[pos.item()] for pos in top_positions]
        
        # Add new indices to selected and remove from remaining
        selected_indices.extend(new_indices)
        remaining_indices -= set(new_indices)
    
    return selected_indices


def analyze_jacobian_fourier_peakiness(
    jacobian_circuit: t.Tensor,
    model: t.nn.Module,
    top_k_modes: int = 5,
    module_name: str = "embedding"
):
    """
    Analyze the Fourier peakiness of the Jacobian for a circuit.
    
    Args:
        jacobian_circuit: Jacobian matrix [num_examples, num_weights]
        model: The neural network model (to get parameter shapes)
        top_k_modes: Number of top modes to report
    """
    # Get parameter information
    parameters = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    
    # Average the gradients across all examples to get circuit direction
    avg_gradient = jacobian_circuit.mean(dim=0)  # Shape: [num_weights]
    
    # Process each parameter group
    idx = 0
    results = {}
    overall_results = {}

    for name, param in parameters:
        # Extract gradients for this parameter
        param_size = param.numel()
        param_grad = avg_gradient[idx:idx+param_size].reshape(param.shape)
        idx += param_size
        
        # linear weights
        if module_name in name.lower():
            # For 2D tensors like embedding tables or linear weights
            if len(param.shape) == 2:
                for i in range(param.shape[0]):
                    # Get the row vector
                    row_grad = param_grad[i].detach().cpu().numpy()
                    
                    # Apply FFT
                    fft_result = np.fft.fft(row_grad)
                    fft_magnitudes = np.abs(fft_result)
                    
                    # Normalize to get percentage of energy
                    total_energy = np.sum(fft_magnitudes)
                    if total_energy > 0:
                        normalized = fft_magnitudes / total_energy
                        
                        # Find top modes
                        top_indices = np.argsort(normalized)[::-1][:top_k_modes]
                        
                        # Calculate peakiness - ratio of top mode to mean of others
                        top_value = normalized[top_indices[0]]
                        others = np.delete(normalized, top_indices[0])
                        mean_others = np.mean(others) if len(others) > 0 else 0
                        peakiness = top_value / mean_others if mean_others > 0 else float('inf')
                        
                        results[f"{name}[{i}]"] = {
                            'top_modes': [(int(idx), float(normalized[idx])) for idx in top_indices],
                            'peakiness': float(peakiness),
                            'concentration': float(top_value)
                        }
                row_grad = param_grad.mean(dim=0).detach().cpu().numpy()
                    # Apply FFT
                fft_result = np.fft.fft(row_grad)
                fft_magnitudes = np.abs(fft_result)
                
                # Normalize to get percentage of energy
                total_energy = np.sum(fft_magnitudes)
                if total_energy > 0:
                    normalized = fft_magnitudes / total_energy
                    
                    # Find top modes
                    top_indices = np.argsort(normalized)[::-1][:top_k_modes]
                    
                    # Calculate peakiness - ratio of top mode to mean of others
                    top_value = normalized[top_indices[0]]
                    others = np.delete(normalized, top_indices[0])
                    mean_others = np.mean(others) if len(others) > 0 else 0
                    peakiness = top_value / mean_others if mean_others > 0 else float('inf')
                    
                    overall_results[f"overall"] = {
                        'top_modes': [(int(idx), float(normalized[idx])) for idx in top_indices],
                        'peakiness': float(peakiness),
                        'concentration': float(top_value)
                    }
    return results, overall_results

def compute_eigenvalues(matrix: Union[t.Tensor, np.ndarray], device: str = "cuda"):
    if isinstance(matrix, t.Tensor):
        matrix = matrix.detach().cpu().float().numpy()
    eigenvals, eigenvecs = np.linalg.eig(matrix)
    real_eigenvals = np.real(eigenvals)
    sorted_indices = np.argsort(real_eigenvals)[::-1]  # [::-1] reverses for descending order
    sorted_eigenvals = real_eigenvals[sorted_indices]
    

    total_sum = np.sum(sorted_eigenvals)
    normalized_eigenvals = sorted_eigenvals / total_sum
    participation_ratio = 1.0 / np.sum(normalized_eigenvals**2)
    
    # 2. Explained variance ratio
    # What percentage of variance is explained by top k eigenvalues
    k = 10  # For example, top 5 eigenvalues
    explained_variance_ratio = np.sum(sorted_eigenvals[:k]) / np.sum(sorted_eigenvals)
    
    # 3. Decay rate
    # How quickly eigenvalues decay relative to the largest eigenvalue
    if len(sorted_eigenvals) > 1:
        decay_rates = sorted_eigenvals[1:] / sorted_eigenvals[:-1]
    
    #print(nd_array)
    #print("Top 50 eigenvalues:", sorted_eigenvals[:50])
    # print(nd_array)
    
        # Use only positive eigenvalues for analysis
    positive_eigenvals = sorted_eigenvals[sorted_eigenvals > 0]
    if len(positive_eigenvals) == 0:
        positive_eigenvals = sorted_eigenvals  # Fallback if no positive eigenvalues
    
    # Normalize eigenvalues to create a probability distribution
    total = np.sum(positive_eigenvals)
    normalized_eigenvals = positive_eigenvals / total
    
    # Calculate entropy
    # Using the Shannon entropy formula: -∑(p_i * log(p_i))
    # Avoid log(0) by filtering out zeros
    nonzero_probs = normalized_eigenvals[normalized_eigenvals > 0]
    entropy = -np.sum(nonzero_probs * np.log(nonzero_probs))
    
    # Maximum possible entropy for this number of eigenvalues would be log(n)
    max_entropy = np.log(len(positive_eigenvals))
    
    # Effective number of eigenvalues (based on entropy)
    # This exponentiates the entropy to give a number that represents
    # how many equally-weighted eigenvalues would give the same entropy
    effective_dim = np.exp(entropy)
    
    #print(f"Entropy: {entropy:.4f}")
    #print(f"Effective Dimension: {effective_dim:.2f} (out of {len(train_data)} eigenvalues)")
    return sorted_eigenvals, effective_dim


if __name__ == "__main__":
    params = ExperimentParams(
        linear_1_tied=True,
        tie_unembed=False,
        movie=True,
        scale_linear_1_factor=1.0,
        scale_embed=1.0,
        use_random_dataset=False,
        freeze_middle=False,
        n_batches=2000,
        n_save_model_checkpoints=40,
        lr=0.005,
        magnitude=False,
        ablation_fourier=False,
        do_viz_weights_modes=False,
        batch_size=128,
        num_no_weight_decay_steps=0,
        save_activations=True,
        weight_decay=0,
        run_id=0,
        activation="quad",
        hidden_size=32,
        embed_dim=16,
        train_frac=0.95
    )
    # p_values = [7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43]
    p_values  = [53]
    # p_sweep_exp(p_values, params, "example2")
    model = MLP(params)
    
    model.load_state_dict(t.load(Path(os.path.dirname(__file__)) / "models/checkpoints/CHECKPOINT_39_P53_frac0.95_hid32_emb16_tieunembedFalse_tielinTrue_freezeFalse_run0.pt"))
    model.to(params.device)
    # print(model.embedding.weight.shape)
    if params.use_random_dataset:
        dataset = make_random_dataset(params.p, params.random_seed)
    else:
        dataset = make_dataset(params.p)
    train_data, test_data = train_test_split(
        dataset, params.train_frac, params.random_seed
    )
    # print(eval_model(model, test_data, params.device))
    # print(eval_model(model, train_data, params.device))
    # print(test(model, test_data, params.device))
    # print(test(model, train_data, params.device))
    model_activation = MLP(params)
    model_activation.load_state_dict(t.load(Path(os.path.dirname(__file__)) / "models/checkpoints/CHECKPOINT_39_P53_frac0.95_hid32_emb16_tieunembedFalse_tielinTrue_freezeFalse_run0.pt"))
    model_activation.to(params.device)
    for i in train_data:
        model_activation.forward(i[0][0].to(params.device), i[0][1].to(params.device))
    # print(model_activation.saved_activations)

    torch_matrix, nd_array, jacobian = compute_empirical_ntk(model, t.nn.CrossEntropyLoss(), train_data)
    sorted_eigenvals, effective_dim = compute_eigenvalues(nd_array + nd_array.T)
    print(sorted_eigenvals)
    print(effective_dim)
    eigenvals, eigenvecs = np.linalg.eig(nd_array + nd_array.T)
    # Sort eigenvalues (and corresponding eigenvectors) by real part in descending order
    real_eigenvals = np.real(eigenvals)
    sorted_indices = np.argsort(real_eigenvals)[::-1]  # [::-1] reverses for descending order
    sorted_eigenvals = real_eigenvals[sorted_indices]
    

    total_sum = np.sum(sorted_eigenvals)
    normalized_eigenvals = sorted_eigenvals / total_sum
    participation_ratio = 1.0 / np.sum(normalized_eigenvals**2)
    
    # 2. Explained variance ratio
    # What percentage of variance is explained by top k eigenvalues
    k = 10  # For example, top 5 eigenvalues
    explained_variance_ratio = np.sum(sorted_eigenvals[:k]) / np.sum(sorted_eigenvals)
    
    # 3. Decay rate
    # How quickly eigenvalues decay relative to the largest eigenvalue
    if len(sorted_eigenvals) > 1:
        decay_rates = sorted_eigenvals[1:] / sorted_eigenvals[:-1]
    
    #print(nd_array)
    print("Top 50 eigenvalues:", sorted_eigenvals[:50])
    # print(nd_array)
    
        # Use only positive eigenvalues for analysis
    positive_eigenvals = sorted_eigenvals[sorted_eigenvals > 0]
    if len(positive_eigenvals) == 0:
        positive_eigenvals = sorted_eigenvals  # Fallback if no positive eigenvalues
    
    # Normalize eigenvalues to create a probability distribution
    total = np.sum(positive_eigenvals)
    normalized_eigenvals = positive_eigenvals / total
    
    # Calculate entropy
    # Using the Shannon entropy formula: -∑(p_i * log(p_i))
    # Avoid log(0) by filtering out zeros
    nonzero_probs = normalized_eigenvals[normalized_eigenvals > 0]
    entropy = -np.sum(nonzero_probs * np.log(nonzero_probs))
    
    # Maximum possible entropy for this number of eigenvalues would be log(n)
    max_entropy = np.log(len(positive_eigenvals))
    
    # Effective number of eigenvalues (based on entropy)
    # This exponentiates the entropy to give a number that represents
    # how many equally-weighted eigenvalues would give the same entropy
    effective_dim = np.exp(entropy)
    
    print(f"Entropy: {entropy:.4f}")
    print(f"Effective Dimension: {effective_dim:.2f} (out of {len(train_data)} eigenvalues)")
    
    sampled_indices = np.random.choice(torch_matrix.shape[0], 5, replace=False).tolist()
    num_neighbors = 20
    # for i in sampled_indices:
    #     # print(t.topk(torch_matrix[i],5),i, torch_matrix[i][i])
    #     datas_to_consider = t.topk(torch_matrix[i],num_neighbors).indices.tolist()
    #     if int(i) not in datas_to_consider:
    #         datas_to_consider.pop()
    #         datas_to_consider.append(int(i))
            
    #     jacobian_to_consider = jacobian[datas_to_consider]
    #     U, S, Vh = t.linalg.svd(jacobian_to_consider)
        
    #     # Find the maximum singular value
    #     max_singular_value = S[0]  
        
    #     # Count values less than 10% of the maximum
    #     threshold = 0.01 * max_singular_value
    #     small_values_count = (S < threshold).sum().item()
        
    #     # Calculate percentage of small singular values
    #     small_values_percentage = (small_values_count / len(S)) * 100
    #     # print(f"Sample {i}:")
    #     # # print(f"Singular values: {S}")
    #     # print(f"Max singular value: {max_singular_value:.4f}")
    #     # print(f"Number of values < 1% of max: {small_values_count}/{len(S)} ({small_values_percentage:.1f}%)")
    #     # print("-" * 50)

    #     # Count values less than 10% of the maximum
    #     threshold = 0.1 * max_singular_value
    #     small_values_count = (S < threshold).sum().item()
        
    #     # Calculate percentage of small singular values
    #     small_values_percentage = (small_values_count / len(S)) * 100
        
    #     print(f"Sample {i}:")
    #     # print(f"Singular values: {S}")
    #     print(f"Max singular value: {max_singular_value:.4f}")
    #     print(f"Number of values < 10% of max: {small_values_count}/{len(S)} ({small_values_percentage:.1f}%)")
    #     print("-" * 50)
        
    # Instead of random sampling, choose a few starting points
    #starting_indices = np.random.choice(torch_matrix.shape[0], 5, replace=False)
    list_of_indices = []
    for start_idx in sampled_indices:
        print(f"Starting analysis from input {start_idx}, train_data_idx {train_data[start_idx]}")
        # Find recursively connected inputs
        circuit_indices = find_recursive_ntk_neighbors(
            torch_matrix, 
            start_idx,
            neighbors_per_step=3,
            total_neighbors=num_neighbors,
            strategy="last"
        )
        print(circuit_indices)
        list_of_indices.append(circuit_indices)
        # Analyze the Jacobian for this circuit
        jacobian_circuit = jacobian[circuit_indices]
        U, S, Vh = t.linalg.svd(jacobian_circuit)
        
        # Find the maximum singular value
        max_singular_value = S[0]
        
        # Count values less than 10% of the maximum
        threshold = 0.1 * max_singular_value
        small_values_count = (S < threshold).sum().item()
        small_values_percentage = (small_values_count / len(S)) * 100
        
        print(f"Max singular value: {max_singular_value:.4f}")
        print(f"Number of values < 10% of max: {small_values_count}/{len(S)} ({small_values_percentage:.1f}%)")
        
        # Optional: Analyze the top weight directions (right singular vectors)
        top_weight_directions = Vh[:3]  # Top 3 directions in weight space
        
        # TODO: You could further analyze these directions, e.g.:
        # 1. Identify which weights have the largest components
        # 2. Check if these align with specific Fourier modes
        # 3. Visualize the pattern of weights being used
        
        print("-" * 50)
        
        # Analyze Fourier peakiness of the circuit's gradients
        fourier_results, overall_results = analyze_jacobian_fourier_peakiness(jacobian_circuit, model, top_k_modes=5, module_name="embedding")
        
        # Print the most peaked parameters
        print(f"Fourier analysis for circuit starting at {start_idx}:")
        
        # Sort results by peakiness
        sorted_results = sorted(
            [(name, data) for name, data in fourier_results.items()],
            key=lambda x: x[1]['peakiness'],
            reverse=True
        )
        
        # Print top 5 most peaked parameters
        for name, data in sorted_results[:5]:
            print(f"  {name}:")
            print(f"    Top modes: {data['top_modes']}")
            print(f"    Peakiness: {data['peakiness']:.2f}")
            print(f"    Concentration: {data['concentration']:.2%}")
            
        print(f" top modes overall: {overall_results['overall']['top_modes']}")
        print(f" peakiness overall: {overall_results['overall']['peakiness']}")
        print(f" concentration overall: {overall_results['overall']['concentration']}")
        # Analyze Fourier peakiness of the circuit's gradients
        fourier_results, overall_results = analyze_jacobian_fourier_peakiness(jacobian_circuit, model, top_k_modes=5, module_name="linear1")
        
        # Print the most peaked parameters
        print(f"Fourier analysis for circuit starting at {start_idx}:")
        
        # Sort results by peakiness
        sorted_results = sorted(
            [(name, data) for name, data in fourier_results.items()],
            key=lambda x: x[1]['peakiness'],
            reverse=True
        )
        
        # Print top 5 most peaked parameters
        for name, data in sorted_results[:5]:
            print(f"  {name}:")
            print(f"    Top modes: {data['top_modes']}")
            print(f"    Peakiness: {data['peakiness']:.2f}")
            print(f"    Concentration: {data['concentration']:.2%}")
            
        print(f" top modes overall: {overall_results['overall']['top_modes']}")
        print(f" peakiness overall: {overall_results['overall']['peakiness']}")
        print(f" concentration overall: {overall_results['overall']['concentration']}")
    
        print("-" * 50)
        
    print(f"Total number of samples: {len(train_data)}")
    
    for i in list_of_indices:
        print(i)
    print(list_of_indices)