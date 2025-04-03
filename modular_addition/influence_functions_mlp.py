import torch as t
from abc import ABC, abstractmethod
from typing import List
import einops
from modular_addition.model_mlp_ihvp import MLP_IHVP, ExperimentParams
from dataset import make_dataset, train_test_split, make_random_dataset
import os
from pathlib import Path
from random import sample
import numpy as np
import random
from compute_fourier_peakness import find_recursive_ntk_neighbors, compute_empirical_ntk, analyze_jacobian_fourier_peakiness
seed_int = 422
t.manual_seed(seed_int)
np.random.seed(seed_int)
random.seed(seed_int)

device = t.device("cuda" if t.cuda.is_available() else "cpu")

class InfluenceCalculable(ABC):
    @abstractmethod
    def get_a_l_minus_1(self):
        # Return the input to the linear layer
        pass

    @abstractmethod
    def get_d_s_l(self):
        # Return the gradient of the loss wrt the output of the linear layer
        pass

    @abstractmethod
    def get_dims(self):
        # Return the dimensions of the weights - (output_dim, input_dim)
        pass

    @abstractmethod
    def get_d_w_l(self):
        # Return the gradient of the loss wrt the weights
        pass


def get_ekfac_factors_and_pseudo_grads(
    model, dataset, mlp_blocks: List[InfluenceCalculable], device
):
    kfac_input_covs = [
        t.zeros((b.get_dims()[1] + 1, b.get_dims()[1] + 1)).to(device)
        for b in mlp_blocks
    ]
    # print(kfac_input_covs[0].shape)
    kfac_grad_covs = [
        t.zeros((b.get_dims()[0], b.get_dims()[0])).to(device) for b in mlp_blocks
    ]
    
    grads = [[] for _ in range(len(mlp_blocks))]
    tot = 0
    for data, target in dataset:
        model.zero_grad()
        # print(data, target)
        x1 = data[0].to(device)
        x2 = data[1].to(device)
        x1 = x1.unsqueeze(0)
        x2 = x2.unsqueeze(0)
        target = target.to(device)
        target = target.unsqueeze(0)
        #target = t.tensor(target).to(device)
        # if len(data.shape) == 1:
        #     data = data.unsqueeze(0)
        # if len(target.shape) == 0:
        #     target = target.unsqueeze(0)
        output = model(x1, x2)
        loss = t.nn.functional.cross_entropy(output, target)
        for i, block in enumerate(mlp_blocks):
            input_cov = t.einsum(
                "...i,...j->ij", block.get_a_l_minus_1(), block.get_a_l_minus_1()
            )
            kfac_input_covs[i] += input_cov
        loss.backward()
        for i, block in enumerate(mlp_blocks):
            grad_cov = t.einsum("...i,...j->ij", block.get_d_s_l(), block.get_d_s_l())
            kfac_grad_covs[i] += grad_cov
            grads[i].append(block.get_d_w_l())
        tot += 1
    kfac_input_covs = [A / tot for A in kfac_input_covs]
    kfac_grad_covs = [S / tot for S in kfac_grad_covs]
    return kfac_input_covs, kfac_grad_covs, grads


def get_grads(model, dataset, mlp_blocks: List[InfluenceCalculable], device):
    grads = [[] for _ in range(len(mlp_blocks))]
    for data, target in dataset:
        model.zero_grad()
        x1 = data[0].to(device)
        x2 = data[1].to(device)
        x1 = x1.unsqueeze(0)
        x2 = x2.unsqueeze(0)
        target = target.to(device)
        target = target.unsqueeze(0)
        # if len(data.shape) == 1:
        #     data = data.unsqueeze(0)
        # if len(target.shape) == 0:
        #     target = target.unsqueeze(0)
        output = model(x1, x2)
        loss = t.nn.functional.cross_entropy(output, target)
        loss.backward()
        for i, block in enumerate(mlp_blocks):
            grads[i].append(block.get_d_w_l())
    return grads


def compute_lambda_ii(pseudo_grads, q_a, q_s):
    """Compute Lambda_ii values for a block."""
    n_examples = len(pseudo_grads)
    squared_projections_sum = 0.0
    for j in range(n_examples):
        dtheta = pseudo_grads[j]
        result = (q_s @ dtheta @ q_a.T).view(-1)
        squared_projections_sum += result**2
    lambda_ii_avg = squared_projections_sum / n_examples
    return lambda_ii_avg


def get_ekfac_ihvp(
    kfac_input_covs, kfac_grad_covs, pseudo_grads, search_grads, damping=0.001
):
    """Compute EK-FAC inverse Hessian-vector products."""
    ihvp = []
    for i in range(len(search_grads)):
        V = t.stack(search_grads[i])
        # Performing eigendecompositions on the input and gradient covariance matrices
        q_a, _, q_a_t = t.svd(kfac_input_covs[i])
        q_s, _, q_s_t = t.svd(kfac_grad_covs[i])
        lambda_ii = compute_lambda_ii(pseudo_grads[i], q_a, q_s)
        ekfacDiag_damped_inv = 1.0 / (lambda_ii + damping)
        ekfacDiag_damped_inv = ekfacDiag_damped_inv.reshape((V.shape[-2], V.shape[-1]))
        intermediate_result = t.einsum("bij,jk->bik", V, q_a_t)
        intermediate_result = t.einsum("ji,bik->bjk", q_s, intermediate_result)
        result = intermediate_result / ekfacDiag_damped_inv.unsqueeze(0)
        ihvp_component = t.einsum("bij,jk->bik", result, q_a)
        ihvp_component = t.einsum("ji,bik->bjk", q_s_t, ihvp_component)
        # flattening the result except for the batch dimension
        ihvp_component = einops.rearrange(ihvp_component, "b j k -> b (j k)")
        ihvp.append(ihvp_component)
    # Concatenating the results across blocks to get the final ihvp
    return t.cat(ihvp, dim=-1)


def get_query_grad(model, query, mlp_blocks: List[InfluenceCalculable], device):
    grads = get_grads(model, [query], mlp_blocks, device)
    return t.cat([q[0].view(-1) for q in grads])


def get_influences(ihvp, query_grad):
    """
    Compute influences using precomputed iHVP and query_grad
    """
    return -1 * t.einsum("j,ij->i", query_grad, ihvp)


def influence(
    model,
    mlp_blocks: List[InfluenceCalculable],
    queries,
    gradient_fitting_data,
    search_data,
    topk,
    device,
):
    kfac_input_covs, kfac_grad_covs, pseudo_grads = get_ekfac_factors_and_pseudo_grads(
        model, gradient_fitting_data, mlp_blocks, device
    )

    search_grads = get_grads(model, search_data, mlp_blocks, device)

    ihvp = get_ekfac_ihvp(kfac_input_covs, kfac_grad_covs, pseudo_grads, search_grads)
    
    sampled_indices = np.random.choice(len(gradient_fitting_data), 5, replace=False).tolist()
    num_neighbors = 20

    torch_matrix, nd_array, jacobian = compute_empirical_ntk(model, t.nn.CrossEntropyLoss(), train_data)
    circuit_indices_list = []
    for start_idx in sampled_indices:
        # print(f"Starting analysis from input {start_idx}, train_data_idx {train_data[start_idx]}")
        # # Find recursively connected inputs
        circuit_indices = find_recursive_ntk_neighbors(
            jacobian,
            start_idx,
            neighbors_per_step=1,
            total_neighbors=num_neighbors,
            strategy="last"
        )
        circuit_indices_list.append(circuit_indices)
        
    for circuit_indices in circuit_indices_list:
                            
        # Analyze the Jacobian for this circuit
        jacobian_circuit = ihvp[circuit_indices]
        U, S, Vh = t.linalg.svd(jacobian_circuit)
        
        # Find the maximum singular value
        max_singular_value = S[0]
        print("singular values ihvp for circuit", S)
        # Count values less than 10% of the maximum
        threshold = 0.1 * max_singular_value
        small_values_count = (S < threshold).sum().item()
        small_values_percentage = (small_values_count / len(S)) * 100
        
        print(f"Max singular value for ihvp for circuit: {max_singular_value:.4f}")
        print(f"Number of values < 10% of max for ihvp for circuit: {small_values_count}/{len(S)} ({small_values_percentage:.1f}%)")
        # Analyze the Jacobian for this circuit
        jacobian_circuit = jacobian[circuit_indices]
        U, S, Vh = t.linalg.svd(jacobian_circuit)
        
        # Find the maximum singular value
        max_singular_value = S[0]
        print("singular values jacobian for circuit", S)
        # Count values less than 10% of the maximum
        threshold = 0.1 * max_singular_value
        small_values_count = (S < threshold).sum().item()
        small_values_percentage = (small_values_count / len(S)) * 100
        
        print(f"Max singular value for jacobian for circuit: {max_singular_value:.4f}")
        print(f"Number of values < 10% of max for jacobian for circuit: {small_values_count}/{len(S)} ({small_values_percentage:.1f}%)")
        
    for circuit_indices in circuit_indices_list:
        jacobian_circuit = jacobian[circuit_indices]
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
        fourier_results, overall_results = analyze_jacobian_fourier_peakiness(jacobian_circuit, model, top_k_modes=5, module_name="fc1.linear.weight")
        
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
            
        print("overall_results", overall_results)
        print("overall_results keys", overall_results.keys())
            
        print(f" top modes overall: {overall_results['overall']['top_modes']}")
        print(f" peakiness overall: {overall_results['overall']['peakiness']}")
        print(f" concentration overall: {overall_results['overall']['concentration']}")
    
        
    U, S, Vh = t.linalg.svd(ihvp)
            # Find the maximum singular value
    max_singular_value = S[0]
    print("singular values for ihvp", S[:100])
    # Count values less than 10% of the maximum
    threshold = 0.1 * max_singular_value
    small_values_count = (S < threshold).sum().item()
    small_values_percentage = (small_values_count / len(S)) * 100
    
    print(f"Max singular value for ihvp: {max_singular_value:.4f}")
    print(f"Number of values < 10% of max for ihvp: {small_values_count}/{len(S)} ({small_values_percentage:.1f}%)")
    
    U, S, Vh = t.linalg.svd(jacobian)
    # Find the maximum singular value
    max_singular_value = S[0]
    print("singular values for jacobian", S[:100])
    # Count values less than 10% of the maximum
    threshold = 0.1 * max_singular_value
    small_values_count = (S < threshold).sum().item()
    small_values_percentage = (small_values_count / len(S)) * 100
    
    print(f"Max singular value for jacobian: {max_singular_value:.4f}")
    print(f"Number of values < 10% of max for jacobian: {small_values_count}/{len(S)} ({small_values_percentage:.1f}%)")
    
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

    all_top_training_samples = []
    all_top_influences = []

    for query in queries:
        query_grad = get_query_grad(model, query, mlp_blocks, device)
        top_influences = get_influences(ihvp, query_grad)
        top_influences, top_samples = t.topk(top_influences, topk)
        all_top_training_samples.append(top_samples)
        all_top_influences.append(top_influences)
    query_grads = []
    for i, data in enumerate(search_data):
        query_grad = get_query_grad(model, data, mlp_blocks, device)
        query_grads.append(query_grad)
    query_grads = t.stack(query_grads)
    sample_mat = ihvp @ query_grads.T
    print("sample_mat.shape", sample_mat.shape)
    
    sample_mat_numpy = sample_mat.detach().cpu().float().numpy()
    
    eigenvals, eigenvecs = np.linalg.eig(sample_mat_numpy + sample_mat_numpy.T)
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


    return all_top_training_samples, all_top_influences

def dataset_sample(dataset, n_samples):
    indices = sample(range(len(dataset)), n_samples)
    return [dataset[i] for i in indices]


def run_influence(model: MLP_IHVP, train_dataset, test_dataset, n_queries):

    model = model.to(device)
    model.eval()

    queries = dataset_sample(test_dataset, n_queries)
    gradient_fitting_data = dataset_sample(train_dataset, len(train_dataset))
    search_data = dataset_sample(train_dataset, len(train_dataset))

    mlp_blocks = [model.fc1, model.fc2]

    topk = 10
    all_top_training_samples, all_top_influences = influence(
        model, mlp_blocks, queries, gradient_fitting_data, search_data, topk, device
    )

    # for i, (top_samples, top_influences) in enumerate(
    #     zip(all_top_training_samples, all_top_influences)
    # ):
    #     print(f"Query: {queries[i]}")
    #     # print(f"argmax of logits: {model(queries[i][0][0].to(device), queries[i][0][1].to(device))}")
    #     print(f"argmax of logits: {model(queries[i][0][0].to(device), queries[i][0][1].to(device)).argmax()}")
    #     print(f"Top {topk} training samples and their influences:")
    #     for s, i in zip(top_samples, top_influences):
    #         s = s.item()
    #         # print(s)
    #         print(
    #             f"index {s}, {train_dataset[s]} Influence: {i}"
    #         )
            

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
        weight_decay=0,
        run_id=0,
        activation="quad",
        hidden_size=32,
        embed_dim=16,
        train_frac=0.95
    )
    model = MLP_IHVP(params)
    
    model.load_state_dict(t.load(Path(os.path.dirname(__file__)) / "models_mlp/checkpoints/CHECKPOINT_39_P53_frac0.95_hid32_emb16_tieunembedFalse_tielinTrue_freezeFalse_run0.pt"))
    model.to(params.device)
    # print(model.embedding.weight.shape)
    if params.use_random_dataset:
        dataset = make_random_dataset(params.p, params.random_seed)
    else:
        dataset = make_dataset(params.p)
    train_data, test_data = train_test_split(
        dataset, params.train_frac, params.random_seed
    )
    # print(len(train_data))
    run_influence(model=model, train_dataset=train_data, test_dataset=test_data, n_queries=5)