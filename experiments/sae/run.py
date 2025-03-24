from typing import Dict, List, Tuple
import torch
from torch.utils.data import TensorDataset
from datasets import load_dataset
from transformer_lens import HookedTransformer
from transformer_lens.utils import tokenize_and_concatenate
from sae_lens import SAE
from jaxtyping import Float
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from qft_nn.ntk import compute_empirical_ntk

def _get_top_k_connections(dependency_matrix: Float[np.ndarray, "num_data num_data"], k: int=10) -> Dict[int, Tuple[int, List[int]]]:
    n_nodes = dependency_matrix.shape[0]
    assert k <= n_nodes - 1, "k must be smaller than n_nodes - 1"

    result = {}    
    for i in range(n_nodes):
        connections = dependency_matrix[i]
        node_connections = [(j, connections[j]) for j in range(n_nodes) if j != i]
        sorted_connections = sorted(node_connections, key=lambda x: x[1], reverse=True)        
        result[i] = sorted_connections[:k]
    return result

def _plot_heatmap(matrix: Float[np.ndarray, "num_data num_data"], title: str, figsize: int=(10,8), cmap: str = "Blues", annot: bool=True, fmt=".2f"):
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(matrix, annot=annot, fmt=fmt, cmap=cmap, ax=ax)
    ax.set_title(title)
    
    if len(matrix.shape) == 2:
        ax.set_xlabel("Column Index")
        ax.set_ylabel("Row Index")
    
    plt.tight_layout()
    plt.show()

def _run(
    model_name: str,
    sae_release: str,
    sae_id: str,
    dataset_path: str,
    dataset_split: str,
    batch_size: int,
    k: int
    ):
    if torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = HookedTransformer.from_pretrained(model_name=model_name, device=device)

    sae, cfg_dict, sparsity = SAE.from_pretrained(
        release=sae_release,
        sae_id=sae_id,
        device=device,
    )

    dataset = load_dataset(
        path=dataset_path,
        split=dataset_split,
        streaming=False,
    )

    token_dataset = tokenize_and_concatenate(
        dataset=dataset,  # type: ignore
        tokenizer=model.tokenizer,  # type: ignore
        streaming=True,
        max_length=sae.cfg.context_size,
        add_bos_token=sae.cfg.prepend_bos,
    )

    batch_tokens = token_dataset[:batch_size]["tokens"]
    input_tokens = batch_tokens[:, :-1]
    label_tokens = batch_tokens[:, 1:]
    dataset = TensorDataset(input_tokens, label_tokens)

    sae.eval()  # prevents error if we're expecting a dead neuron mask for who grads
    with torch.no_grad():
        _, cache = model.run_with_cache(input_tokens, prepend_bos=True)
        sae_activations =  sae.encode(cache[sae.cfg.hook_name]) # [batch_size ctx_len sae_dim]

    sae_matrix = sae_activations.reshape(-1, sae_activations.shape[2]) # [batch_size*ctx_len sae_dim]
    sae_dependency_graph = (sae_matrix @ sae_matrix.T).numpy()
    
    entk = compute_empirical_ntk(model=model, criterion=torch.nn.functional.cross_entropy, data=dataset, llm=True)

    assert sae_dependency_graph.shape == entk.shape

    _plot_heatmap(matrix=entk, title="Empirical NTK")
    _plot_heatmap(matrix=sae_dependency_graph, title="SAE Dependency Graph")    

    # Compare top k
    sae_top_k_connections = _get_top_k_connections(dependency_matrix=sae_dependency_graph, k=k)
    entk_top_k_connections =  _get_top_k_connections(dependency_matrix=entk, k=k)

    num_of_common_datapoints = []
    for i in range(0, k):
        entk_top_k_features_for_data_i = [entk_top_k_connections[i][j][0] for j in range(0, k)]
        sae_top_k_features_for_data_i = [sae_top_k_connections[i][j][0] for j in range(0, k)]
        
        set1 = set(entk_top_k_features_for_data_i)
        set2 = set(sae_top_k_features_for_data_i)
        common_elements = set1.intersection(set2)
    
        num_of_common_datapoints.append(len(common_elements))

    # Plot Distribution of ratios
    print(f"Average num of common_datapoints: {np.mean(num_of_common_datapoints)}")


if __name__ == "__main__":
    _run(
        model_name="gpt2-small",
        sae_release="gpt2-small-res-jb",
        sae_id="blocks.8.hook_resid_pre",
        dataset_path="NeelNanda/pile-10k",
        dataset_split="train",
        batch_size=32,
        k=20
    )


    





    
    