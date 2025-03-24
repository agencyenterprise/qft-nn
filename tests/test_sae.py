import torch
from datasets import load_dataset
from transformer_lens import HookedTransformer
from transformer_lens.utils import tokenize_and_concatenate
from sae_lens import SAE

def test_sae_order_persistance():
    model_name="gpt2-small",
    sae_release="gpt2-small-res-jb",
    sae_id="blocks.8.hook_resid_pre",
    dataset_path="NeelNanda/pile-10k",
    dataset_split="train",
    batch_size=32,
    k=20
    
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

    sae.eval()  # prevents error if we're expecting a dead neuron mask for who grads
    with torch.no_grad():
        _, cache = model.run_with_cache(batch_tokens, prepend_bos=True)
        sae_activations =  sae.encode(cache[sae.cfg.hook_name]) # [batch_size ctx_len sae_dim]

    sae_matrix = sae_activations.reshape(-1, sae_activations.shape[2]) # [batch_size*ctx_len sae_dim]
    sae_dependency_graph = (sae_matrix @ sae_matrix.T).numpy()

    assert torch.allclose(sae.encode(cache[sae.cfg.hook_name][0]), sae_activations[0], rtol=1e-5, atol=1e-8)
    assert torch.allclose(sae.encode(cache[sae.cfg.hook_name][7]), sae_activations[7], rtol=1e-5, atol=1e-8)