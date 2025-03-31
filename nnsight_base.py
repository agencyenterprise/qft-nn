import torch
from typing import Callable

import nnsight
import gc
import os
from huggingface_hub import hf_hub_download
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import random
import numpy as np
from utils_funs import normalize_and_compute_clusters
# add your Hugging Face token here
os.environ["HF_TOKEN"]=""


device="cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {device}")
print(f"Torch version: {torch.__version__}")
print(f"NNSight version: {nnsight.__version__}")
print("torch.cuda.get_device_name(0):", torch.cuda.get_device_name(0))
print("torch.cuda.is_available():", torch.cuda.is_available())

MODEL_NAME = 'meta-llama/Meta-Llama-3.1-8B-Instruct'
SAE_NAME = 'Llama-3.1-8B-Instruct-SAE-l19'
SAE_LAYER = 'model.layers.19'
EXPANSION_FACTOR = 16 if SAE_NAME == 'Llama-3.1-8B-Instruct-SAE-l19' else 8

dataset = load_dataset("dbpedia_14", split="train", streaming=False)
# Sample 4 examples from each class in AG News
sampled_prompts = []
label_descriptions = {
    4: "Office holder",
    8: "Village",
    10: "Plant",
    11: "Album"
}

# For reproducibility

# Collect examples by label
examples_by_label =  {4: [], 8: [], 10: [], 11: []}
for item in dataset:
    if item['label'] in [4, 8, 10, 11]:
        examples_by_label[item['label']].append(item)

num_samples_per_class = 256
true_labels = []
prompt_info = []  # To store (prompt, label) pairs

# Sample 4 from each label
i = 0
for label in [4, 8, 10, 11]:
    
    sample = random.sample(examples_by_label[label], num_samples_per_class + 0 * i)
    true_labels.extend([i] * len(sample))
    # Store each sample with its label
    for item in sample:
        prompt_info.append((item['content'], i))
    
    i += 1
    for item in sample:
        sampled_prompts.append(item['content'])
        # print(f"Label {label} ({label_descriptions[label]}): {item['content'][:100]}...")

sampled_prompts = []
true_labels = []

# Shuffle the combined data
random.seed(12345)  # Set seed for reproducibility
random.shuffle(prompt_info)

# Unpack the shuffled data
for prompt, label in prompt_info:
    sampled_prompts.append(prompt)
    true_labels.append(label)
    # print(f"Label {label}: length: {len(prompt)}")

print(f"Total samples: {len(sampled_prompts)}")
print(f"Class distribution: {np.bincount(true_labels)}")

# Calculate statistics on prompt lengths
prompt_lengths = [len(prompt) for prompt in sampled_prompts]

# Overall statistics
mean_length = np.mean(prompt_lengths)
std_length = np.std(prompt_lengths)
min_length = np.min(prompt_lengths)
max_length = np.max(prompt_lengths)

print(f"\nPrompt Length Statistics:")
print(f"Mean: {mean_length:.2f} characters")
print(f"Std Dev: {std_length:.2f} characters")
print(f"Min: {min_length} characters")
print(f"Max: {max_length} characters")

# Statistics per class
for class_id in range(len(label_descriptions)):
    class_lengths = [len(sampled_prompts[i]) for i, label in enumerate(true_labels) if label == class_id]
    class_mean = np.mean(class_lengths)
    class_std = np.std(class_lengths)
    label_name = list(label_descriptions.keys())[class_id]
    label_desc = label_descriptions[list(label_descriptions.keys())[class_id]]
    
    print(f"\nClass {class_id} ({label_desc}) Length Statistics:")
    print(f"Mean: {class_mean:.2f} characters")
    print(f"Std Dev: {class_std:.2f} characters")
    print(f"Min: {np.min(class_lengths)} characters")
    print(f"Max: {np.max(class_lengths)} characters")

class SparseAutoEncoder(torch.nn.Module):
    def __init__(
        self,
        d_in: int,
        d_hidden: int,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.d_in = d_in
        self.d_hidden = d_hidden
        self.device = device
        self.encoder_linear = torch.nn.Linear(d_in, d_hidden)
        self.decoder_linear = torch.nn.Linear(d_hidden, d_in)
        self.dtype = dtype
        self.to(self.device, self.dtype)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a batch of data using a linear, followed by a ReLU."""
        return torch.nn.functional.relu(self.encoder_linear(x))

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """Decode a batch of data using a linear."""
        return self.decoder_linear(x)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """SAE forward pass. Returns the reconstruction and the encoded features."""
        f = self.encode(x)
        return self.decode(f), f


def load_sae(
    path: str,
    d_model: int,
    expansion_factor: int,
    device: torch.device = torch.device("cpu"),
):
    sae = SparseAutoEncoder(
        d_model,
        d_model * expansion_factor,
        device,
    )
    sae_dict = torch.load(
        path, weights_only=True, map_location=device
    )
    sae.load_state_dict(sae_dict)

    return sae

InterventionInterface = Callable[[torch.Tensor], torch.Tensor]


class ObservableLanguageModel:
    def __init__(
        self,
        model: str,
        device: str,
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.dtype = dtype
        self.device = device
        self._original_model = model

        self._model = nnsight.LanguageModel(
            self._original_model,
            device_map=device,
            torch_dtype=getattr(torch, dtype) if isinstance(dtype, str) else dtype
        )

        # Quickly run a trace to force model to download due to nnsight lazy download
        input_tokens = self._model.tokenizer.apply_chat_template([{"role": "user", "content": "hello"}])
        with self._model.trace(input_tokens):
          pass

        self.tokenizer = self._model.tokenizer

        self.d_model = self._attempt_to_infer_hidden_layer_dimensions()

        self.safe_mode = False  # Nnsight validation is disabled by default, slows down inference a lot. Turn on to debug.

        self._model.to(self.device)

    def _attempt_to_infer_hidden_layer_dimensions(self):
        config = self._model.config
        if hasattr(config, "hidden_size"):
            return int(config.hidden_size)

        raise Exception(
            "Could not infer hidden number of layer dimensions from model config"
        )

    def _find_module(self, hook_point: str):
        submodules = hook_point.split(".")
        module = self._model
        while submodules:
            module = getattr(module, submodules.pop(0))
        return module
    
    
# model = ObservableLanguageModel(
#     model=MODEL_NAME,
#     device=device,
#     dtype=torch.bfloat16,
# )

# input_tokens = model.tokenizer.apply_chat_template(
#     [
#         {"role": "user", "content": "Hello, how are you?"},
#     ],
#     add_generation_prompt=True,
#     return_tensors="pt",
# ).to(model.device)
# with model._model.trace(input_tokens) as tracer:
#     module = model._find_module(SAE_LAYER)
#     feature_cache = module.output[0].save()

# print(feature_cache.shape)


file_path = hf_hub_download(
    repo_id=f"Goodfire/{SAE_NAME}",
    filename=f"{SAE_NAME}.pth",
    repo_type="model"
)

print("file_path", file_path)

# sae = load_sae(
#     file_path,
#     d_model=model.d_model,
#     expansion_factor=EXPANSION_FACTOR,
#     device=model.device,
# )

# features = sae.encode(feature_cache)
# print("features.shape", features.shape)

# with model._model.trace(
#     input_tokens,
# ):
#     # Target tokens are the input tokens shifted right 
#     target_tokens_loss = input_tokens[:, 1:]  # All tokens except the first one
    
#     # Zero gradients before computation
#     model._model.zero_grad()
    
#     # Forward pass to get logits
#     logits = model._model.output.logits
#     logits = logits[:, :-1, :]
    


    
#     # Compute loss - sum the cross-entropy loss for all tokens
#     loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
#     loss = loss_fn(logits.reshape(-1, logits.size(-1)), target_tokens_loss.reshape(-1))
#     loss.backward()
    
#     # # Save gradients for all parameters
#     param_grads = {}
#     # for name, param in model._model.named_parameters():
#     #     if param.requires_grad and param.grad is not None:
#     #         # Save each parameter's gradient
#     #         param_grads[name] = param.grad.save()
#     module_grad = module.output[0].grad.save()

# # Print information about saved parameter gradients
# print(f"Collected gradients for {len(param_grads)} parameters")
# if param_grads:
#     # Print a few examples of parameter gradients
#     for i, (name, grad) in enumerate(list(param_grads.items())[:3]):  # Show first 3 as examples
#         print(f"Parameter: {name}, Gradient shape: {grad.shape}")
        
        
# print(module_grad.shape)
# print(1)

# del model
# gc.collect()
# torch.cuda.empty_cache()
# print("starting to sleep")
# time.sleep(20)
# print("done sleeping")

# huggingface_model = AutoModelForCausalLM.from_pretrained(
#     MODEL_NAME, 
#     device_map="auto",
#     torch_dtype=torch.bfloat16
# )

def get_gradients_with_huggingface(model, model_name, prompts, layer_name=None):
    """
    Get gradients of a model and organize them into a Jacobian matrix efficiently.
    
    Args:
        model: The HuggingFace model
        model_name: Name of the HuggingFace model
        prompts: List of text prompts to use
        layer_name: Optional layer to specifically extract gradients from
        
    Returns:
        Jacobian matrix of shape [n_samples, n_params]
    """
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Get the model parameters we care about, filtering by layer if specified
    
    if layer_name:
        parameters = []
        parameter_names = []
        for layer in layer_name:

            for name, param in model.named_parameters():
                if layer in name and param.requires_grad:
                    parameters.append(param)
                    parameter_names.append(name)
            # print(f"Found {len(parameters)} parameters in layer {layer}")
    else:
        # WARNING: Using all parameters for large models will likely exceed memory
        parameters = [p for p in model.parameters() if p.requires_grad]
        parameter_names = [name for name, p in model.named_parameters() if p.requires_grad]
    # print("parameters", parameters)
    # print("parameter_names", parameter_names)

    n_params = sum(p.numel() for p in parameters)
    n_samples = len(prompts)
    
    # # print(f"Creating Jacobian of shape [{n_samples}, {n_params}] = {n_samples * n_params} elements")
    # if n_params > 1_000_000_000:
    #     print("WARNING: Very large parameter count detected!")
    #     print("Consider using layer_name to filter parameters.")
        
    #     # Layer filtering is highly recommended for large models
    #     if not layer_name:
    #         # print("Listing available top-level modules:")
    #         for name, _ in model.named_children():
    #             print(f"  - {name}")
    #         raise ValueError("Parameter count too large. Please specify a layer_name to filter.")
    
    # Initialize Jacobian matrix - one row per prompt, one column per parameter
    jacobian = torch.zeros(n_samples, n_params, device=model.device, dtype=torch.bfloat16)
    
    # Process each prompt one at a time to build the Jacobian
    for i, prompt in enumerate(prompts):
        # Tokenize input
        inputs = tokenizer.encode(
            prompt,
            return_tensors="pt"
            ).to(model.device)

        # Zero gradients
        model.zero_grad()
        
        # Target tokens are the input tokens shifted right
        input_tokens = inputs[:, :-1]
        target_tokens = inputs[:, 1:]
        # print("input_tokens", input_tokens)
        # print("target_tokens", target_tokens)
        # Zero gradients
        model.zero_grad()
        
        # Forward pass
        logits = model(input_tokens).logits
        # Compute loss
        loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
        loss = loss_fn(logits.reshape(-1, logits.size(-1)), target_tokens.reshape(-1))        
        # Backward pass
        loss.backward()
        
        # Extract gradients directly into the Jacobian matrix
        idx = 0
        for param in parameters:
            if param.grad is not None:
                # Flatten the gradient and add it to the Jacobian
                grad_flatten = param.grad.view(-1)
                jacobian[i, idx:idx + grad_flatten.numel()] = grad_flatten.float()
                idx += grad_flatten.numel()
            
            # Clear gradient to save memory
            param.grad = None
            
    return jacobian, parameter_names

# Example usage - using layer filtering for large models
# Try with layer filtering - focusing only on a specific layer
# layer_names = []
# for i in range(11, 20):
#     layer_names.append(f"model.layers.{i}")
# print("layer_names", layer_names)
    
# full_ntk_matrix = torch.zeros(len(sampled_prompts), len(sampled_prompts), device=huggingface_model.device, dtype=torch.bfloat16)
    
# for i, prompt in enumerate(sampled_prompts):
#     for j, prompt2 in enumerate(sampled_prompts):
#         print(f"Computing NTK for {i} and {j}")
#         if i > j:
#             jacobian, param_names = get_gradients_with_huggingface(
#                 model=huggingface_model,
#                 model_name=MODEL_NAME,
#                 prompts=[prompt, prompt2],
#                 layer_name=layer_names
#             )

#             print(f"Jacobian shape: {jacobian.shape}")

#             # Compute Neural Tangent Kernel (NTK) for the filtered parameters
#             ntk_matrix = jacobian @ jacobian.T
#             print(f"NTK matrix shape: {ntk_matrix.shape}")
#             print(f"NTK matrix:\n{ntk_matrix}")
            
#             full_ntk_matrix[i, i] = ntk_matrix[0, 0]
#             full_ntk_matrix[i, j] = ntk_matrix[0, 1]
#             full_ntk_matrix[j, i] = ntk_matrix[1, 0]
#             full_ntk_matrix[j, j] = ntk_matrix[1, 1]
#             print("full_ntk_matrix", full_ntk_matrix)



# print(ntk_matrix)
# print(full_ntk_matrix)
# for i in range(len(sampled_prompts)):
#     print(f"row {i}: {full_ntk_matrix[i]}")
    
# del jacobian
# del huggingface_model
# print(1)

if __name__ == "__main__":
    model = ObservableLanguageModel(
        model=MODEL_NAME,
        device=device,
        dtype=torch.bfloat16,
    )


    sae = load_sae(
        file_path,
        d_model=model.d_model,
        expansion_factor=EXPANSION_FACTOR,
        device=model.device,
    )
    # Initialize dicts for raw activations and features
    model_activations_matrix = {}
    model_features_matrix = {}
    # Initialize matrices for aggregated activations and features
    sae_aggregation_matrix = torch.zeros(len(sampled_prompts), model.d_model * EXPANSION_FACTOR, device=device, dtype=torch.bfloat16)
    activation_aggregation_matrix = torch.zeros(len(sampled_prompts), model.d_model, device=device, dtype=torch.bfloat16)
    for i, prompt in enumerate(sampled_prompts):
        print(f"Processing prompt {i} of {len(sampled_prompts)}")
        input_tokens = model.tokenizer.encode(
            prompt,
            return_tensors="pt",
        ).to(model.device)
        with model._model.trace(input_tokens) as tracer:
            module = model._find_module(SAE_LAYER)
            module_activations = module.output[0].save()
            module_features = sae.encode(module_activations).save()
        
        model_activations_matrix[i] = module_activations.squeeze(0).detach()
        model_features_matrix[i] = module_features.squeeze(0).detach()
        
    for i, prompt in enumerate(sampled_prompts):
        # sae_aggregation_matrix[i] = model_features_matrix[i].amax(dim=0)
        # activation_aggregation_matrix[i] = model_activations_matrix[i].amax(dim=0)
        sae_aggregation_matrix[i] = torch.nn.functional.normalize(model_features_matrix[i], p=2, dim=0).mean(dim=0)
        activation_aggregation_matrix[i] = torch.nn.functional.normalize(model_activations_matrix[i], p=2, dim=0).mean(dim=0)
        
        mat = (module_features / torch.linalg.norm(sae.decoder_linear.weight.T.detach(), dim=-1, ord=2)).nan_to_num().squeeze(0).detach()
    ntk_matrix = torch.rand(len(sampled_prompts), len(sampled_prompts), device=device, dtype=torch.bfloat16)
    for i in range(len(sampled_prompts)):
        ntk_matrix[i, i] = 1
        
    normalize_and_compute_clusters(sae_aggregation_matrix, activation_aggregation_matrix, ntk_matrix, true_labels, sampled_prompts)