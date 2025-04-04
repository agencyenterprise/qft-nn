import torch as t
# from influence_functions_mlp import InfluenceCalculable
from train import ExperimentParams
from abc import ABC, abstractmethod

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


class MLPBlock(InfluenceCalculable, t.nn.Module):
    def __init__(self, input_dim, output_dim, params: ExperimentParams, bias=True, use_activation=True):
        super().__init__()
        self.linear = t.nn.Linear(input_dim, output_dim, bias=bias)
        if params.activation == "relu" and use_activation:
            self.relu = t.nn.ReLU()
        else:
            self.relu = None
        self.input = None
        self.use_relu = params.activation == "relu"
        self.use_quad = params.activation == "quad"
        self.d_s_l = None
        self.d_w_l = None
        self.use_activation = use_activation
        
        # Save gradient of loss wrt output of linear layer (Ds_l, where s_l = self.linear(a_l_minus_1))
        def hook_fn(module, grad_input, grad_output):
            self.d_s_l = grad_output[0]

        self.linear.register_full_backward_hook(hook_fn)

    def forward(self, x):
        if isinstance(x, tuple):
            # print("x[0].shape", x[0].shape)
            # print("x[1].shape", x[1].shape)
            self.input = x[0] + x[1]
            # print("self.input", self.input.shape)
            x1 = self.linear(x[0])
            x2 = self.linear(x[1])
            x = x1 + x2
        else:
            self.input = x
            x = self.linear(x)
        if self.use_activation:
            # self.input = x
            if self.use_relu:
                x = self.relu(x)
            elif self.use_quad:
                x = x ** 2
        return x

    def get_a_l_minus_1(self):
        # Return the input to the linear layer as a homogenous vector
        # print(self.input)
        # print(self.input.shape)
        return (
            t.cat([self.input, t.ones((self.input.shape[0], 1)).to(device)], dim=-1)
            .clone()
            .detach()
        )

    def get_d_s_l(self):
        # Return the gradient of the loss wrt the output of the linear layer
        return self.d_s_l.clone().detach()

    def get_dims(self):
        # Return the dimensions of the weights - (output_dim, input_dim)
        # print("get_dims", self.linear.weight.shape)
        return self.linear.weight.shape

    def get_d_w_l(self):
        # Return the gradient of the loss wrt the weights
        w_grad = self.linear.weight.grad
        if self.linear.bias is not None:
            b_grad = self.linear.bias.grad.unsqueeze(-1)
        else:
            b_grad = t.zeros((self.linear.weight.shape[0], 1), device=device)
        full_grad = t.cat([w_grad, b_grad], dim=-1)
        return full_grad.clone().detach()


class MLP_IHVP(t.nn.Module):
    def __init__(self, params: ExperimentParams):
        super().__init__()
        self.embedding = t.nn.Embedding(params.p, params.embed_dim)
        self.fc1 = MLPBlock(params.embed_dim, params.hidden_size, params, use_activation=True, bias=True)
        self.tie_unembed = params.tie_unembed
        if params.tie_unembed:
            self.fc2 = MLPBlock(params.hidden_size, params.embed_dim, params, use_activation=False, bias=True)
        else:
            self.fc2 = MLPBlock(params.hidden_size, params.p, params, use_activation=False, bias=False)
        

    def forward(self, a, b):
        x1 = self.embedding(a)
        x2 = self.embedding(b)
        # print("x1 after embedding", x1.shape)
        # print("x2 after embedding", x2.shape)
        x = self.fc1((x1, x2))
        x = self.fc2(x)
        if self.tie_unembed:
            x = x @ self.embedding.weight.T
        return x


class MLP(t.nn.Module):
    def __init__(self, params):
        super().__init__()
        self.embedding = t.nn.Embedding(params.p, params.embed_dim)
        self.linear1r = t.nn.Linear(params.embed_dim, params.hidden_size, bias=True)
        self.linear1l = t.nn.Linear(params.embed_dim, params.hidden_size, bias=True)
        self.tie_unembed = params.tie_unembed
        if params.tie_unembed:
            self.linear2 = t.nn.Linear(params.hidden_size, params.embed_dim, bias=True)
        else:
            self.linear2 = t.nn.Linear(params.hidden_size, params.p, bias=False)
        if params.activation == "relu":
            self.act = t.nn.ReLU()
        elif params.activation == "gelu":
            self.act = t.nn.GELU()
        elif params.activation == "quad":
            self.act = lambda x: x ** 2
        else:
            raise ValueError(f"Unknown activation function {params.activation}")
        self.vocab_size = params.p
        self.linear1r.weight.data *= params.scale_linear_1_factor
        self.linear1l.weight.data *= params.scale_linear_1_factor
        self.embedding.weight.data *= params.scale_embed

        self.saved_activations = {}
        self.params = params

    def forward(self, a, b):
        # print(a)
        # print(self.embedding.weight)
        x1 = self.embedding(a)
        x2 = self.embedding(b)
        if self.params.linear_1_tied:
            x1 = self.linear1r(x1)
            x2 = self.linear1r(x2)
        else:
            x1 = self.linear1l(x1)
            x2 = self.linear1r(x2)
        x = x1 + x2
        x = self.act(x)
        x_saved = x
        x = self.linear2(x)

        if self.params.save_activations:
            if len(a.shape) == 0:
                a = a.unsqueeze(0)
                b = b.unsqueeze(0)
            #print(a, b, x_saved.shape)
            self.saved_activations[(a[0], b[0])] = (
                x_saved.clone().detach()
            )  # (batch_size, embed_dim)

        if self.tie_unembed:
            x = x @ self.embedding.weight.T
        return x
