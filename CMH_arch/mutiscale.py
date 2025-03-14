import torch
from torch import nn
from CMH_arch import FEDBCK
from .mlp import GraphMLP
from .mask import Mask
import os
device = 'cuda'
os.environ['CUDA_LAUNCH_BLOCKING']='1'

class CMH(nn.Module):
    """Spatio-Temporal-Decoupled Masked Pre-training for Traffic Forecasting"""

    def __init__(self, dataset_name, pre_trained_tmae_path,pre_trained_smae_path, mask_args, backend_args):
        super().__init__()
        self.dataset_name = dataset_name
        self.pre_trained_tmae_path = pre_trained_tmae_path
        self.tmae = Mask(**mask_args)
        # load pre-trained model
        self.load_pre_trained_model()
        self.nn1 = nn.Linear(2304, 1024)
        self.nn2 = nn.Linear(1024, 32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.step = 2
        self.if_spatial = True
        self.adj_mx = backend_args["supports"]
        self.num_nodes = backend_args["num_nodes"]
        self.node_hidden = backend_args["node_hidden"]
        self.backend = nn.ModuleList(
                [FEDBCK(first=True, hidden_dim=192, **backend_args)]
        )
        for _ in range(self.step-1):
            self.backend.append(
                FEDBCK(first=False, hidden_dim=192-20, **backend_args)
            )
        self.adj_mx_forward_encoder = nn.Sequential(
            GraphMLP(input_dim=self.num_nodes, hidden_dim=self.node_hidden)
        )
        self.long_linear = nn.Sequential(
            nn.Linear(in_features=288, out_features=12, bias=True),
        )
    def load_pre_trained_model(self):
        """Load pre-trained model"""

        # load parameters
        checkpoint_dict = torch.load(self.pre_trained_tmae_path)
        self.tmae.load_state_dict(checkpoint_dict["model_state_dict"])
        
        # freeze parameters
        for param in self.tmae.parameters():
            param.requires_grad = False
    def forward(self, history_data: torch.Tensor, long_history_data: torch.Tensor, future_data: torch.Tensor, batch_seen: int, epoch: int, **kwargs) -> torch.Tensor:
        """Feed forward of STDMAE.

        Args:
            history_data (torch.Tensor): Short-term historical data. shape: [B, L, N, 3]
            long_history_data (torch.Tensor): Long-term historical data. shape: [B, L * P, N, 3]

        Returns:
            torch.Tensor: prediction with shape [B, N, L].
        """

        # reshape
        short_term_history = history_data

        batch_size, _, num_nodes, _ = history_data.shape

        hidden_states_t = self.tmae(long_history_data[..., [0]])  #long_history_data[..., [0]][32,864,307,1]
        hidden_states_t = hidden_states_t.view(batch_size, num_nodes, -1)

        hidden_states_t = self.nn1(hidden_states_t)
        hidden_states_t = self.relu(hidden_states_t)
        hidden_states_t = self.dropout(hidden_states_t)
        hidden_states_t = self.nn2(hidden_states_t)
        hidden_states_t = hidden_states_t.permute(0, 2, 1).unsqueeze(-1)
        node_forward_emb = []
        node_backward_emb = []
        node_backward_forward_emb = []
        if self.if_spatial:
            node_forward = self.adj_mx[0].to(device)
            node_forward = self.adj_mx_forward_encoder(node_forward.unsqueeze(0)).expand(batch_size, -1, -1)
            node_backward = self.adj_mx[1].to(device)
            node_backward = self.adj_mx_forward_encoder(node_backward.unsqueeze(0)).expand(batch_size, -1, -1)
        node_forward = node_forward.transpose(1, 2).unsqueeze(-1)  # [32,64,307,1]
        node_backward = node_backward.transpose(1, 2).unsqueeze(-1)
        node_backward_forward = node_forward + node_backward
        node_forward_emb.append(node_forward)
        node_backward_emb.append(node_backward)
        node_backward_forward_emb.append(node_backward_forward)
        predicts = []
        hidden_forward_emb = []
        hidden_backward_emb = []
        hidden_backward_forward_emb = []
        for index, layer in enumerate(self.backend):
            y_hat,\
                hidden_forward_list, hidden_backward_list,hidden_backward_forward_list, \
                node_forward_emb, node_backward_emb, node_backward_forward_emb = layer(short_term_history, hidden_states_t,\
                                                                                       node_forward_emb, node_backward_emb, node_backward_forward_emb, \
                                                                                       hidden_forward_emb, hidden_backward_emb,hidden_backward_forward_emb,predicts)
            predicts.append(y_hat)
            hidden_forward_emb = hidden_forward_list
            hidden_backward_emb = hidden_backward_list
            hidden_backward_forward_emb = hidden_backward_forward_list
            node_forward_emb = node_forward_emb
            node_backward_emb = node_backward_emb
            node_backward_forward_emb = node_backward_forward_emb


        return y_hat

