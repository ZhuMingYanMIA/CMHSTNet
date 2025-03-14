import torch
from torch import nn
import torch.nn.functional as F

from .mlp import MultiLayerPerceptron, GraphMLP, MultiLayerPercept

device = 'cuda'
class FEDBCK(nn.Module):
    """
    The implementation of CIKM 2022 short paper
        "Spatial-Temporal Identity: A Simple yet Effective Baseline for Multivariate Time Series Forecasting"
    Link: https://arxiv.org/abs/2208.05233
    """

    def __init__(self, first, hidden_dim, **model_args):
        super().__init__()
        # attributes
        self.num_nodes = model_args["num_nodes"]
        self.node_dim = model_args["node_dim"]
        self.input_len = model_args["input_len"]
        self.input_dim = model_args["input_dim"]
        self.embed_dim = model_args["embed_dim"]
        self.output_len = model_args["output_len"]
        self.num_layer = model_args["num_layer"]
        self.temp_dim_tid = model_args["temp_dim_tid"]
        self.temp_dim_diw = model_args["temp_dim_diw"]
        self.time_of_day_size = model_args["time_of_day_size"]
        self.day_of_week_size = model_args["day_of_week_size"]

        self.if_time_in_day = model_args["if_T_i_D"]
        self.if_day_in_week = model_args["if_D_i_W"]
        self.if_spatial = model_args["if_node"]
        self.first = first
        self.adj_mx = model_args["supports"]
        self.node_hidden = model_args["node_hidden"]
        self.fusion_dim = model_args["fusion_dim"]
        self.nhead = model_args["nhead"]
        if self.if_spatial:
            self.adj_mx_forward_encoder = nn.Sequential(
                GraphMLP(input_dim=self.num_nodes, hidden_dim=self.node_hidden)
            )

        # temporal embeddings
        if self.if_time_in_day:
            self.time_in_day_emb = nn.Parameter(
                torch.empty(self.time_of_day_size, self.temp_dim_tid))
            nn.init.xavier_uniform_(self.time_in_day_emb)
        if self.if_day_in_week:
            self.day_in_week_emb = nn.Parameter(
                torch.empty(self.day_of_week_size, self.temp_dim_diw))
            nn.init.xavier_uniform_(self.day_in_week_emb)

        # embedding layer
        self.time_series_emb_layer = nn.Conv2d(
            in_channels=self.input_dim * self.input_len, out_channels=self.embed_dim, kernel_size=(1, 1), bias=True)

        # encoding
        self.hidden_dim = self.embed_dim+self.node_dim * \
            int(self.if_spatial)+self.temp_dim_tid*int(self.if_time_in_day) + \
            self.temp_dim_diw*int(self.if_day_in_week)
        self.encoder = nn.Sequential(
            *[MultiLayerPerceptron(self.hidden_dim, self.hidden_dim) for _ in range(self.num_layer)])

        self.fusion_graph = nn.Sequential(
            *[MultiLayerPercept(input_dim=hidden_dim,
                                hidden_dim=hidden_dim,
                                )
                for _ in range(self.num_layer)]
        )

        self.fusion_forward_linear = nn.Linear(in_features=hidden_dim, out_features=self.fusion_dim,
                                               bias=True)
        self.fusion_backward_linear = nn.Linear(in_features=hidden_dim, out_features=self.fusion_dim,
                                                bias=True)
        self.forward_att = nn.MultiheadAttention(embed_dim=self.fusion_dim,
                                                 num_heads=self.nhead,
                                                 batch_first=True)
        self.forward_fc = nn.Sequential(
            nn.Linear(in_features=self.fusion_dim, out_features=self.fusion_dim, bias=True),
            nn.Sigmoid()
        )
        self.backward_att = nn.MultiheadAttention(embed_dim=self.fusion_dim,
                                                  num_heads=self.nhead,
                                                  batch_first=True)
        self.backward_fc = nn.Sequential(
            nn.Linear(in_features=self.fusion_dim, out_features=self.fusion_dim, bias=True),
            nn.Sigmoid()
        )

        self.fusion_model = nn.Sequential(
            *[MultiLayerPercept(input_dim=self.hidden_dim,
                                hidden_dim=self.hidden_dim
                                )
              for _ in range(self.num_layer)]
        )

        # regression
        self.regression_layer = nn.Conv2d(
            in_channels=self.hidden_dim, out_channels=self.output_len, kernel_size=(1, 1), bias=True)

    def forward(self, history_data: torch.Tensor,\
                hidden_states_t,node_forward_emb,node_backward_emb,node_backward_forward_emb,\
                hidden_forward_emb, hidden_backward_emb,hidden_backward_forward_emb,predicts) -> torch.Tensor:
        """Feed forward of STID.

        Args:
            history_data (torch.Tensor): history data with shape [B, L, N, C]

        Returns:
            torch.Tensor: prediction wit shape [B, L, N, C]
        """

        # prepare data
    
        input_data = history_data[..., range(self.input_dim)]
        # print(self.adj.shape)

        if self.if_time_in_day:
            if self.if_time_in_day:
                t_i_d_data = history_data[..., 1]
                # In the datasets used in STID, the time_of_day feature is normalized to [0, 1]. We multiply it by 288 to get the index.
                # If you use other datasets, you may need to change this line.
                time_in_day_emb = self.time_in_day_emb[
                    (t_i_d_data[:, -1, :] * self.time_of_day_size).type(torch.LongTensor)]

            # print(time_in_day_emb.shape)
        else:
            time_in_day_emb = None
        if self.if_day_in_week:
            d_i_w_data = history_data[..., 2]
            day_in_week_emb = self.day_in_week_emb[(
                d_i_w_data[:, -1, :]).type(torch.LongTensor)]
        else:
            day_in_week_emb = None

        # time series embedding

        batch_size, _, num_nodes, _ = input_data.shape
        input_data = input_data.transpose(1, 2).contiguous()
        input_data = input_data.view(
            batch_size, num_nodes, -1).transpose(1, 2).unsqueeze(-1)
        time_series_emb1 = self.time_series_emb_layer(input_data)
        time_series_emb = []
        time_series_emb.append(time_series_emb1)

        # temporal embeddings
        tem_emb = []
        if time_in_day_emb is not None:
            tem_emb.append(time_in_day_emb.transpose(1, 2).unsqueeze(-1))
        if day_in_week_emb is not None:
            tem_emb.append(day_in_week_emb.transpose(1, 2).unsqueeze(-1))

        # concate all embeddings

        if not self.first:
            time_series_emb = []
            node_forward_emb = node_forward_emb[0]
            hidden_forward_emb = hidden_forward_emb[0]
            hidden_forward_emb = self.forward_att(hidden_forward_emb, hidden_forward_emb, hidden_forward_emb)[0]
            hidden_forward_emb = self.forward_fc(hidden_forward_emb)
            node_forward_emb = node_forward_emb.squeeze(-1).transpose(1, 2)
            node_forward_emb = [node_forward_emb*hidden_forward_emb]
            node_forward_emb = node_forward_emb[0].unsqueeze(-1).transpose(1, 2)
            node_forward_emb = [node_forward_emb]

            node_backward_emb = node_backward_emb[0]
            hidden_backward_emb = hidden_backward_emb[0]
            hidden_backward_emb = self.backward_att(hidden_backward_emb, hidden_backward_emb, hidden_backward_emb)[0]
            hidden_backward_emb = self.backward_fc(hidden_backward_emb)
            node_backward_emb = node_backward_emb.squeeze(-1).transpose(1, 2)
            node_backward_emb = [node_backward_emb*hidden_backward_emb]
            node_backward_emb = node_backward_emb[0].unsqueeze(-1).transpose(1, 2)
            node_backward_emb = [node_backward_emb]



        hidden_forward_list = []
        hidden_backward_list = []
        hidden_backward_forward_list = []
        forward_emb = torch.cat([hidden_states_t] + time_series_emb + predicts + node_forward_emb + tem_emb, dim=1)
        # print(forward_emb.shape)
        forward_emb = forward_emb.squeeze(-1).transpose(1, 2)
        hidden_forward = self.fusion_graph(forward_emb)
        hidden_forward = self.fusion_forward_linear(hidden_forward)
        hidden_forward_list.append(hidden_forward)

        backward_emb = torch.cat([hidden_states_t] + time_series_emb + predicts + node_backward_emb + tem_emb, dim=1)
        backward_emb = backward_emb.squeeze(-1).transpose(1, 2)
        hidden_backward = self.fusion_graph(backward_emb)
        hidden_backward = self.fusion_backward_linear(hidden_backward)
        hidden_backward_list.append(hidden_backward)
        # fusion_layer
        hidden_total = torch.cat(hidden_forward_list + hidden_backward_list, dim=2)
        hidden = self.fusion_model(hidden_total)
        hidden = hidden.transpose(1, 2).unsqueeze(-1)
        prediction = self.regression_layer(hidden)
        # print(prediction.shape)

        return prediction, hidden_forward_list, hidden_backward_list,hidden_backward_forward_list, node_forward_emb, node_backward_emb, node_backward_forward_emb
