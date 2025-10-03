import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, seq_len, hidden_size, dropout):
        super(ResBlock, self).__init__()
        self.temporal = nn.Sequential(
            nn.Linear(seq_len, hidden_size), nn.BatchNorm1d(hidden_size), nn.LeakyReLU(),
            nn.Linear(hidden_size, seq_len), nn.Dropout(dropout)
        )
        self.channel = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.GELU(), nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size), nn.Dropout(dropout)
        )
        self.res_scale = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, x):
        temporal_out = self.temporal(x.transpose(1, 2)).transpose(1, 2)
        channel_out = self.channel(x)
        x = x + self.res_scale * temporal_out
        x = x + self.res_scale * channel_out
        return x


class HATL_Seq2Seq(nn.Module):
    def __init__(self, configs):
        super(HATL_Seq2Seq, self).__init__()
        self.configs = configs

        self.task_type = configs.task_type
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.hidden_size = configs.d_model
        self.input_dim = configs.enc_in
        self.output_dim = configs.c_out

        self.input_projection = nn.Linear(self.input_dim, self.hidden_size)
        self.model_backbone = nn.ModuleList(
            [ResBlock(self.seq_len, self.hidden_size, configs.dropout) for _ in range(configs.e_layers)]
        )
        self.lstm = nn.LSTM(input_size=self.hidden_size, hidden_size=self.hidden_size, num_layers=1, batch_first=True)
        self.multihead_attention = nn.MultiheadAttention(embed_dim=self.hidden_size, num_heads=configs.n_heads)
        self.conv_attention = nn.Conv1d(in_channels=self.hidden_size, out_channels=self.hidden_size, kernel_size=3,
                                        padding=1)
        self.time_attention = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size), nn.Tanh(), nn.Linear(self.hidden_size, 1, bias=False)
        )
        self.positional_encoding = nn.Parameter(torch.zeros(1, self.seq_len, self.hidden_size))
        nn.init.xavier_uniform_(self.positional_encoding)


        self.forecast_projection = nn.Linear(self.seq_len, self.pred_len)
        self.dimension_reduction = nn.Linear(self.hidden_size, self.output_dim)


    def mixed_attention(self, x):
        B, L, D = x.size()
        x = x + self.positional_encoding[:, :L, :]
        x_transposed = x.permute(1, 0, 2)
        attn_output, _ = self.multihead_attention(x_transposed, x_transposed, x_transposed)
        attn_output = attn_output.permute(1, 0, 2)
        conv_input = x.permute(0, 2, 1)
        conv_output = self.conv_attention(conv_input).permute(0, 2, 1)
        x = attn_output + conv_output
        time_attn_weights = F.softmax(self.time_attention(x).squeeze(-1), dim=-1).unsqueeze(-1)
        x = x * time_attn_weights
        return x

    def encode(self, x_enc):
        x_enc = self.input_projection(x_enc)
        x_enc = self.mixed_attention(x_enc)
        for layer in self.model_backbone:
            x_enc = layer(x_enc)
        x_enc, _ = self.lstm(x_enc)
        return x_enc

    def decode(self, x_enc_out):
        output = self.forecast_projection(x_enc_out.transpose(1, 2)).transpose(1, 2)
        output = self.dimension_reduction(output)
        return output

    def forecast(self, x_enc):
        out = self.encode(x_enc)
        output = self.decode(out)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):

        dec_out = self.forecast(x_enc)
        return dec_out[:, -self.pred_len:, :]

