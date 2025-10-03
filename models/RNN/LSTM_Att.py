import torch.nn as nn
import torch
import torch.nn.functional as F


class LSTM_Att(nn.Module):
    def __init__(self, configs):
        super(LSTM_Att, self).__init__()
        self.configs = configs
        self.seq_len = configs.seq_len
        self.output_size = configs.output_size
        self.num_layers = configs.num_layers
        self.input_size = configs.input_size
        self.hidden_size = configs.hidden_size
        self.kernel_size = configs.kernel_size
        self.padding = configs.kernel_size[0] // 2, configs.kernel_size[1] // 2
        self.bias = True
        self.batch_first = True
        self._state_is_tuple = True

        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True)
        self.hidden_size = self.hidden_size
        self.linear = nn.Linear(self.hidden_size, self.output_size)

        # 初始时间步和最终时间步的隐藏状态作为全连接层输入
        self.w_omega = nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size))
        self.u_omega = nn.Parameter(torch.Tensor(self.hidden_size, 1))

        nn.init.uniform_(self.w_omega, -0.1, 0.1)
        nn.init.uniform_(self.u_omega, -0.1, 0.1)


    def attention_net(self, x):
        """
        :param x: [batch_size, seq_len, hidden_size]
        :return:
        """
        u = torch.tanh(torch.matmul(x, self.w_omega))  # [batch, seq_len, hidden_size]
        att = torch.matmul(u, self.u_omega)  # [batch, seq_len, 1]
        att_score = F.softmax(att, dim=1)

        scored_x = x * att_score  # [batch, seq_len, hidden_size]

        context = torch.sum(scored_x, dim=1)  # [batch, hidden_size]
        return context

    def forward(self, x):
        out, (h_n, c_n) = self.lstm(x)
        atten_output = self.attention_net(out)
        out = self.linear(atten_output)
        return out