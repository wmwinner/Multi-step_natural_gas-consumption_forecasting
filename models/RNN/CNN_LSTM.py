import torch.nn as nn


class CNN_LSTM(nn.Module):
    def __init__(self, configs):
        super(CNN_LSTM, self).__init__()
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

        # hidden_size表示隐藏层状态的进入，这里的输入表示上一个隐藏层的结束，将其状态也传入了下一层
        self.conv1 = nn.Conv2d(in_channels=self.input_size,
                               out_channels=self.hidden_size,
                               kernel_size=self.kernel_size,
                               padding=self.padding,
                               bias=self.bias)
        self.relu1 = nn.ReLU()
        self.maxpooling1 = nn.MaxPool2d(self.kernel_size[0], stride=1, padding=1)

        # LSTM layers
        self.lstm1 = nn.LSTM(input_size=self.hidden_size, hidden_size=self.hidden_size * 2, num_layers=self.num_layers,
                             batch_first=True)
        self.lstm2 = nn.LSTM(input_size=self.hidden_size * 2, hidden_size=self.hidden_size * 4,
                             num_layers=self.num_layers, batch_first=True)

        # Fully connected layers
        self.linear1 = nn.Linear(in_features=self.hidden_size * 4, out_features=self.hidden_size * 2)
        self.dropout1 = nn.Dropout(0.5)
        self.linear2 = nn.Linear(in_features=self.hidden_size * 2, out_features=self.output_size)

    def forward(self, input_tensor):
        input_tensor = input_tensor.reshape([input_tensor.shape[0], 1, self.seq_len, self.input_size])

        input_tensor = self.conv1(input_tensor)
        input_tensor = self.relu1(input_tensor)
        input_tensor = self.maxpooling1(input_tensor)

        input_tensor = input_tensor.reshape([input_tensor.shape[0], self.seq_len, -1])
        out, _ = self.lstm1(input_tensor)
        out, _ = self.lstm2(out)
        out = out[:, -1, :]

        out = self.linear1(out)
        out = self.dropout1(out)
        out = self.linear2(out)
        return out
