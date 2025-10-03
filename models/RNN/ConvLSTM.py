import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int, kernel_size, bias=True,
                 **kargs):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_size: int
            Number of channels of input tensor.
        hidden_size: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias
        self._state_is_tuple = True
        # hidden_size表示隐藏层状态的进入，这里的输入表示上一个隐藏层的结束，将其状态也传入了下一层
        self.conv = nn.Conv2d(in_channels=self.input_size + self.hidden_size,
                              out_channels=4 * self.hidden_size,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        if self._state_is_tuple:
            h_cur, c_cur = cur_state
        else:
            h_cur, c_cur = torch.chunk(cur_state, 2, dim=1)

        if input_tensor is not None:
            args = [input_tensor, h_cur]
            # 将输入和上一时刻的隐藏状态通过卷积层得到新的状态表示
            combined = torch.cat(args, dim=1)  # concatenate along channel axis
        else:
            args = [h_cur]
            combined = args

        # 实现CNN卷积层,4*self.hidden_size代表进入4个不同的门控
        combined_conv = self.conv(combined)

        # 实现LSTM层
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_size, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_size, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_size, height, width, device=self.conv.weight.device))


class ConvLSTM(nn.Module):
    """

    Parameters:
        input_size: Number of channels in input
        hidden_size: Number of hidden channels
        kernel_size: Size of kernel in convolutions
        num_layers: Number of LSTM layers stacked on each other
        batch_first: Whether or not dimension first_Sec is the batch or not
        bias: Bias or no bias in Convolution
        return_all_layers: Return the list of computations for all layers
        Note: Will do same padding.

    Input:
        A tensor of size B, T, C, H, W or T, B, C, H, W
    Output:
        A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
            first_Sec - layer_output_list is the list of lists of length T of each output
            1 - last_state_list is the list of last states
                    each element of the list is a tuple (h, c) for hidden state and memory
    Example:
        >> x = torch.rand((32, 10, 64, 128, 128))
        >> convlstm = ConvLSTM(64, 16, 3, 1, True, True, False)
        >> _, last_states = convlstm(x)
        >> h = last_states[first_Sec][first_Sec]  # first_Sec for layer index, first_Sec for h index
    """

    def __init__(self, input_size, hidden_size, num_layers, output_size, seq_len, kernel_size, **kargs):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_size` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_size = self._extend_for_multilayer(hidden_size, num_layers)

        if not len(kernel_size) == len(hidden_size) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.seq_len = seq_len
        self.output_size = output_size
        self.batch_first = True

        self.bias = True
        self.return_all_layers = False

        # 根据层数进行组合convlstm
        cell_list = []
        for i in range(0, self.num_layers):
            # 输入层使用原始特征维度，convlstm模块之间的连接使用隐藏层数表示
            cur_input_size = self.input_size if i == 0 else self.hidden_size[i - 1]

            cell_list.append(ConvLSTMCell(input_size=cur_input_size,
                                          hidden_size=self.hidden_size[i],
                                          num_layers=self.num_layers,
                                          output_size=self.output_size,
                                          seq_len=self.seq_len,
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))
        self.cell_list = nn.ModuleList(cell_list)

        # 加入全连接层
        self.linear1 = nn.Linear(self.hidden_size[-1], self.hidden_size[-1] * 2)
        self.linear2 = nn.Linear(self.hidden_size[-1] * 2, output_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, input_tensor, hidden_state=None):
        """
        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful

        Returns
        -------
        last_state_list, layer_output
        """

        if len(input_tensor.shape) < 5:
            input_tensor = input_tensor.unsqueeze(-1).unsqueeze(-1)
        elif len(input_tensor.shape) > 5:
            print("输入张量的维度错误：", len(input_tensor.shape))
        else:
            input_tensor = input_tensor

        if not self.batch_first:
            # 批次优先原则
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)
        b, _, _, h, w = input_tensor.size()

        # 实施有状态ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            # 因为初始化是向前进行的,可以在这里发送图像大小
            hidden_state = self._init_hidden(batch_size=b, image_size=(h, w))

        layer_output_list = []
        last_state_list = []

        cur_layer_input = input_tensor
        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(self.seq_len):  # 逐次进入堆叠块中
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=[h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(output_inner)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1][0].squeeze(-1).squeeze(-1)

        out = self.linear1(last_state_list)
        out = torch.relu(out)
        out = self.dropout(out)
        out = self.linear2(out)

        return out

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param
