import torch
import torch.nn as nn
import torch.nn.functional as F
from .BasicModule import BasicModule


class DPCNN(BasicModule):
    """
    DPCNN for sentences classification.
    """

    def __init__(self, channel_size=250, word_embedding_dimension=300, num_class=1):
        super(DPCNN, self).__init__()
        # self.config = config
        self.channel_size = channel_size
        self.conv_region_embedding = nn.Conv2d(
            1, self.channel_size, (3, word_embedding_dimension), stride=1)
        self.conv3 = nn.Conv2d(self.channel_size, self.channel_size, (3, 1), stride=1)
        self.pooling = nn.MaxPool2d(kernel_size=(3, 1), stride=2)
        self.padding_conv = nn.ZeroPad2d((0, 0, 1, 1))
        self.padding_pool = nn.ZeroPad2d((0, 0, 0, 1))
        self.act_fun = nn.ReLU()
        self.linear_out = nn.Linear(2 * self.channel_size, num_class)

    def forward(self, x):
        batch = x.shape[0]
        x = x.unsqueeze(1)
        # Region embedding input: [batch_size, 1, seq_len, emb_dim]
        x = self.conv_region_embedding(x)  # [batch_size, channel_size, length, 1]

        x = self.padding_conv(x)  # pad保证等长卷积，先通过激活函数再卷积
        x = self.act_fun(x)
        x = self.conv3(x)
        x = self.padding_conv(x)
        x = self.act_fun(x)
        x = self.conv3(x)

        while x.size()[-2] > 2:
            x = self._block(x)

        # print(x.shape)
        x = x.view(batch, 2 * self.channel_size)
        x = self.linear_out(x)

        return x

    def _block(self, x):
        # Pooling
        x = self.padding_pool(x)
        px = self.pooling(x)

        # Convolution
        x = self.padding_conv(px)
        x = F.relu(x)
        x = self.conv3(x)

        x = self.padding_conv(x)
        x = F.relu(x)
        x = self.conv3(x)

        # Short Cut
        x = x + px

        return x

    def predict(self, x):
        self.eval()
        out = self.forward(x)
        predict_labels = torch.max(out, 1)[1]
        self.train(mode=True)
        return predict_labels
