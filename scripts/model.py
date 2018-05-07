import torch.nn as nn
import torch.nn.functional as F

# Convolutional → RELU → MaxPooling → Convolutional → RELU → MaxPooling → Linear → SoftMax
class ConvModel(nn.Module):

    def __init__(self, imgSize, nbClass):
        super(ConvModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 20, (5,5))
        # conv1 : img 3 * 32 * 32 -> img 20 * 28 * 28
        self.maxpool1 = nn.MaxPool2d((2,2), stride=(2,2))
        # maxpool1 : img 20 * 28 * 28 -> img 20 * 14 * 14
        self.conv2 = nn.Conv2d(20, 50, (5,5))
        # conv2 : img 20 * 14 * 14 -> img 50 * 10 * 10
        self.maxpool2 = nn.MaxPool2d((2,2), stride=(2,2))
        # maxpool2 : img 50 * 10 * 10 -> 50 * 5 * 5

        self.linear1_dim = int((((imgSize - 4) / 2 - 4) / 2) ** 2 * 50)
        self.linear1 = nn.Linear(self.linear1_dim, nbClass)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, input):
        out = self.conv1(input)
        out = F.relu(out)
        out = self.maxpool1(out)

        out = self.conv2(out)
        out = F.relu(out)
        out = self.maxpool2(out)

        out = out.view(-1, self.linear1_dim)
        out = self.linear1(out)
        out = self.log_softmax(out)
        return out
