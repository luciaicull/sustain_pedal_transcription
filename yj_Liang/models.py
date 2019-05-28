import torch.nn as nn

class OnsetConv(nn.Module):
    def __init__(self):
        super(OnsetConv, self).__init__()

        self.multipleConv1 = nn.Conv2d()

