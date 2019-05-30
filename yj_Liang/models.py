import torch.nn as nn

class OnsetConv(nn.Module):
    def __init__(self):
        super(OnsetConv, self).__init__()

        self.multipleConv1 = nn.Conv2d()
        self.multipleConv2 = nn.Conv2d()
        self.multipleConv3 = nn.Conv2d()

        self.concat = nn.Sequential(
            nn.BatchNorm2d(),
            nn.ReLU(),
            nn.MaxPool2d(),
            nn.Dropout(),
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(),
            nn.BatchNorm2d(),
            nn.ReLU(),
            nn.MaxPool2d(),
            nn.Dropout()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(),
            nn.BatchNorm2d(),
            nn.ReLU(),
            nn.MaxPool2d(),
            nn.Dropout()
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(),
            nn.BatchNorm2d(),
            nn.ReLU(),
            nn.MaxPool2d(),
            nn.Dropout()
        )


    def forward(self, mel):
        x = mel.view(mel.size(0), 1, mel.size(1), mel.size(2))


        x1 = self.multipleConv1(x)
        x2 = self.multipleConv2(x)
        x3 = self.multipleConv3(x)



class SegmentConv(nn.Module):
    def __init__(self):
        super(SegmentConv, self).__init__()

    def forward(self, mel):
        pass