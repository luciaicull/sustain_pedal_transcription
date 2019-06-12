import torch.nn as nn
import torch.nn.functional as F
from constants import *
from mel import melspectrogram

class ConvModel(nn.Module):
    def __init__(self, device="cpu"):
        super(ConvModel, self).__init__()

        self.device = device
        self.melspectrogram = melspectrogram.to(device)

        self.MelBatchNorm = nn.BatchNorm2d(1)

        self.pre_conv0 = nn.Conv2d(1, 7, kernel_size=(3, 20))
        self.pre_conv1 = nn.Conv2d(1, 7, kernel_size=(20, 3))
        self.pre_conv2 = nn.Conv2d(1, 7, kernel_size=(3, 3), padding=1)

        self.conv = nn.Sequential(
            nn.BatchNorm2d(21),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),

            nn.Conv2d(21, 21, (3,3), padding=1),
            nn.BatchNorm2d(21),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),

            nn.Conv2d(21, 21, (3, 3), padding=1),
            nn.BatchNorm2d(21),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),

            nn.Conv2d(21, 21, (3, 3), padding=1),
            nn.BatchNorm2d(21),
            nn.ReLU(),
            nn.MaxPool2d(4, 4),
            nn.Dropout(0.25),
        )

        self.fc = nn.Linear(504,1)

    def forward(self, mel):
        # print('mel:' + str(mel))
        # print(mel.shape)
        x0 = self.pre_conv0(F.pad(mel, (9, 10, 1, 1)))
        x1 = self.pre_conv1(F.pad(mel, (1, 1, 9, 10)))
        x2 = self.pre_conv2(mel)
        # print(x0.shape, x1.shape, x2.shape)

        x = torch.cat([x0, x1, x2], dim=1)
        # print(x)
        x = self.conv(x)
        # print(x)
        x = x.view(x.shape[0], -1)
        # print(x)
        x = self.fc(x)
        # print(x)
        x = torch.sigmoid(x)
        # print(x)
        # print('=======')
        return x


def run_on_batch(model, audio, label, loss_reduction="mean", device="cpu"):
    audio = audio.to(device)
    label = label.to(device)
    audio = audio.squeeze(0)
    #print('audio shape')
    #print(audio.shape)

    mel = model.melspectrogram(audio.reshape(-1, audio.shape[-1])[:, :-1]).transpose(-1, -2)
    n_pad = N_STEP - mel.shape[1]
    # print(n_pad)

    mel = mel.unsqueeze(dim=1)
    mel = F.pad(mel, (0, 0, 0, n_pad), mode='replicate')
    # print(mel.shape)

    pred = model(mel)
    #print(pred, label)

    loss = F.binary_cross_entropy(pred, label, reduction=loss_reduction)

    return pred, loss

def run_on_dataset(model, dataset, device="cpu"):
    preds = None
    losss = None 
    for idx, batch in enumerate(dataset):
        with torch.no_grad():
            pred, loss = run_on_batch(model, batch[0], batch[1], loss_reduction="none", device=device) # Does this make a problem with padding?

        # Save memory on the GPU as dataset could be very big
        pred = pred.to("cpu")
        loss = loss.to("cpu")
        if preds is None:
            preds = pred
        else:
            preds = torch.cat((preds, pred), dim=0) # Stack them to create on big batch of tensors
        if losss is None:
            losss = loss
        else:
            losss = torch.cat((losss, loss), dim=0)
    return preds, losss
