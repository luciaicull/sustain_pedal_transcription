import torch.nn as nn
import torch.nn.functional as F
from constants import *
from mel import melspectrogram
import torchaudio

class OnsetConv(nn.Module):
    def __init__(self):
        super(OnsetConv, self).__init__()

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


class SegmentConv(nn.Module):
    def __init__(self):
        super(SegmentConv, self).__init__()

        self.MelBatchNorm = nn.BatchNorm2d(1)

        self.pre_conv0 = nn.Conv2d(1, 7, kernel_size=(3, 20))
        self.pre_conv1 = nn.Conv2d(1, 7, kernel_size=(20, 3))
        self.pre_conv2 = nn.Conv2d(1, 7, kernel_size=(3, 3), padding=1)

        self.conv = nn.Sequential(
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
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),

            nn.Conv2d(21, 21, (3, 3), padding=1),
            nn.BatchNorm2d(21),
            nn.ReLU(),
            nn.MaxPool2d(4, 4),
            nn.Dropout(0.25),
        )

        self.fc = nn.Linear(504, 1)

    def forward(self, mel):
        #print('mel:' + str(mel))
        #print(mel.shape) # = (1, 1, 200, 128)
        x0 = self.pre_conv0(F.pad(mel, (9, 10, 1, 1)))
        x1 = self.pre_conv1(F.pad(mel, (1, 1, 9, 10)))
        x2 = self.pre_conv2(mel)
        #print(x0.shape, x1.shape, x2.shape) # = ((1, 7, 200, 128), (1, 7, 200, 128), (1, 7, 200, 128))

        x = torch.cat([x0, x1, x2], dim=1)
        #print(x.shape)  # = (1, 21, 200, 128)
        x = self.conv(x)
        #print(x.shape) # = (1, 21, 6, 4)
        x = x.view(x.shape[0], -1)
        # print(x.shape) # = (1, 504)
        x = self.fc(x)
        #print(x.shape) # = (1, 1)
        x = torch.sigmoid(x)
        #print(x.shape) # = (1, 1)
        #print('=======')
        return x


def run_on_batch(model, audio, label, loss_reduction="mean"):
    audio = audio.to('cuda:0')
    label = label.to('cuda:0')
    audio = audio.squeeze(0)
    #print('audio shape')
    #print(audio.shape)

    mel_transform = torchaudio.transforms.MelSpectrogram(
        sr=SAMPLING_RATE, n_fft=N_FFT, ws=WIN_LENGTH, hop=HOP_LENGTH, f_max=float(MEL_FMAX), f_min=float(MEL_FMIN), n_mels=NUM_MELS).cuda()

    mel = None

    if True:
        mel = melspectrogram(audio.reshape(-1, audio.shape[-1])[:, :-1]).transpose(-1, -2)
        n_pad = N_STEP - mel.shape[1]
        # print(n_pad)

        mel = mel.unsqueeze(dim=1)
        mel = F.pad(mel, (0, 0, 0, n_pad), mode='replicate')
        
    else:
        for idx, sample in enumerate(audio):
            sample = sample.unsqueeze(dim=0)

            # print(audio.type())
            print(sample.type())

            transformed_audio = mel_transform(sample)
            # print(mel)
            # print(transformed_audio.shape)
            # print(transformed_audio.type())
            if mel is None:
                mel = transformed_audio
            else:
                mel = torch.cat((mel, transformed_audio), dim=0)
    
    # print(mel.shape)

    pred = model(mel)
    #print(pred, label)

    loss = F.binary_cross_entropy(pred, label, reduction=loss_reduction)

    return pred, loss

def run_on_dataset(model, dataset):
    preds = None
    losss = None 
    for idx, batch in enumerate(dataset):
        with torch.no_grad():
            pred, loss = run_on_batch(model, batch[0], batch[1], loss_reduction="none") # Does this make a problem with padding?

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
