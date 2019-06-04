import os
from datetime import datetime

import numpy as np
# from tensorboardX import SummaryWriter
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm

import models
from data_loader import *
import torch.nn.functional as F


def cycle(iterable):
    while True:
        for item in iterable:
            yield item


config = dict(
    logdir='runs/pedal-' + datetime.now().strftime('%y%m%d-%H%M%S'),
    device=[0],
    iterations=500000,
    resume_iteration=None,
    checkpoint_interval=1000,
    #model_name='OnsetConv',
    model_name='SegmentConv',

    load_mode='ram',  # 'lazy'
    num_workers=4,

    batch_size=1,
    sequence_length=16000 * 10,
    model_complexity=48,

    learning_rate=0.001,
    learning_rate_decay_steps=10000,
    learning_rate_decay_rate=0.98,

    clip_gradient_norm=3,

    validation_interval=500,

    debug=False
)


def train(logdir, device, model_name, iterations, resume_iteration, checkpoint_interval, load_mode, num_workers,
          batch_size, sequence_length,
          model_complexity, learning_rate, learning_rate_decay_steps, learning_rate_decay_rate, clip_gradient_norm,
          validation_interval, debug):
    default_device = 'cpu' if len(device) == 0 else 'cuda:0'

    os.makedirs(logdir)
    # writer = SummaryWriter(logdir + '/train')
    # valid_writer = SummaryWriter(logdir + '/valid')

    dataset = SegmentExcerptDataset(set='train')

    loader = DataLoader(dataset, batch_size, shuffle=True, num_workers=num_workers)

    validation_dataset = SegmentExcerptDataset(set='test')

    model_class = getattr(models, model_name)
    model = model_class()

    if resume_iteration is None:
        model = model.to(default_device)
        optimizer = torch.optim.Adam(model.parameters(), learning_rate)
        resume_iteration = 0
    else:
        model_state_path = os.path.join(logdir, 'model-{:d}.pt' % resume_iteration)
        checkpoint = torch.load(model_state_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        if len(device) == 1:
            model = model.to(default_device)
        elif len(device) >= 2:
            model = torch.nn.DataParallel(model, device_ids=device).to(default_device)
        optimizer = torch.optim.Adam(model.parameters(), learning_rate)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    scheduler = StepLR(optimizer, step_size=learning_rate_decay_steps, gamma=learning_rate_decay_rate)

    i = 1
    for batch in cycle(loader):

        # print(batch[0])
        optimizer.zero_grad()
        scheduler.step()
        loss = 0
        pred, loss = models.run_on_batch(model, batch[0], batch[1])
        loss.backward()
        if clip_gradient_norm:
            clip_grad_norm_(model.parameters(), clip_gradient_norm)
        # writer.add_scalar('loss', loss, global_step=i)

        optimizer.step()
        print("loss: {:.3f}".format(loss))
        # loop.set_postfix_str("loss: {:.3f}".format(loss))

        if i % validation_interval == 0:
            model.eval()
            with torch.no_grad():
                for batch in validation_dataset:
                    pred, loss = models.run_on_batch(model, batch[0], batch[1])
                    print("valid_loss: {:.3f}".format(loss))
                    # valid_writer.add_scalar('loss', loss, global_step=i)
            model.train()

        if i % checkpoint_interval == 0:
            state_dict = model.module.state_dict() if len(device) >= 2 else model.state_dict()
            torch.save({'model_state_dict': state_dict,
                        'optimizer_state_dict': optimizer.state_dict(),
                        'model_name': model_name},
                       os.path.join(logdir, 'model-{:d}.pt' % i))
        i += 1


if __name__ == '__main__':
    train(**config)

'''
import torch

import data_loader
import models
from constants import *

import random
import numpy as np

class Runner(object):
    def __init__(self):
        self.model = models.OnsetConv()
        # self.model = models.SegmentConv()

        self.criterion = torch.nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.learning_rate = LEARNING_RATE
        self.device = DEFAULT_DEVICE

    def accuracy(self, source, target, mode='train'):
        target = target.cpu()
        correct = (source == target).sum().item()

        return correct / float(source.size(0))

    def run(self, data_loader, mode='train'):
        if mode == 'train':
            #self.model.train()

            epoch_loss = 0
            epoch_acc = 0

            for batch, (excerpt, pedal) in enumerate(data_loader):
                excerpt = excerpt.to(self.device)
                pedal = pedal.to(self.device)

                print(batch)
                print(excerpt)
                print(pedal)

if __name__ == '__main__':
    runner = Runner()

    for epoch in range(NUM_EPOCHS):
        runner.run()
'''