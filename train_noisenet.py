import numpy as np
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
from torch.optim.lr_scheduler import StepLR
from torch.optim import lr_scheduler
from tqdm import tqdm

from models import NoiseNet
from utils.DataUtils.NoiseDataset import GenImageDataset
from utils.DataUtils.CommonTools import weights_init_kaiming
import functools


def main():
    device = "cuda:3"
    torch.random.manual_seed(42)

    model = NoiseNet(is_train=True).to(device)
    model.apply(functools.partial(weights_init_kaiming, scale=0.001))

    data_set = GenImageDataset(
        # 'data_new/test/clean',
        'data_new/train/clean',
        phase='train',
        crop_size=128
    )

    train_loader = torch.utils.data.DataLoader(
        data_set,
        batch_size=1, shuffle=True,
        num_workers=1, pin_memory=True, drop_last=False)

    lr = 1e-4

    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr, weight_decay=0.9)
    # scheduler = StepLR(optimizer, step_size=150, gamma=.1)
    scheduler = lr_scheduler.MultiStepLR(optimizer,milestones=[100, 200, 300],gamma=0.1) 

    epochs = 1000

    model.train()

    for e in tqdm(range(epochs)):
        e_loss = 0
        acc = 0
        acc_2 = 0

        print('lr: {:.2e}'.format(optimizer.param_groups[0]['lr']))
        for n_img, trg in tqdm(train_loader):
            trg = trg.unsqueeze(1).to(device).float()

            optimizer.zero_grad()
            out = model(n_img.to(device))

            loss = criterion(out, trg)
            loss.backward()
            optimizer.step()

            e_loss += loss.item()

            acc += (np.abs(out.detach().cpu().numpy() -
                    trg.detach().cpu().numpy()) < 1).sum()
            acc_2 += (np.abs(out.detach().cpu().numpy() -
                      trg.detach().cpu().numpy()) <= 2).sum()

        print([float("{:.1f}".format(x))
              for x in out.detach().cpu().numpy().flatten()][:5])
        print(trg.detach().cpu().numpy().T[0][:5])

        print("Epoch: {}\t Loss: {:.3f}\t acc: {:.3f} \t acc_2: {:.3f} ".format(
            e,
            e_loss / len(data_set),
            acc / len(data_set),
            acc_2 / len(data_set)),
            end='')

        scheduler.step()
        torch.save(model.state_dict(), 'checkpoints/NoiseNet/latest.pth')


if __name__ == '__main__':
    main()
