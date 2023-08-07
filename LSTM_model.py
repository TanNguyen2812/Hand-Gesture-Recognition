import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import time
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import top_k_accuracy_score as top_k_acc
from torch.nn.functional import softmax
import matplotlib.pyplot as plt


class HandGesture(nn.Module):
    def __init__(self, input_dim, n_layers, hidden_dim,  num_classes):
        super().__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.bn = nn.BatchNorm1d(num_features=input_dim)
        self.lstm = nn.LSTM(input_dim, self.hidden_dim, self.n_layers, batch_first=True, dropout=0.5)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.n_layers, x.size(0), self.hidden_dim).to('cuda')
        c0 = torch.zeros(self.n_layers, x.size(0), self.hidden_dim).to('cuda')
        x = x.permute(0, 2, 1)
        x = self.bn(x)
        x = x.permute(0, 2, 1)
        out, (ht, ct) = self.lstm(x, (h0, c0))
        out = self.fc(ht[-1])
        return out


def UniformlySample(num_frames, clip_len):
    """Uniformly sample indices for training clips.
    Args:
        num_frames (int): The number of frames.
        clip_len (int): The length of the clip.
    """
    num_clips = 1
    p_interval = (1,1)
    allinds = []
    for clip_idx in range(num_clips):
        old_num_frames = num_frames
        pi = p_interval
        ratio = np.random.rand() * (pi[1] - pi[0]) + pi[0]
        num_frames = int(ratio * num_frames)
        off = np.random.randint(old_num_frames - num_frames + 1)

        if num_frames < clip_len:
            start = np.random.randint(0, num_frames)
            inds = np.arange(start, start + clip_len)
        elif clip_len <= num_frames < 2 * clip_len:
            basic = np.arange(clip_len)
            inds = np.random.choice(
                clip_len + 1, num_frames - clip_len, replace=False)
            offset = np.zeros(clip_len + 1, dtype=np.int64)
            offset[inds] = 1
            offset = np.cumsum(offset)
            inds = basic + offset[:-1]
        else:
            bids = np.array(
                [i * num_frames // clip_len for i in range(clip_len + 1)])
            bsize = np.diff(bids)
            bst = bids[:clip_len]
            offset = np.random.randint(bsize)
            inds = bst + offset
        inds = inds + off
        num_frames = old_num_frames
        allinds.append(inds)
    return np.concatenate(allinds)


class Hand_Dataset(Dataset):
    def __init__(self, path, seq_len):
        self.data_path = path
        self.ano = pd.read_pickle(path)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.ano)

    def __getitem__(self, idx):
        kp = self.ano[idx]['keypoints']
        label = self.ano[idx]['label']
        video_len = len(kp)
        inds = UniformlySample(video_len, self.seq_len)
        kp = kp[inds]
        return {'keypoints': kp, 'label':label}

def train_batch(model, data, optimizer, loss_fcn):
    model.train()
    kp = data['keypoints'].float()
    kp = kp.to('cuda')
    label = data['label']
    label = label.to('cuda')
    output = model(kp)
    loss = loss_fcn(output, label)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

def val_batch(model, data, loss_fcn):
    model.eval()
    kp = data['keypoints'].float()
    kp = kp.to('cuda')
    label = data['label']
    label = label.to('cuda')
    output = model(kp)
    loss = loss_fcn(output, label)
    return loss.item()


if __name__ == "__main__":
    Model = HandGesture(input_dim=126,hidden_dim=256, n_layers=5, num_classes=10)
    criterion = nn.CrossEntropyLoss()
    train_dataset = Hand_Dataset(path='TrainData2.pkl', seq_len=12)
    val_dataset = Hand_Dataset(path='ValData2.pkl', seq_len=12)
    test_dataset = Hand_Dataset(path='TestData2.pkl', seq_len=12)

    train_dataloader = DataLoader(dataset=train_dataset,
                                           batch_size=15,
                                           shuffle=True,
                                            num_workers=1)

    val_dataloader = DataLoader(dataset=val_dataset,
                                           batch_size=15,
                                           shuffle=True,
                                            num_workers=1)

    test_dataloader = DataLoader(dataset=test_dataset,
                                        batch_size=1,
                                        shuffle=True,
                                        num_workers=1)
    Model.to('cuda')

    # Model.eval()
    # for sample in train_dataloader:
    #     kp = sample['keypoints'].float()
    #     kp = kp.to('cuda')
    #     output = Model(kp)
    #     print(output.shape)
    optimizer = torch.optim.SGD(Model.parameters(), lr=0.01, momentum=0.9, nesterov=True)
    scheduler = StepLR(optimizer, step_size=100, gamma=0.5)
    num_epochs = 200
    history = {'train loss':[], 'val loss':[], 'lr':[]}
    for epoch in range(num_epochs):
        loss_train_sum = 0
        loss_val_sum = 0

        for batch_train in train_dataloader:

            loss_train_batch = train_batch(Model, batch_train, optimizer, criterion)
            loss_train_sum += loss_train_batch

        for batch_val in val_dataloader:
            loss_val_batch = val_batch(Model, batch_val, criterion)
            loss_val_sum += loss_val_batch

        scheduler.step()

        loss_train_epoch = loss_train_sum/len(train_dataloader)
        loss_val_epoch = loss_val_sum/len(val_dataloader)

        history['train loss'].append(loss_train_epoch)
        history['val loss'].append(loss_val_epoch)
        history['lr'].append(scheduler.get_lr())
        print(f'Epoch: {epoch} train loss: {loss_train_epoch:.4f} val loss: {loss_val_epoch:.4f}')


    torch.save(Model.state_dict(), 'Model_2.pth')
    # Model.load_state_dict(torch.load('model.pth'))
    #test
    Model.eval()
    with torch.no_grad():
        gt = []
        pred = []
        # n_samples = 0
        # n_correct = 0
        for sample in test_dataloader:
            kp = sample['keypoints'].float()
            kp = kp.to('cuda')
            label = sample['label'].long()
            # label = label.to('cuda')
            gt.append(int(label))
            output = Model(kp)
            output = softmax(output).to('cpu')
            pred.append(output.detach().numpy()[0])
            # n_samples += label.size(0)
            # n_correct += (predicted == label).sum().item()
    gt = np.array(gt)
    pred = np.array(pred)
    np.save('gt', gt)
    np.save('pred', pred)
    print(f'top 1 acc: {top_k_acc(gt, pred, k=1)}')
    print(f'top 5 acc: {top_k_acc(gt, pred, k=5)}')

    epochs = range(num_epochs)
    plt.plot(epochs, history['train loss'], label='train loss')
    plt.plot(epochs, history['val loss'], label='val loss')
    plt.legend()
    plt.show()
