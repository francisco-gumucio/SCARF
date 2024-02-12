import torch
import torch.nn as nn
from sklearn import metrics
from geomloss import SamplesLoss
from torch.optim import Adam, RMSprop

from src.utils import *
from src.train_classifier import generate_from_gan


def compute_demographic_parity(model, s_mb, x_mb):
    y_pred = model(s_mb, x_mb)
    y_pos = y_pred[s_mb.squeeze() == 1]
    y_neg = y_pred[s_mb.squeeze() == 0]
    dp = y_pos.mean() - y_neg.mean()

    p_y_pos = (y_pos >= 0.5).sum() / len(y_pos)
    p_y_neg = (y_neg >= 0.5).sum() / len(y_neg)

    return dp.abs(), p_y_pos - p_y_neg


def compute_equal_opportunity(model, s_mb, x_mb, y_mb):
    if torch.is_tensor(y_mb):
        y_mb = y_mb.cpu().numpy()
    y_pred = model(s_mb, x_mb)
    y_pos = y_pred[(s_mb.squeeze() == 1).cpu().numpy() & (y_mb.squeeze() == 1)]
    y_neg = y_pred[(s_mb.squeeze() == 0).cpu().numpy() & (y_mb.squeeze() == 1)]
    eo = y_pos.mean() - y_neg.mean()

    p_y_pos = (y_pos >= 0.5).sum() / len(y_pos)
    p_y_neg = (y_neg >= 0.5).sum() / len(y_neg)

    return eo.abs(), p_y_pos - p_y_neg 


@count_time
def risk_minimization(batch_size, seq_len, true_model, clf, G, loader, dp_lambda, device, save_path, criterion='dp'):

    loss_fn = torch.nn.BCELoss()
    optim_C = torch.optim.Adam(clf.parameters(), lr=0.0005)

    num_epoch = 0
    patience = 20
    best_loss = float('inf')
    while num_epoch < 500:
        data_loader = generate_from_gan(batch_size, seq_len, loader, clf, G, device)

        loss, num = 0., 0
        for s_mb, x_mb, y_mb in zip(*data_loader):
            s_mb = s_mb.to(device)

            loss_C, loss_F = 0., 0.
            for i in range(x_mb.size(1)):
                y_pred = y_mb[:, i]
                y_true = true_model.sample(s_mb, x_mb[:, i])
                y_true = torch.FloatTensor(y_true).to(device)

                loss_C += loss_fn(y_pred, y_true)
                if criterion == 'dp':
                    loss_F += compute_demographic_parity(clf, s_mb.detach(), x_mb[:, i].detach())[0]
                if criterion == 'eo':
                    loss_F += compute_equal_opportunity(clf, s_mb.detach(), x_mb[:, i].detach(), y_true.detach())[0]
            loss += loss_C + dp_lambda * loss_F
            num += x_mb.shape[0]

        optim_C.zero_grad()
        loss.backward()
        optim_C.step()

        num_epoch += 1
        if loss <= best_loss:
            torch.save(clf.state_dict(), save_path)
            print("Save")
            best_loss = loss
            counter = 0
        print(f"epochs: {num_epoch}, loss: {loss:.5f}")


wloss = SamplesLoss("sinkhorn", p=1, blur=0.01)
def valid_classifier(seq_len, true_model, clf, G, loader, device, verbose=True, criterion='dp', idx=0):
    batchs = [len(s_mb) for s_mb, _, _ in loader]
    data_loader = generate_from_gan(batchs[0], seq_len, loader, clf, G, device, idx=idx)

    for s_mb, x_mb, y_mb in zip(*data_loader):
        s_mb = s_mb.to(device)
        x_mb = x_mb.to(device)

        for i in range(x_mb.size(1)):
            y_pred = y_mb[:, i].round().detach().cpu().numpy()
            y_true = true_model.sample(s_mb, x_mb[:, i])
            
            acc = metrics.accuracy_score(y_true, y_pred) * 100
            if criterion == 'dp':
                dp_fair = compute_demographic_parity(clf, s_mb, x_mb[:, i])[1]
            if criterion == 'eo':
                dp_fair = compute_equal_opportunity(clf, s_mb, x_mb[:, i], y_true)[1]
            short_fair = compute_short_fairness(clf, s_mb, x_mb[:, i])[1]
            long_fair = compute_long_fairness(clf, s_mb, x_mb[:, i])
            w_loss = wloss(x_mb[:, i][s_mb.squeeze()==0], x_mb[:, i][s_mb.squeeze()==1])
        
            if verbose:
                print(f"Step:{i:6.0f}, ACC:{acc:6.3f}%, DP/EO-Fair: {dp_fair.item():6.3f}, Short-Fair:{short_fair.item():6.3f}, Long-Fair:{long_fair.item():6.3f}, W-dist:{w_loss:6.4f}")
    return short_fair.item(), long_fair.item()