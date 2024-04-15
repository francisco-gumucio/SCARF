import copy
import numpy as np
from sklearn import metrics
from geomloss import SamplesLoss

import torch
import torch.nn as nn
from torch.optim import Adam, RMSprop
from torch.utils.data import DataLoader, TensorDataset
from src.utils import *


wloss = SamplesLoss("sinkhorn", p=1, blur=0.01)


def generate_from_gan(batch_size, seq_len, loader, clf, G, device, idx=0):
    gen_s, gen_x, gen_y = [], [], []

    for s_mb, x_mb, y_mb in loader:
        x_mb = x_mb.to(device)
        z_mb = torch.randn(x_mb.shape[0], seq_len-1, x_mb.shape[-1]).to(device)

        gen_x_mb, gen_y_mb, _ = G(x_mb[:, idx], z_mb, s_mb, clf)
        
        gen_s.append(s_mb)
        gen_x.append(gen_x_mb)
        gen_y.append(gen_y_mb)

    return [gen_s, gen_x, gen_y]


@count_time
def risk_minimization(batch_size, seq_len, true_model, clf, G, loader, w_lambda, f_lambda, device, save_path):

    loss_fn = torch.nn.BCELoss()
    optim_C = torch.optim.Adam(clf.parameters(), lr=0.0005)

    num_epoch = 0
    patience = 20
    best_loss = float('inf')
    while num_epoch < 500:
        data_loader = generate_from_gan(batch_size, seq_len, loader, clf, G, device)

        loss, num = 0., 0
        old_param = clf.get_params()
        for s_mb, x_mb, y_mb in zip(*data_loader):
            s_mb = s_mb.to(device)

            loss_C, loss_F = 0., 0.
            for i in range(x_mb.size(1)):
                y_pred = y_mb[:, i]
                y_true = true_model.sample(s_mb, x_mb[:, i])
                y_true = torch.FloatTensor(y_true).to(device)

                loss_C += loss_fn(y_pred, y_true)
                loss_F += compute_short_fairness(clf, s_mb.detach(), x_mb[:, i].detach())[0]

            wwloss = wloss(x_mb[:,-1][s_mb.squeeze()==0], x_mb[:,-1][s_mb.squeeze()==1])
            loss_epoch = loss_C + f_lambda * loss_F + w_lambda * wwloss

            loss += loss_epoch
            num += x_mb.shape[0]

        optim_C.zero_grad()
        loss.backward()
        optim_C.step()

        num_epoch += 1
        new_param = clf.get_params()
        gap = np.linalg.norm(new_param - old_param)

        if loss <= best_loss:
            torch.save(clf.state_dict(), save_path)
            print("Save")
            best_loss = loss
            counter = 0
        # else:
        #     counter += 1
        #     if counter == patience:
        #         break
        
        print(f"epochs: {num_epoch}, loss: {loss:.5f}, gap: {gap:.5f}")


def valid_classifier(seq_len, true_model, clf, G, loader, device, idx=0, verbose=True):
    batchs = [len(s_mb) for s_mb, _, _ in loader]
    data_loader = generate_from_gan(batchs[0], seq_len, loader, clf, G, device, idx=idx)

    for s_mb, x_mb, y_mb in zip(*data_loader):
        s_mb = s_mb.to(device)
        x_mb = x_mb.to(device)

        for i in range(x_mb.size(1)):
            y_pred = y_mb[:, i].round().detach().cpu().numpy()
            y_true = true_model.sample(s_mb, x_mb[:, i])
            
            acc = metrics.accuracy_score(y_true, y_pred) * 100
            short_fair = compute_short_fairness(clf, s_mb, x_mb[:, i])[1]
            long_fair = compute_long_fairness(clf, s_mb, x_mb[:, i])
            w_loss = wloss(x_mb[:, i][s_mb.squeeze()==0], x_mb[:, i][s_mb.squeeze()==1])
        
            if verbose:
                print(f"Step:{i:6.0f}, ACC:{acc:6.3f}%, Short-Fair:{short_fair.item():6.3f}, Long-Fair:{long_fair.item():6.3f}, W-dist:{w_loss:6.4f}")
    return short_fair.item(), long_fair.item()



@count_time
def retrain_classifier(true_model, clf, G, loader, valid_loader, zs, vzs, 
                       w_lambda, f_lambda, writer, device, save_path):
    loss_fn = nn.BCELoss()
    optim_C = Adam(clf.parameters(), lr=0.0005)

    epoch, counter = 0, 0
    short_fair, long_fair = float('inf'), float('inf')
    while True:
        epoch_loss = 0.
        epoch_wloss = 0.
        epoch_floss = 0.
        for k, ((s_mb, x_mb, y_mb), z_mb) in enumerate(zip(loader, zs)):
            s_mb = s_mb.to(device)
            x_mb = x_mb.to(device)
            z_mb = z_mb.to(device)

            # Train CLF Model
            optim_C.zero_grad()

            gen_x_mb, gen_y_prob, gen_y_mb = G(x_mb[:, 0], z_mb, s_mb.cpu(), clf)
        
            loss_C, loss_F = 0., 0.
            for i in range(gen_x_mb.size(1)):
                y_true = true_model.sample(s_mb, gen_x_mb[:, i])
                y_true = torch.FloatTensor(y_true).to(device)
                loss_C += loss_fn(gen_y_prob[:, i], y_true)
                loss_F += compute_short_fairness(clf, s_mb.detach(), gen_x_mb[:, i].detach())[0]

            wwloss = wloss(gen_x_mb[:,-1][s_mb.squeeze()==0], gen_x_mb[:,-1][s_mb.squeeze()==1])
            loss_C = loss_C + f_lambda * loss_F + w_lambda * wwloss
            
            epoch_wloss += wwloss.item()
            epoch_loss += loss_C.item()
            epoch_floss += loss_F.item()

            loss_C.backward()
            optim_C.step()

        epoch += 1
        print(f"Epoch: {epoch}, L: {epoch_loss:.4f} F: {epoch_floss:.4f} W:{epoch_wloss:.4f}")
        writer.add_scalar('Step-3-Classifier-W1', epoch_wloss, epoch)
        writer.add_scalar('Step-3-Classifier-SF', epoch_floss, epoch)
            
        sf, lf = evaluate_classifier(valid_loader, true_model, clf, G, vzs, device, verbose=False)
        print(f"Short: {short_fair:0.4f}, Long:{long_fair:0.4f}")
        if long_fair > abs(lf):
            short_fair, long_fair = abs(sf), abs(lf)
            torch.save(clf.state_dict(), save_path)
            print("Save Model!")
            counter = 0
        else:
            counter += 1
            if counter == 20:
                break


def evaluate_classifier(loader, true_model, clf, G, zs, device, idx=0, verbose=True):
    s0 = [s0 for s0, x0, y0 in loader][0]
    x0 = [x0 for s0, x0, y0 in loader][0][:, idx].to(device)
    zs = [z_mb for z_mb in zs][0].to(device)

    gen_x_mb, _, gen_y_mb = G(x0, zs, s0, clf)
    for i in range(gen_x_mb.size(1)):
        y_true = true_model.sample(s0.to(device), gen_x_mb[:, i])
        y_pred = gen_y_mb[:, i].cpu().numpy()

        acc = metrics.accuracy_score(y_true, y_pred) * 100
        short_fair = compute_short_fairness(clf, s0, gen_x_mb[:, i])[1]
        long_fair = compute_long_fairness(clf, s0, gen_x_mb[:, i])
        w_loss = wloss(gen_x_mb[:, i][s0.squeeze()==0], gen_x_mb[:, i][s0.squeeze()==1])
        
        if verbose:
            print(f"Step:{i:6.0f}, ACC:{acc:6.2f}%, Short-Fair:{short_fair.item():6.2f}, Long-Fair:{long_fair.item():6.2f}, W-dist:{w_loss:6.4f}")
    return short_fair.item(), long_fair.item()
