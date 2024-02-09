import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from src.utils import *


def logistic(x):
    return torch.log(1 + torch.exp(x))


class FairModel(nn.Module):

    def __init__(self, n_features, l2_reg, sf_reg, lf_reg, path, device):
        super().__init__()
        self.l2_reg = l2_reg
        self.sf_reg = sf_reg
        self.lf_reg = lf_reg
        self.path = path

        self.linear = nn.Linear(n_features, 1)
        self.sigmoid = nn.Sigmoid()
        self.loss = nn.BCELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=0.01)

        self.device = device
        self.to(device)

    def forward(self, s, x, is_prob=True):
        if not torch.is_tensor(s):
            s = torch.FloatTensor(s)
            x = torch.FloatTensor(x)
        s = s.to(self.device)
        x = x.to(self.device)

        s = F.one_hot(s.squeeze().type(torch.LongTensor), num_classes=2).to(x.device)
        sx = torch.cat([s, x], dim=-1)
        h = self.linear(sx)
        p = self.sigmoid(h)
        if is_prob:
            return p
        else:
            return h

    def predict(self, s, x):
        pred = self.forward(s, x)
        pred_y = pred.round().detach().cpu().numpy()
        return pred_y

    def sample(self, s, x, scale=1.0):
        prob = self(s, x)
        y = torch.bernoulli(prob * scale)
        return y.detach().numpy()

    def compute_loss(self, s, x, y):
        pred_y = self(s, x)
        y = y.to(self.device)
        loss = self.loss(pred_y, y)
        l2 = torch.norm(self.linear.weight) ** 2
        return loss + self.l2_reg * l2

    def compute_short_fairness(self, s, x):
        x = x[s.squeeze() == 0]
        s = s[s.squeeze() == 0]
        s_pos = torch.ones_like(s)
        s_neg = torch.zeros_like(s)

        y_pos = self(s_pos, x, is_prob=False)
        y_neg = self(s_neg, x, is_prob=False)

        p_y_pos = sum(y_pos >= 0) / len(y_pos)
        p_y_neg = sum(y_neg >= 0) / len(y_neg)

        fair_cons1 = torch.mean(logistic(y_pos) + logistic(-y_neg) - 1)
        fair_cons2 = torch.mean(logistic(y_neg) + logistic(-y_pos) - 1)
        
        return torch.relu(fair_cons1), p_y_pos - p_y_neg

    def compute_long_fairness(self, s, x):
        neg_x = x[s.squeeze() == 0]
        neg_s = torch.zeros((neg_x.shape[0], 1))
        neg_pred_y = self(neg_s, neg_x, is_prob=False)

        pos_x = x[s.squeeze() == 1]
        pos_s = torch.zeros((pos_x.shape[0], 1))
        pos_pred_y = self(pos_s, pos_x, is_prob=False)

        p_y_pos = sum(pos_pred_y >= 0) / len(pos_pred_y)
        p_y_neg = sum(neg_pred_y >= 0) / len(neg_pred_y)

        fair_cons1 = torch.mean(logistic(pos_pred_y)) + torch.mean(logistic(-neg_pred_y)) - 1
        fair_cons2 = torch.mean(logistic(neg_pred_y)) + torch.mean(logistic(-pos_pred_y)) - 1

        return torch.relu(fair_cons1), p_y_pos - p_y_neg
    
    def train(self, s, OXs, OYs, Xs, Ys, num_epochs):
        s = torch.FloatTensor(s)
        OXs = torch.FloatTensor(OXs)
        OYs = torch.FloatTensor(OYs)
        Xs = torch.FloatTensor(Xs)
        Ys = torch.FloatTensor(Ys)

        epoch = 0.
        while epoch <= num_epochs:

            loss, o_loss, s_fair = 0., 0., 0.
            for i in range(Xs.size(1)):
                o_loss += self.compute_loss(s, OXs[:, i], OYs[:, i])
                s_fair += self.compute_short_fairness(s, Xs[:, i])[0]
            l_fair = self.compute_long_fairness(s, Xs[:, i])[0]
            
            loss = o_loss + self.sf_reg * s_fair + self.lf_reg * l_fair

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            epoch += 1
            if epoch == 1 or epoch % 100 == 0:
                print(f"Epoch: {epoch}, loss: {loss.item():0.4f}")

        torch.save(self.state_dict(), self.path)
        
    def eval(self, valid_data):
        s, x, y = valid_data
        s = torch.FloatTensor(s)
        x = torch.FloatTensor(x)
        y = torch.FloatTensor(y)

        for i in range(x.size(1)):
            pred_y = self.predict(s, x[:, i])

            acc = metrics.accuracy_score(y[:, i], pred_y) * 100
            short_fair = self.compute_short_fairness(s, x[:, i])[1]
            long_fair = self.compute_long_fairness(s, x[:, i])[1]
            print(f"Step:{i:6.0f}, ACC:{acc:6.3f}%, Short-Fair:{short_fair.item():6.3f}, "
                  f"Long-Fair:{long_fair.item():6.3f}")