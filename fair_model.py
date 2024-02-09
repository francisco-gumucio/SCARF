import torch
import torch.nn as nn
import torch.nn.functional as F


def combine_features(s, X):
    return torch.cat([s.unsqueeze(1), X], dim=-1)


class FairModel(nn.Module):

    def __init__(self, n_features, lr, l2_reg, sf_reg, lf_reg):
        super().__init__()
        self.l2_reg = l2_reg
        self.sf_reg = sf_reg
        self.lf_reg = lf_reg

        self.linear = nn.Linear(n_features, 1)
        self.sigmoid = nn.Sigmoid()
        self.loss = nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        self.old_linear_weight = None
        self.old_linear_bias = None

        nn.init.normal_(self.linear.weight)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, s, X):
        Xs = combine_features(s, X)
        h = self.linear(Xs)
        p = self.sigmoid(h)
        return h.squeeze(), p.squeeze()

    def predict(self, s, X):
        _, p self.forward(s, X)
        pred_y = torch.round(p)
        return pred_y.deatch().cpu().numpy()

    @property
    def params(self):
        w = self.linear.weight.detach().cpu().numpy()[0]
        b = self.linear.bias.detach().cpu().numpy()
        wb = np.hstack([w, b])
        return wb

    def compute_loss(self, s, X, y):
        _, pred_y = self.forward(s, X)
        loss = self.loss(pred_y, y)
        l2 = torch.norm(self.linear.weight) ** 2
        return loss + self.l2_reg * l2

    def train(self, s, OXs, OYs, Xs, Ys, epochs=0, tol=1e-7):

        losses, o_losses, s_fairs, l_fairs = [], [], [], []

        gap = 1e30
        pre_loss = 1e30
        while gap > tol or epochs > 0:

            loss, o_loss, s_fair = 0, 0, 0
            for i, (OX, Oy, X, y) in enumerate(zip(OX, Oy, X, y)):
                Oy = 

                o_loss += self.compute_loss(s, OX, Oy)
