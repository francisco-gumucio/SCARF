import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, features, labels):

        self.features = features  # All columns except the last are features
        self.labels = labels     # Last column is the label

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        features = torch.tensor(self.features[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return features, label

class logReg(nn.Module):
    '''
    Logistic Regression model

    It works only for binary logistic regression. 
    '''
    def __init__(self, num_features):
        '''
        Parameters
        ----------
        num_features : int
            length of feature vector
        '''
        super().__init__()
        self.num_features = num_features

        self.layers = nn.Sequential(
            nn.Linear(num_features,1),
            nn.Sigmoid())

    def forward(self, s, x):
        '''
        Forward pass for logistic regression

        Parameters
        ----------
        x : Tensor
            Input tensor for the model
        
        Return
        ------
        x : Tensor
            Output of the model
        '''
        if x.shape[1] == self.num_features:
            y = self.layers(x)
        else:
            sx = torch.cat((s, x), dim = 1)
            y = self.layers(sx)
        return y
    
    def predict(self, s, x):
    
        with torch.no_grad():
            y_prob = self.forward(s, x)
            y_pred = torch.round(y_prob)
        
        return y_pred

    

def PGD_effort(model, x, improvable_indices, iter, lr, delta, norm='l1'):
    
    efforts = torch.zeros_like(x, requires_grad=True).to(x.device)
    
    loss = torch.nn.BCELoss(reduction='sum')

    unimprovable_indices = []

    s = x[:, 0]


    for index in range(x.shape[1]):
        if index not in improvable_indices:
            unimprovable_indices.append(index)

    for i in range(iter):

        Yhat = model(s, x + efforts)
        Yhat = torch.sigmoid(Yhat)

        cost = loss(Yhat,torch.ones_like(Yhat, device=x.device))
        model.zero_grad()

        if efforts.grad is not None:
            efforts.grad.zero_()

        
        cost.backward()

        with torch.no_grad():
            # Update only on improvable indices
            grad_adjusted = efforts.grad
            efforts[:, improvable_indices] -= (lr / ((i + 1) ** 0.5)) * grad_adjusted[:, improvable_indices]
            # Normalize effort per instance (row-wise)
            if norm == 'l2':
                norms = efforts[:, improvable_indices].norm(p=2, dim=1, keepdim=True)
            elif norm == 'l1':
                norms = efforts[:, improvable_indices].abs().sum(dim=1, keepdim=True)
            else:
                raise ValueError(f"Unsupported norm '{norm}'. Use 'l1' or 'l2'.")

            scaling = torch.clamp(norms / delta, min=1.0)
            efforts[:, improvable_indices] /= scaling
            
            # Set gradients to zero for unimprovable indices
            efforts[:, unimprovable_indices] = 0.0

        # Detach efforts to keep it as a leaf tensor
        efforts = efforts.detach().clone().requires_grad_(True)
        
    Yhat = model.predict(s, x + efforts)

    return Yhat, efforts

def trainer_fb_fair(model, train_loader, improvable_indices, optimizer, device, n_epochs, lambda_, delta_effort = 1, effort_iter = 10, effort_lr = 0.1):
    loss_func = torch.nn.BCELoss(reduction = 'mean')

    p_losses = []
    f_losses = []

    sensitive_attrs = [0, 1]

    for epoch in range(n_epochs):
        local_p_loss = []
        local_f_loss = []  
        for batch in train_loader:
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            s = x[:, 0]
            yhat = model(s, x)

            cost = 0

            # prediction loss
            p_loss = loss_func(yhat.reshape(-1), y)
            cost += (1-lambda_)*p_loss

            f_loss = 0

            x_batch = x[yhat.squeeze() < 0.5].to(device)
            y_batch = y[yhat.squeeze() < 0.5].to(device)
            yhat_max, _ = PGD_effort(model, x_batch, improvable_indices, effort_iter, effort_lr, delta_effort)
            loss_mean = loss_func(yhat_max, torch.ones_like(yhat_max, device=yhat_max.device))
            loss_z = torch.zeros(len(sensitive_attrs), device = device)
            for z in sensitive_attrs:
                    z = int(z)
                    group_idx = x_batch[:, 0] == z
                    if group_idx.sum() == 0:
                        continue
                    loss_z[z] = loss_func(yhat_max.reshape(-1)[group_idx], torch.ones(group_idx.sum(), device=yhat_max.device))
                    f_loss += torch.abs(loss_z[z] - loss_mean)

            cost += lambda_*f_loss
            
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()
            local_p_loss.append(p_loss.item())
            if hasattr(f_loss, 'item'):
                local_f_loss.append(f_loss.item())
            else:
                local_f_loss.append(f_loss)

        p_losses.append(np.mean(local_p_loss))
        f_losses.append(np.mean(local_f_loss))

        if epoch % 10 == 0:
            print(f'Epoch {epoch} p_loss: {p_losses[-1]}, f_loss: {f_losses[-1]}')

    plt.plot(p_losses, label = 'Prediction Loss')
    plt.plot(f_losses, label = 'Fairness Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    return model
    

def test_fb_fair(model, test_loader, improvable_indices, device, delta_effort = 1, effort_iter = 20, effort_lr = 1):

    b = next(iter(test_loader))

    data = torch.empty(0, b[0].shape[1]+1)

    for batch in test_loader:
        x, y = batch
        x = x.to(device)
        y = y.to(device)
        s = x[:, 0]
        yhat = model.predict(s, x)
        neg_indices = yhat.squeeze() < 0.5
        pos_indices = yhat.squeeze() >= 0.5

        x_neg = x[yhat.squeeze() < 0.5].to(device)
        x_pos = x[yhat.squeeze() >= 0.5].to(device)
        y_pos = y[yhat.squeeze() >= 0.5].to(device)
        yhat_max, efforts = PGD_effort(model, x_neg, improvable_indices, effort_iter, effort_lr, delta_effort)

        pos_data = torch.cat((x_pos, torch.round(y_pos.view(len(y_pos), 1))), dim = 1)
        neg_data = torch.cat((x_neg + efforts, torch.round(yhat_max.view(len(yhat_max), 1))), dim = 1)
        data = torch.cat((data.to(device), pos_data.to(device), neg_data.to(device)), dim = 0)
    
    indices = torch.randperm(data.shape[0]) 
    data = data[indices] 

    return model, efforts, data

def trainer_er(model, train_loader, improvable_indices, optimizer, device, n_epochs,
               lambda_, tau=0.5, delta_effort=1.0, effort_iter=10,
               effort_lr=0.1, effort_norm='l2'):
    """
    Train a classifier using ER (Effort-based Recourse) fairness regularization.
    
    Args:
        model: torch.nn.Module that takes s, x as input.
        train_loader: DataLoader yielding (x, y) batches.
        improvable_indices: feature indices that can be changed.
        optimizer: model optimizer.
        device: torch device.
        n_epochs: number of epochs.
        lambda_: weight of the fairness loss.
        tau: threshold below which recourse is applied.
        delta_effort: max norm for efforts.
        effort_iter: number of PGD steps.
        effort_lr: step size for PGD.
        effort_norm: 'l1' or 'l2'.
    """
    loss_func = torch.nn.BCELoss(reduction='mean')
    sensitive_attrs = [0, 1]

    p_losses = []
    f_losses = []

    for epoch in range(n_epochs):
        local_p_loss = []
        local_f_loss = []

        for batch in train_loader:
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            s = x[:, 0].long()
            yhat = model(s, x)

            # prediction loss
            p_loss = loss_func(yhat.reshape(-1), y)
            cost = (1 - lambda_) * p_loss

            # Find samples needing recourse
            recourse_mask = yhat.squeeze() < tau
            x_rec = x[recourse_mask]
            s_rec = s[recourse_mask]

            if x_rec.size(0) == 0:
                cost.backward()
                optimizer.step()
                continue

            # Apply PGD effort to improve prediction
            yhat_max, efforts = PGD_effort(model, x_rec, improvable_indices,
                                           effort_iter, effort_lr, delta_effort)

            # Fairness loss: equalize *expected recourse effort* across groups
            effort_by_group = {}
            valid_group = {}

            for z in sensitive_attrs:
                group_mask = s_rec == z
                if group_mask.sum() > 0:
                    if effort_norm == 'l1':
                        group_effort = efforts[group_mask].abs().sum(dim=1)
                    else:  # 'l2'
                        group_effort = efforts[group_mask].norm(p=2, dim=1)
                    effort_by_group[z] = group_effort.mean() * 10
                    valid_group[z] = True
                else:
                    effort_by_group[z] = torch.tensor(0.0, device=device)  # Optional, to keep dictionary complete
                    valid_group[z] = False

            if valid_group[0] and valid_group[1]:
                f_loss = torch.abs(effort_by_group[0] - effort_by_group[1])
            else:
                f_loss = torch.tensor(1.0, device=device)


            cost += lambda_ * f_loss

            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

            local_p_loss.append(p_loss.item())
            local_f_loss.append(f_loss.item())

        p_losses.append(np.mean(local_p_loss))
        f_losses.append(np.mean(local_f_loss))

        if epoch % 10 == 0:
            print(f"Epoch {epoch} | p_loss: {p_losses[-1]:.4f}, ER fairness loss: {f_losses[-1]:.4f}")
            print("PGD effort norms:", efforts.norm(p=2, dim=1).mean().item())

            if valid_group[0]:
                print("Mean effort for group 0:", effort_by_group[0])
            if valid_group[1]:
                print("Mean effort for group 1:", effort_by_group[1])

    # Plot loss curves
    plt.plot(p_losses, label='Prediction Loss')
    plt.plot(f_losses, label='ER Fairness Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title("Training Losses (Prediction vs. ER Fairness)")
    plt.legend()
    plt.show()

    return model

def test_er(model, test_loader, improvable_indices, device, delta_effort=1.0, 
            effort_iter=20, effort_lr=1.0, effort_norm='l2'):
    """
    Evaluate a model for Effort-based Recourse (ER) fairness on a test set.

    Args:
        model: Trained model that takes (s, x) as input.
        test_loader: DataLoader yielding (x, y) batches.
        improvable_indices: feature indices eligible for intervention.
        device: torch device.
        delta_effort: maximum norm bound for effort.
        effort_iter: number of PGD steps.
        effort_lr: step size for PGD.
        effort_norm: norm type ('l1' or 'l2') for effort.

    Returns:
        model: the input model (unchanged).
        efforts: final batch's effort tensor.
        data: concatenated (x + yhat) data for all test samples.
        er_disparity: difference in expected effort between groups.
    """
    b = next(iter(test_loader))
    data = torch.empty(0, b[0].shape[1] + 1).to(device)

    effort_by_group = {0: [], 1: []}

    for batch in test_loader:
        x, y = batch
        x = x.to(device)
        y = y.to(device)
        s = x[:, 0].long()
        yhat = model.predict(s, x)

        # Separate positive and negative predictions
        neg_mask = yhat.squeeze() < 0.5
        pos_mask = yhat.squeeze() >= 0.5

        x_neg = x[neg_mask]
        s_neg = s[neg_mask]
        y_neg = y[neg_mask]

        x_pos = x[pos_mask]
        y_pos = y[pos_mask]

        if x_neg.size(0) > 0:
            yhat_max, efforts = PGD_effort(model, x_neg, improvable_indices, 
                                           effort_iter, effort_lr, delta_effort)
        else:
            yhat_max = torch.empty(0, device=device)
            efforts = torch.zeros_like(x_neg)

        # Compute expected effort for each group
        for z in [0, 1]:
            group_mask = s_neg == z
            if group_mask.sum() > 0:
                if effort_norm == 'l1':
                    group_effort = efforts[group_mask].abs().sum(dim=1) * 10
                else:  # 'l2'
                    group_effort = efforts[group_mask].norm(p=2, dim=1) * 10
                effort_by_group[z].append(group_effort)

        # Concatenate data for analysis
        pos_data = torch.cat((x_pos, torch.round(y_pos.view(-1, 1))), dim=1)
        neg_data = torch.cat((x_neg + efforts, torch.round(yhat_max.view(-1, 1))), dim=1)
        data = torch.cat((data, pos_data, neg_data), dim=0)

    # Shuffle the full data
    indices = torch.randperm(data.shape[0])
    data = data[indices]

    # Compute ER disparity
    group_effort_means = {}
    for z in [0, 1]:
        if len(effort_by_group[z]) > 0:
            all_group_efforts = torch.cat(effort_by_group[z])
            group_effort_means[z] = all_group_efforts.mean().item()
        else:
            group_effort_means[z] = 0.0

    er_disparity = abs(group_effort_means[0] - group_effort_means[1])

    return model, efforts, data, er_disparity

def trainer_iler(model, train_loader, improvable_indices, optimizer, device, n_epochs, 
                 lambda_, tau=0.5, delta_effort=1.0, effort_iter=10, 
                 effort_lr=0.1, effort_norm='l2'):
    
    loss_func = torch.nn.BCELoss(reduction='mean')
    sensitive_attrs = [0, 1]

    p_losses = []
    f_losses = []

    for epoch in range(n_epochs):
        local_p_loss = []
        local_f_loss = []

        for batch in train_loader:
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            s = x[:, 0]  # sensitive attribute
            yhat = model(s, x)

            # prediction loss
            p_loss = loss_func(yhat.reshape(-1), y)
            cost = (1 - lambda_) * p_loss

            # Find examples below threshold
            improvable_mask = yhat.squeeze() < tau
            x_batch = x[improvable_mask]
            s_batch = s[improvable_mask]

            if x_batch.size(0) == 0:
                cost.backward()
                optimizer.step()
                continue

            # Generate optimal effort via PGD
            yhat_max, efforts = PGD_effort(model, x_batch, improvable_indices, 
                                           effort_iter, effort_lr, delta_effort)

            # Fairness loss: ILER = max difference in normalized effort between groups
            effort_by_group = {}
            valid_group = {}

            for z in sensitive_attrs:
                group_mask = s_batch == z
                if group_mask.sum() > 0:
                    if effort_norm == 'l1':
                        group_effort = efforts[group_mask].abs().sum(dim=1)
                    else:  # 'l2'
                        group_effort = efforts[group_mask].norm(p=2, dim=1)
                    effort_by_group[z] = group_effort.mean()
                    valid_group[z] = True
                else:
                    valid_group[z] = False

            if valid_group[0] and valid_group[1]:
                f_loss = torch.abs(effort_by_group[0] - effort_by_group[1])
            else:
                f_loss = torch.tensor(1.0, device=device)


            cost += lambda_ * f_loss

            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

            local_p_loss.append(p_loss.item())
            local_f_loss.append(f_loss.item())

        p_losses.append(np.mean(local_p_loss))
        f_losses.append(np.mean(local_f_loss))

        if epoch % 10 == 0:
            print(f"Epoch {epoch} | p_loss: {p_losses[-1]:.4f}, ILER loss: {f_losses[-1]:.4f}")
            print("PGD effort norms:", efforts.norm(p=2, dim=1).mean().item())

            if valid_group[0]:
                print("Mean effort for group 0:", effort_by_group[0])
            if valid_group[1]:
                print("Mean effort for group 1:", effort_by_group[1])

    # Plotting
    plt.plot(p_losses, label='Prediction Loss')
    plt.plot(f_losses, label='ILER Fairness Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title("Training Losses (Prediction vs. IFLCR Fairness)")
    plt.show()

    return model

def test_iler(model, test_loader, improvable_indices, device, delta_effort=1.0, effort_iter=20, 
              effort_lr=1.0, effort_norm='l1'):
    b = next(iter(test_loader))
    data = torch.empty(0, b[0].shape[1] + 1).to(device)

    all_efforts = []
    effort_by_group = {0: [], 1: []}

    for batch in test_loader:
        x, y = batch
        x = x.to(device)
        y = y.to(device)
        s = x[:, 0].long()
        yhat = model.predict(s, x)

        # Separate positive and negative predictions
        neg_mask = yhat.squeeze() < 0.5
        pos_mask = yhat.squeeze() >= 0.5

        x_neg = x[neg_mask]
        s_neg = s[neg_mask]
        y_neg = y[neg_mask]

        x_pos = x[pos_mask]
        y_pos = y[pos_mask]

        if x_neg.size(0) > 0:
            yhat_max, efforts = PGD_effort(model, x_neg, improvable_indices, effort_iter, effort_lr, delta_effort)
        else:
            yhat_max = torch.empty(0, device=device)
            efforts = torch.zeros_like(x_neg)

        # Log efforts by sensitive group
        for group in [0, 1]:
            group_mask = s_neg == group
            if group_mask.sum() > 0:
                if effort_norm == 'l1':
                    group_effort = efforts[group_mask].abs().sum(dim=1)
                else:  # 'l2'
                    group_effort = efforts[group_mask].norm(p=2, dim=1)
                effort_by_group[group].append(group_effort)

        # Prepare for output logging
        pos_data = torch.cat((x_pos, torch.round(y_pos.view(-1, 1))), dim=1)
        neg_data = torch.cat((x_neg + efforts, torch.round(yhat_max.view(-1, 1))), dim=1)
        data = torch.cat((data, pos_data, neg_data), dim=0)

    # Shuffle output data
    indices = torch.randperm(data.shape[0])
    data = data[indices]

    # Compute ILER disparity
    group_effort_means = {}
    for group in [0, 1]:
        if len(effort_by_group[group]) > 0:
            group_effort_all = torch.cat(effort_by_group[group])
            group_effort_means[group] = group_effort_all.mean().item()
        else:
            group_effort_means[group] = 0.0

    iler_disparity = abs(group_effort_means[0] - group_effort_means[1])

    return model, efforts, data, iler_disparity


def trainer_bounded_effort(model, train_loader, improvable_indices, optimizer, device, n_epochs, 
                           lambda_, tau=0.5, delta_effort=1.0, effort_iter=10, 
                           effort_lr=0.1, optimal_effort=False, effort_norm='l1'):
    loss_func = torch.nn.BCELoss(reduction='mean')

    p_losses = []
    f_losses = []

    sensitive_attrs = [0, 1]

    for epoch in range(n_epochs):
        local_p_loss = []
        local_f_loss = []

        for batch in train_loader:
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            s = x[:, 0]  
            yhat = model(s, x)

            # prediction loss
            p_loss = loss_func(yhat.reshape(-1), y)
            cost = (1 - lambda_) * p_loss

            # apply effort to samples below threshold
            x_batch = x[yhat.squeeze() < tau]
            s_batch = s[yhat.squeeze() < tau]
            if x_batch.size(0) == 0:
                cost.backward()
                optimizer.step()
                continue

            yhat_max, efforts = PGD_effort(model, x_batch, improvable_indices, effort_iter, effort_lr, delta_effort)

            # fairness loss: equalize success across sensitive groups
            f_loss = 0
            loss_mean = (len(x_batch) / len(x)) * loss_func(yhat_max.reshape(-1), torch.ones(len(yhat_max), device=device))
            loss_z = torch.zeros(len(sensitive_attrs), device=device)

            for z in sensitive_attrs:
                z = int(z)
                group_idx = s_batch == z
                if group_idx.sum() == 0:
                    continue
                group_loss = loss_func(yhat_max.reshape(-1)[group_idx], torch.ones(group_idx.sum(), device=device))
                group_weight = s_batch[group_idx].shape[0] / s[s == z].shape[0]
                loss_z[z] = group_weight * group_loss
                f_loss += torch.abs(loss_z[z] - loss_mean)

            cost += lambda_ * f_loss

            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

            local_p_loss.append(p_loss.item())
            local_f_loss.append(f_loss.item() if hasattr(f_loss, 'item') else f_loss)

        p_losses.append(np.mean(local_p_loss))
        f_losses.append(np.mean(local_f_loss))

        if epoch % 10 == 0:
            print(f"Epoch {epoch} | p_loss: {p_losses[-1]:.4f}, f_loss: {f_losses[-1]:.4f}")

    plt.plot(p_losses, label='Prediction Loss')
    plt.plot(f_losses, label='Fairness Loss (Bounded Effort)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    return model

def test_bounded_effort(model, test_loader, improvable_indices, device, delta_effort=1.0, effort_iter=20, effort_lr=1.0, 
                        optimal_effort=False, effort_norm='l1'):
    b = next(iter(test_loader))
    data = torch.empty(0, b[0].shape[1] + 1)

    for batch in test_loader:
        x, y = batch
        x = x.to(device)
        y = y.to(device)
        s = x[:, 0]
        yhat = model.predict(s, x)

        neg_indices = yhat.squeeze() < 0.5
        pos_indices = yhat.squeeze() >= 0.5

        x_neg = x[neg_indices]
        x_pos = x[pos_indices]
        y_pos = y[pos_indices]

        # Apply bounded effort
        if x_neg.size(0) > 0:
            yhat_max, efforts = PGD_effort(model, x_neg, improvable_indices, effort_iter, effort_lr, delta_effort)
        else:
            yhat_max = torch.empty(0, device=device)
            efforts = torch.empty_like(x_neg)

        # Concatenate data for logging/analyzing
        pos_data = torch.cat((x_pos, torch.round(y_pos.view(-1, 1))), dim=1)
        neg_data = torch.cat((x_neg + efforts, torch.round(yhat_max.view(-1, 1))), dim=1)
        data = torch.cat((data.to(device), pos_data.to(device), neg_data.to(device)), dim=0)

    # Shuffle data
    indices = torch.randperm(data.shape[0])
    data = data[indices]

    return model, efforts, data


def trainer_dp_fair(model, train_loader, optimizer, device, n_epochs, lambda_):
    loss_func = torch.nn.BCELoss(reduction = 'mean')

    p_losses = []
    f_losses = []

    sensitive_attrs = [0, 1]

    for epoch in range(n_epochs):
        local_p_loss = []
        local_f_loss = []  
        for batch in train_loader:
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            s = x[:, 0]
            yhat = model(s, x)

            cost = 0

            # prediction loss
            p_loss = loss_func(yhat.reshape(-1), y)
            cost += (1-lambda_)*p_loss

            f_loss = 0

            y_hat_mean = torch.mean(yhat)
            loss_z = torch.zeros(len(sensitive_attrs), device = device)
            for z in sensitive_attrs:
                    z = int(z)
                    group_idx = x[:, 0] == z
                    if group_idx.sum() == 0:
                        continue

                    loss_z[z] = torch.abs(torch.mean(yhat[s == 0]) - torch.mean(yhat[s == 1]))
            f_loss += torch.max(loss_z)

            cost += lambda_*f_loss
            
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()
            local_p_loss.append(p_loss.item())
            if hasattr(f_loss, 'item'):
                local_f_loss.append(f_loss.item())
            else:
                local_f_loss.append(f_loss)

        p_losses.append(np.mean(local_p_loss))
        f_losses.append(np.mean(local_f_loss))

        if epoch % 10 == 0:
            print(f'Epoch {epoch} p_loss: {p_losses[-1]}, f_loss: {f_losses[-1]}')

    plt.plot(p_losses, label = 'Prediction Loss')
    plt.plot(f_losses, label = 'Fairness Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    return model

def test_dp_fair(model, test_loader, device):
    b = next(iter(test_loader))
    data = torch.empty(0, b[0].shape[1]+1).to(device)

    for batch in test_loader:
        x, y = batch
        x = x.to(device)
        y = y.to(device)
        s = x[:, 0]
        yhat = model.predict(s, x)

        _data = torch.cat((x, yhat.to(device)), dim = 1).to(device)
        data = torch.cat((data, _data), dim = 0)
        

    return model, data

def trainer_eo_fair(model, train_loader, optimizer, device, n_epochs, lambda_):
    loss_func = torch.nn.BCELoss(reduction='mean')

    p_losses = []
    f_losses = []

    sensitive_attrs = [0, 1]

    for epoch in range(n_epochs):
        local_p_loss = []
        local_f_loss = []

        for batch in train_loader:
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            s = x[:, 0]  # sensitive attribute
            yhat = model(s, x)

            # Prediction loss
            p_loss = loss_func(yhat.reshape(-1), y)
            cost = (1 - lambda_) * p_loss

            # Fairness loss: Equal Opportunity focuses only on y == 1
            f_loss = 0
            loss_z_y1 = torch.zeros(len(sensitive_attrs), device=device)

            for z in sensitive_attrs:
                z = int(z)
                idx_z_y1 = (x[:, 0] == z) & (y == 1)
                if idx_z_y1.sum() > 0:
                    loss_z_y1[z] = torch.mean(yhat[idx_z_y1])

            # EO fairness loss = disparity in TPRs (i.e., predictions where y==1)
            non_zero_y1 = loss_z_y1[loss_z_y1 != 0]
            if len(non_zero_y1) > 1:
                mean_y1 = non_zero_y1.mean()
                disparity_y1 = torch.abs(loss_z_y1 - mean_y1)
                f_loss += torch.max(disparity_y1)

            cost += lambda_ * f_loss

            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

            local_p_loss.append(p_loss.item())
            local_f_loss.append(f_loss.item() if hasattr(f_loss, 'item') else f_loss)

        p_losses.append(np.mean(local_p_loss))
        f_losses.append(np.mean(local_f_loss))

        if epoch % 10 == 0:
            print(f'Epoch {epoch} p_loss: {p_losses[-1]:.4f}, f_loss: {f_losses[-1]:.4f}')

    plt.plot(p_losses, label='Prediction Loss')
    plt.plot(f_losses, label='Fairness Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    return model

def test_eo_fair(model, test_loader, device):
    b = next(iter(test_loader))
    data = torch.empty(0, b[0].shape[1]+1)

    for batch in test_loader:
        x, y = batch
        x = x.to(device)
        y = y.to(device)
        s = x[:, 0]
        yhat = model.predict(s, x)

        _data = torch.cat((x, yhat.to(device)), dim = 1)
        data = torch.cat((data.to(device), _data.to(device)), dim = 0)
        

    return model, data

def trainer_eodd_fair(model, train_loader, optimizer, device, n_epochs, lambda_):
    loss_func = torch.nn.BCELoss(reduction = 'mean')

    p_losses = []
    f_losses = []

    sensitive_attrs = [0, 1]
    y_values = [0, 1]

    for epoch in range(n_epochs):
        local_p_loss = []
        local_f_loss = []  
        for batch in train_loader:
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            s = x[:, 0]
            yhat = model(s, x)

            cost = 0

            # prediction loss
            p_loss = loss_func(yhat.reshape(-1), y)
            cost += (1-lambda_)*p_loss
            mask = (s == 0) & (y > 0.5)
            tpr_0 = yhat[mask].sum() / mask.sum().clamp(min=1)
            mask = (s == 1) & (y > 0.5)
            tpr_1 = yhat[mask].sum() / mask.sum().clamp(min=1)
            mask = (s == 0) & (y < 0.5)
            fpr_0 = yhat[mask].sum() / mask.sum().clamp(min=1)
            mask = (s == 1) & (y < 0.5)
            fpr_1 = yhat[mask].sum() / mask.sum().clamp(min=1)

            f_loss = 0

            f_loss += torch.abs(tpr_0 - tpr_1) + torch.abs(fpr_0 - fpr_1)

            '''

            loss_z = torch.zeros(int(len(sensitive_attrs)*len(y_values)), device = device)
            p_pos = torch.mean(yhat_pos)
            p_neg = torch.mean(yhat_neg)
            for z in sensitive_attrs:
                    z = int(z)
                    group_idx = x_pos[:, 0] == z
                    if group_idx.sum() == 0:
                        continue

                    loss_z[z] = torch.mean(yhat_pos[group_idx]) - p_pos
                    group_idx = x_neg[:, 0] == z
                    loss_z[z + len(sensitive_attrs)] = torch.mean(yhat_neg[group_idx]) - p_neg
            f_loss += torch.abs(torch.sum(loss_z))

            '''

            cost += lambda_*f_loss
            
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()
            local_p_loss.append(p_loss.item())
            if hasattr(f_loss, 'item'):
                local_f_loss.append(f_loss.item())
            else:
                local_f_loss.append(f_loss)

        p_losses.append(np.mean(local_p_loss))
        f_losses.append(np.mean(local_f_loss))

        if epoch % 10 == 0:
            print(f'Epoch {epoch} p_loss: {p_losses[-1]}, f_loss: {f_losses[-1]}')

    plt.plot(p_losses, label = 'Prediction Loss')
    plt.plot(f_losses, label = 'Fairness Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    return model
    
def test_eodd_fair(model, test_loader, device):
    b = next(iter(test_loader))
    data = torch.empty(0, b[0].shape[1]+1)

    for batch in test_loader:
        x, y = batch
        x = x.to(device)
        y = y.to(device)
        s = x[:, 0]
        yhat = model.predict(s, x)

        _data = torch.cat((x, yhat.to(device)), dim = 1)
        data = torch.cat((data.to(device), _data.to(device)), dim = 0)
        

    return model, data

