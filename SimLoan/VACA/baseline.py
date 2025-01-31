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

    

def PGD_effort(model, x, improvable_indices, iter, lr, delta):
    
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
            grad_adjusted = lr * efforts.grad
            efforts[:, improvable_indices] -= (lr / ((i + 1) ** 0.5)) * grad_adjusted[:, improvable_indices]
            efforts[:, improvable_indices] = torch.clamp(efforts[:, improvable_indices], -delta, delta)
            
            # Set gradients to zero for unimprovable indices
            efforts[:, unimprovable_indices] = 0.0

        # Detach efforts to keep it as a leaf tensor
        efforts = efforts.detach().clone().requires_grad_(True)
        
    Yhat = model.predict(s, x + efforts)

    return Yhat, efforts

def trainer_fb_fair(model, train_loader, improvable_indices, optimizer, device, n_epochs, lambda_, delta_effort = 1, effort_iter = 20, effort_lr = 1):
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

                    loss_z[z] = torch.mean(yhat[group_idx]) - y_hat_mean
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
            x_pos = x[y.squeeze() >= 0.5]
            yhat_pos = yhat[y.squeeze() >= 0.5]

            cost = 0

            # prediction loss
            p_loss = loss_func(yhat.reshape(-1), y)
            cost += (1-lambda_)*p_loss

            f_loss = 0

            y_hat_mean = torch.mean(yhat_pos)
            loss_z = torch.zeros(len(sensitive_attrs), device = device)
            for z in sensitive_attrs:
                    z = int(z)
                    group_idx = x_pos[:, 0] == z
                    if group_idx.sum() == 0:
                        continue

                    loss_z[z] = torch.mean(yhat_pos[group_idx]) - y_hat_mean
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
            x_pos = x[y.squeeze() >= 0.5]
            x_neg = x[y.squeeze() < 0.5]
            yhat_pos = yhat[y.squeeze() >= 0.5]
            yhat_neg = yhat[y.squeeze() < 0.5]

            cost = 0

            # prediction loss
            p_loss = loss_func(yhat.reshape(-1), y)
            cost += (1-lambda_)*p_loss

            f_loss = 0

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

