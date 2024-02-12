import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from src.data_generator import *

SAVE_PATH = "../res/taiwan_true_model.pkl"

class TrueModel(nn.Module):

    def __init__(self, hiddens, seed=0):
        super().__init__()
        
        layers = []
        for in_dim, out_dim in zip(hiddens[:-1], hiddens[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU(inplace=True))
        layers.pop()
        self.sigmoid = nn.Sigmoid()
        self.model = nn.Sequential(*layers)

        self.loss_fn = nn.BCELoss()
        self.optim = Adam(self.parameters())

    def forward(self, sx, scale=1.0):
        return self.sigmoid(scale * self.model(sx))

    def predict(self, s, x, scale=1.0):
        sx = to_tensor(s, x)
        pred = self(sx, scale)
        pred_y = pred.detach().round().cpu().numpy()
        return pred_y

    # Synthetic-patience: 10
    # Taiwan-patience: 8
    def fit(self, s, x, y, patience=10):
        sx, y = to_tensor(s, x, y)

        epoch, counter = 0, 0
        best_loss = float('inf')
        while True:
            pred = self(sx)
            loss = self.loss_fn(pred, y)

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            
            epoch += 1
            if loss.item() <= best_loss:
                torch.save(self.state_dict(), SAVE_PATH)
                best_loss = loss.item()
                counter = 0
            else:
                counter += 1
                if counter == patience:
                    break
        print(f"TrueModel Fit Done in {epoch} epochs!")

    def sample(self, s, x, scale=2):
        sx = to_tensor(s, x)
        prob = self(sx, scale)
        y = torch.bernoulli(prob)
        return y.detach().cpu().numpy()


def generate_init_taiwan_dataset():

    sen_attr = ['X2']
    feat_attr = ['X12', 'X13', 'X14', 'X15', 'X16', 'X17']

    df = pd.read_csv("../data/taiwan.csv", usecols=sen_attr+feat_attr+['Y', 'X1'])
    df = df[(df['X12']>=0) & (df['X13']>=0) & (df['X14']>=0) & (df['X15']>=0) & (df['X16']>=0) & (df['X17']>=0)]
    df['X2'] -= 1

    for name in feat_attr:
        df[name] = df[name] / df['X1']
    df.drop(['X1'], axis=1)

    df00 = df[(df['X2'] == 0) & (df['Y'] == 0)].sample(n=2500, random_state=0)
    df01 = df[(df['X2'] == 0) & (df['Y'] == 1)].sample(n=2500, random_state=0)
    df10 = df[(df['X2'] == 1) & (df['Y'] == 0)].sample(n=2500, random_state=0)
    df11 = df[(df['X2'] == 1) & (df['Y'] == 1)].sample(n=2500, random_state=0)
    data = pd.concat([df00, df01, df10, df11]).sample(frac = 1)

    s = data[sen_attr].values
    X = data[feat_attr].values
    y = data['Y'].values.reshape(-1, 1)

    return s.astype(np.float64), X.astype(np.float64), y


@count_time
def generate_taiwan_sequential_datasets(seq_len, hiddens, epsilon, device, seed=0):
    # 初始化真实模型
    model = TrueModel(hiddens, seed) 

    # 生成初始数据集
    s0, x0, y0 = generate_init_taiwan_dataset()
    # if not Path(SAVE_PATH).exists():
    model.fit(s0, x0, y0)
    # 重新读取最好模型
    model.load_state_dict(torch.load(SAVE_PATH, map_location=device))
    y0 = model.sample(s0, x0)

    # 生成序列数据集
    x, y = generate_next_dataset(s0, x0, y0, 4, model, seq_len, epsilon)

    x = np.transpose(np.array(x, dtype=np.float32), axes=(1, 0, 2))
    y = np.transpose(np.array(y, dtype=np.int32), axes=(1, 0, 2))
    return s0, x, y, model


def get_components_of_change(model, s, x):
    sx = to_tensor(s, x)
    sx.requires_grad = True
    
    # 计算x对应的梯度
    prob = model(sx)
    loss = nn.BCELoss()(prob, torch.ones_like(prob))
    loss.backward()
    x_grad = (sx.grad[:, 1:]).detach().cpu().numpy()

    # 通过处理得到合适的大小
    def process(grad):
        if max(abs(grad)) != 0.0:
            while max(abs(grad)) < 1.:
                grad *= 10
        return grad
    x_grad = np.array(list(map(process, x_grad)))
    
    # 对于没有贷款的实例：sign=1表示实例向更利于贷款方向改变，sign=-1表示不变
    # 对于贷款的实例：sign=1表示实例按时还款向更利于贷款方向改变，
    # sign=-1表示没有按时还款，向更不利于贷款方向改变
    sign = 2 * torch.bernoulli(prob) - 1
    return x_grad, sign.detach().cpu().numpy()


def generate_next_dataset(s0, x0, y0, start, model, seq_len, epsilon):
    max_grad = 0.3
    max_val = 1.
    
    x, y = [x0], [y0]
    for i in range(seq_len - 1):
        nx = copy.deepcopy(x[-1])
        ny = copy.deepcopy(y[-1]).flatten()

        x_grad, sign = get_components_of_change(model, s0, nx)
        # 发放贷款情况：每一步的改变控制在[-2, 2]范围内
        idx = np.squeeze(ny == 1) & np.squeeze(s0 == 1)
        nx[idx] -= np.clip(1 * epsilon * sign[idx] * x_grad[idx], a_min=-max_grad, a_max=max_grad)
        idx = np.squeeze(ny == 1) & np.squeeze(s0 == 0)
        nx[idx] -= np.clip(3 * epsilon * sign[idx] * x_grad[idx], a_min=-max_grad, a_max=max_grad)

        idx = np.squeeze(ny == 0) & np.squeeze(s0 == 1)
        nx[idx] -= np.clip(2 * epsilon * sign[idx] * x_grad[idx], a_min=-max_grad, a_max=max_grad)
        idx = np.squeeze(ny == 0) & np.squeeze(s0 == 0)
        nx[idx] -= np.clip(3 * epsilon * sign[idx] * x_grad[idx], a_min=-max_grad, a_max=max_grad)

        nx = np.clip(nx, a_min=0., a_max=max_val)

        pred_ny = model.sample(s0, nx)
        x.append(nx)
        y.append(pred_ny)

    return x, y