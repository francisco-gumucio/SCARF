import os

from datasets._heterogeneous import HeterogeneousSCM
from datasets._adjacency import Adjacency
from utils.distributions import *
import pandas as pd
from torch_geometric.loader import DataLoader
from torch_geometric.utils import degree
from datasets.transforms import ToTensor
import torch
import math

class TaiwanSCM(HeterogeneousSCM):

    def __init__(self, root_dir,
                 split: str = 'train',
                 num_samples_tr: int = 630,
                 lambda_: float = 0.05,
                 transform=None,
                 device = None,
                 model = None,
                 ):
        
        assert split in ['train', 'valid', 'test', 'total']

        self.df = pd.read_csv('taiwan.csv', header=0, index_col=0)
        self.name = 'taiwan'
        self.split = split
        self.num_samples_tr = num_samples_tr
        self.device = device
        self.model = model

        structural_eq = None

        noises_distr = None

        self.categorical_nodes = []

        super().__init__(root_dir=root_dir,
                         transform=transform,
                         nodes_to_intervene=['LIMIT_BAL_2', 'PAY_AMT1_2'],
                         structural_eq=structural_eq,
                         noises_distr=noises_distr,
                         nodes_list=['SEX', 'LIMIT_BAL_1', 'PAY_AMT1_1', 'PAY_AMT2_1',
                                     'Y_1', 'LIMIT_BAL_2', 'PAY_AMT1_2', 'PAY_AMT2_2',
                                     'Y_2'],
                         adj_edges={'SEX': ['LIMIT_BAL_1', 'PAY_AMT2_1', 'LIMIT_BAL_2', 'PAY_AMT2_2'],
                                    'LIMIT_BAL_1': ['PAY_AMT1_1', 'PAY_AMT2_1', 'Y_1', 'LIMIT_BAL_2'],
                                    'PAY_AMT1_1': ['PAY_AMT2_1', 'Y_1', 'PAY_AMT1_2'],
                                    'PAY_AMT2_1': ['Y_1', 'PAY_AMT2_2'],
                                    'Y_1': ['LIMIT_BAL_2', 'PAY_AMT2_2', 'PAY_AMT1_2'],
                                    'LIMIT_BAL_2': ['PAY_AMT1_2', 'PAY_AMT2_2', 'Y_2'],
                                    'PAY_AMT1_2': ['PAY_AMT2_2', 'Y_2'],
                                    'PAY_AMT2_2': ['Y_2']
                                    },
                         lambda_=lambda_
                         )

    @property
    def likelihoods(self):
        likelihoods_tmp = []

        for i, lik_name in enumerate(self.nodes_list):  # Iterate over nodes
            if self.nodes_list[i] in ['SEX', 'Y_1', 'Y_2']:
                dim = 1
                lik_name = 'b'
            else:
                dim = 1
                lik_name = 'n'
            likelihoods_tmp.append([self._get_lik(lik_name,
                                                  dim=dim,
                                                  normalize='dim')])
        return likelihoods_tmp
    
    @property
    def edge_dimension(self):
        edge_num = 0

        for key in self.adj_edges:
            edge_num += len(self.adj_edges[key])

        return edge_num + self.num_nodes
        

    def _create_data(self):
        X_vals = self.df.values
        if self.split == 'train':
            self.X = X_vals[:int(self.num_samples_tr*0.7)]
        elif self.split == 'valid':
            self.X = X_vals[int(self.num_samples_tr*0.7):int(self.num_samples_tr*0.8)]
        elif self.split == 'test':
            self.X = X_vals[int(self.num_samples_tr*0.8):]
        elif self.split == 'total':
            self.X = X_vals
        self.U = np.zeros([self.X.shape[0], self.X.shape[1]])


    def set_transform(self, transform):
        self.transform = transform

    def prepare_adj(self, normalize_A=None, add_self_loop=True):
        assert normalize_A is None, 'Normalization on A is not implemented'
        self.normalize_A = normalize_A
        self.add_self_loop = add_self_loop

        if add_self_loop:
            SCM_adj = np.eye(self.num_nodes, self.num_nodes)
        else:
            SCM_adj = np.zeros([self.num_nodes, self.num_nodes])

        for node_i, children_i in self.adj_edges.items():
            row_idx = self.nodes_list.index(node_i)
            # print('\nnode_i', node_i)
            for child_j in children_i:
                # print('\tchild_j', child_j)
                SCM_adj[row_idx, self.nodes_list.index(child_j)] = 1
        # Create Adjacency Object
        self.dag = SCM_adj
        self.adj_object = Adjacency(SCM_adj)

    def node_is_image(self):
        return [False for _ in range(len(self.nodes_list))]

    def get_random_train_sampler(self):
        # self.train_dataset.set_transform(self._default_transforms())

        def tmp_fn(num_samples):
            dataloader = DataLoader(self.train_dataset, batch_size=num_samples, shuffle=True)
            return next(iter(dataloader))

        return tmp_fn

    def get_deg(self, indegree=True, bincount=False):
        d_list = []
        idx = 1 if indegree else 0
        for data in self.train_dataset:
            d = degree(data.edge_index[idx], num_nodes=data.num_nodes, dtype=torch.long)
            d_list.append(d)

        d = torch.cat(d_list)
        if bincount:
            deg = torch.bincount(d, minlength=d.numel())
        else:
            deg = d

        return deg.float()
    
    def get_edge_dim(self):
        edges = 0
        for key in self.adj_edges:
            edges += len(self.adj_edges[key])
        edges += self.num_nodes
        return edges
    
    def get_attributes_dict(self):
        attributes_dict = {
        'fair_attributes': ['s'],
        'unfair attributes': ['x1', 'z1', 'y1', 'x2', 'z2', 'y2'],
        'immutable_attributes': ['s'],
        'mutable attributes': ['x1', 'z1', 'y1', 'x2', 'z2', 'y2']
        }
        return attributes_dict
    
    def sample_obs(self, obs_id, parents_dict=None, n_samples=1, u=None):
        '''
        Only possible if the true Structural Eauations are known
        f = self.structural_eq[f'x{obs_id}']
        if u is None:
            u = np.array(self.noises_distr[f'u{obs_id}'].sample(n_samples))

        if not isinstance(parents_dict, dict):
            return f(u), u
        else:
            return f(u, **parents_dict), u
        '''
        assert obs_id < len(self.nodes_list)
        node_name = self.nodes_list[obs_id]
        lik = self.likelihoods[obs_id][0]
        f = self.structural_eq[node_name]
        u_is_none = u is None
        if u_is_none:
            u = self._sample_noise(node_name, n_samples)
        x = f(u, **parents_dict) if isinstance(parents_dict, dict) else f(u)
        x = x.astype(np.float32)
        if node_name in self.categorical_nodes:
            x_out = np.zeros([x.shape[0], lik.domain_size])
            x = x.astype(np.int32)
            for i in range(x.shape[0]):
                x_out[i, x[i, 0]] = 1

            x = x_out.copy().astype(np.float32)
        return x, u


