import os

from datasets._heterogeneous import HeterogeneousSCM
from datasets._adjacency import Adjacency
from utils.distributions import *
import pandas as pd
from torch_geometric.data import DataLoader
from torch_geometric.utils import degree
import torch

class ToySCM(HeterogeneousSCM):

    def __init__(self, root_dir,
                 X, 
                 split: str = 'train',
                 num_samples_tr: int = 630,
                 lambda_: float = 0.05,
                 transform=None,
                 ):
        assert split in ['train', 'valid', 'test']

        self.X = X
        self.name = 'toy'
        self.split = split
        self.num_samples_tr = num_samples_tr

        if self.split == 'train':
            self.train_dataset = X
        elif self.split == 'valid':
            self.valid_dataset = X
        else:
            self.test_dataset = X

    

        super().__init__(root_dir=root_dir,
                         transform=transform,
                         nodes_to_intervene=['x1'],
                         structural_eq=None,
                         noises_distr=None,
                         nodes_list=['s', 'x1', 'z1', 'y1',
                                     'x2', 'z2', 'y2'],
                         adj_edges={'s': ['x1', 'z1', 'x2', 'z2'],
                                    'x1': ['z1', 'y1', 'x2'],
                                    'z1': ['y1', 'z2'],
                                    'y1': ['x2', 'z2'],
                                    'x2': ['z2', 'y2'],
                                    'z2': ['y2']
                                    },
                         lambda_=lambda_,
                         )

    @property
    def likelihoods(self):
        likelihoods_tmp = []

        for i, lik_name in enumerate(self.nodes_list):  # Iterate over nodes
            if self.nodes_list[i] in ['is_exciting', 'at_least_1_teacher_referred_donor', 'fully_funded',
                                      'at_least_1_green_donation', 'great_chat',
                                      'three_or_more_non_teacher_referred_donors',
                                      'one_non_teacher_referred_donor_giving_100_plus',
                                      'donation_from_thoughtful_donor']:
                dim = 1
                lik_name = 'b'
            else:
                dim = 1
                lik_name = 'd'
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

