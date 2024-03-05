import os

from datasets._heterogeneous import HeterogeneousSCM
from utils.distributions import *
import pandas as pd

class ToySCM(HeterogeneousSCM):

    def __init__(self, root_dir,
                 X, 
                 split: str = 'train',
                 num_samples_tr: int = 630,
                 lambda_: float = 0.05,
                 transform=None,
                 ):
        assert split in ['train', 'valid', 'test']

        self.name = 'toy'
        self.split = split
        self.num_samples_tr = num_samples_tr

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

    def node_is_image(self):
        return [False for _ in range(len(self.nodes_list))]
