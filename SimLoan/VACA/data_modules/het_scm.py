# source code: https://github.com/psanch21/VACA/
import os

import pytorch_lightning as pl
import torch
from sklearn import preprocessing
from torch_geometric.loader import DataLoader
from torch_geometric.utils import degree
from torchvision import transforms as transform_lib

from data_modules._scalers import MaskedTensorLikelihoodScaler
from data_modules._scalers import MaskedTensorStandardScaler
from datasets.transforms import ToTensor
from utils.constants import Cte


class HeterogeneousSCMDataModule(pl.LightningDataModule):
    name = 'het_scm'

    def __init__(
            self,
            data_dir: str = "./",
            dataset_name: str = 'toy',
            num_samples_tr: int = 10000,
            num_workers: int = 0,
            normalize: str = None,
            normalize_A: str = None,
            likelihood_names: str = None,
            seed: int = 42,
            batch_size: int = 32,
            lambda_: float = 0.05,
            equations_type: str = 'linear',
            device = None,
            *args,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.data_dir = data_dir

        self.equations_type = equations_type

        self.num_workers = num_workers
        self.normalize = normalize
        self.normalize_A = normalize_A
        self.scaler = None
        self.seed = seed
        self.batch_size = batch_size
        self.dataset_name = dataset_name

        self._shuffle_train = True

        self.device=device

        if dataset_name == Cte.TOY:
            root_dir = './'
            from datasets.toy import ToySCM

            self.train_dataset = ToySCM(root_dir=root_dir,
                                           split='train',
                                           num_samples_tr=num_samples_tr,
                                           lambda_=lambda_,
                                           transform=None
                                           )
            self.valid_dataset = ToySCM(root_dir=root_dir,
                                           split='valid',
                                           num_samples_tr=num_samples_tr,
                                           lambda_=lambda_,
                                           transform=None
                                           )
            self.test_dataset = ToySCM(root_dir=root_dir,
                                          split='test',
                                          num_samples_tr=num_samples_tr,
                                          lambda_=lambda_,
                                          transform=None
                                          )
            self.total_dataset = ToySCM(root_dir=root_dir,
                                          split='total',
                                          num_samples_tr=num_samples_tr,
                                          lambda_=lambda_,
                                          transform=None
                                          )
            
        elif dataset_name == Cte.TAIWAN:
            root_dir = './'
            from datasets.taiwan import TaiwanSCM

            self.train_dataset = TaiwanSCM(root_dir=root_dir,
                                           split='train',
                                           num_samples_tr=num_samples_tr,
                                           lambda_=lambda_,
                                           transform=None
                                           )
            self.valid_dataset = TaiwanSCM(root_dir=root_dir,
                                           split='valid',
                                           num_samples_tr=num_samples_tr,
                                           lambda_=lambda_,
                                           transform=None
                                           )
            self.test_dataset = TaiwanSCM(root_dir=root_dir,
                                          split='test',
                                          num_samples_tr=num_samples_tr,
                                          lambda_=lambda_,
                                          transform=None
                                          )
            self.total_dataset = TaiwanSCM(root_dir=root_dir,
                                          split='total',
                                          num_samples_tr=num_samples_tr,
                                          lambda_=lambda_,
                                          transform=None
                                          )
        else:
            raise NotImplementedError

        if dataset_name in [Cte.CHAIN, Cte.COLLIDER, Cte.TRIANGLE, Cte.MGRAPH, Cte.LOAN, Cte.ADULT, Cte.ADULT_AB_AGE,
                    Cte.ADULT_AB_EDU, Cte.ADULT_AB_HOURS, Cte.ADULT_AB_A_E, Cte.ADULT_AB_A_H, Cte.ADULT_AB_E_H,
                    Cte.ADULT_AB_A_E_H, Cte.LOAN_AB_LAMOUNT, Cte.LOAN_AB_LDURATION, Cte.LOAN_AB_INCOME,
                    Cte.LOAN_AB_SAVINGS, Cte.LOAN_AB_LA_LD, Cte.LOAN_AB_LA_I, Cte.LOAN_AB_LA_S, Cte.LOAN_AB_LD_I,
                    Cte.LOAN_AB_LD_S, Cte.LOAN_AB_I_S, Cte.LOAN_AB_LA_LD_I, Cte.LOAN_AB_LA_LD_S, Cte.LOAN_AB_LA_I_S,
                    Cte.LOAN_AB_LD_I_S, Cte.LOAN_AB_LA_LD_I_S]:
            root_dir = os.path.join(data_dir, 'VACA')

            self.train_dataset = dataset_fn(root_dir=root_dir,
                                            split='train',
                                            num_samples=int(0.7 * num_samples_tr),
                                            equations_type=equations_type,
                                            likelihood_names=likelihood_names,
                                            lambda_=lambda_,
                                            transform=None)

            self.valid_dataset = dataset_fn(root_dir=root_dir,
                                            split='valid',
                                            num_samples=int(0.2 * num_samples_tr),
                                            equations_type=equations_type,
                                            likelihood_names=likelihood_names,
                                            lambda_=lambda_,
                                            transform=None)

            self.test_dataset = dataset_fn(root_dir=root_dir,
                                           split='test',
                                           num_samples=int(0.1 * num_samples_tr),
                                           equations_type=equations_type,
                                           likelihood_names=likelihood_names,
                                           lambda_=lambda_,
                                           transform=None)
            
            self.total_dataset = dataset_fn(root_dir=root_dir,
                                             split='total',
                                             num_samples=num_samples_tr,
                                             equations_type=equations_type,
                                             likelihood_names=likelihood_names,
                                             lambda_=lambda_,
                                             transform=None)
            

    @property
    def likelihood_list(self):
        return self.train_dataset.likelihood_list

    @property
    def topological_nodes(self):
        topological_nodes, _ = self.train_dataset.get_topological_nodes_pa()
        return topological_nodes

    @property
    def topological_parents(self):
        _, topological_pa = self.train_dataset.get_topological_nodes_pa()
        return topological_pa

    @property
    def node_dim(self):
        return self.train_dataset.node_dim

    @property
    def num_nodes(self):
        return self.train_dataset.num_nodes

    @property
    def edge_dimension(self):
        return self.train_dataset.num_edges

    @property
    def is_heterogeneous(self):
        return self.train_dataset.is_heterogeneous

    def set_shuffle_train(self, value):
        self._shuffle_train = value

    def get_node_dim_image(self):
        keys = self.train_dataset.nodes_list
        is_image = self.train_dataset.node_is_image()
        node_dims = self.train_dataset.get_node_dimensions()
        node_dim_image = {}
        for i, key in enumerate(keys):
            node_dim_image[key] = (node_dims[i], is_image[i])

        return node_dim_image

    def get_random_train_sampler(self):
        self.train_dataset.set_transform(self._default_transforms())

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

    def prepare_data(self):

        self.train_dataset.prepare_data(normalize_A=self.normalize_A, add_self_loop=True)
        self.valid_dataset.prepare_data(normalize_A=self.normalize_A, add_self_loop=True)
        self.test_dataset.prepare_data(normalize_A=self.normalize_A, add_self_loop=True)
        self.total_dataset.prepare_data(normalize_A=self.normalize_A, add_self_loop=True)
        if self.normalize == 'std':
            self.scaler = MaskedTensorStandardScaler(list_dim_to_scale_x0=self.train_dataset.get_dim_to_scale_x0(),
                                                     list_dim_to_scale=self.train_dataset.get_dim_to_scale(),
                                                     total_num_dimensions=self.train_dataset.num_dimensions)
            self.scaler.fit(self.train_dataset.X0)
        elif self.normalize == 'lik':
            self.scaler = MaskedTensorLikelihoodScaler(likelihoods=self.train_dataset.likelihoods,
                                                       mask_x0=self.train_dataset.mask_X0[0, :])
            self.scaler.fit(self.train_dataset.X0)
        else:
            self.scaler = preprocessing.FunctionTransformer(func=lambda x: x,
                                                            inverse_func=lambda x: x)
            

    def train_dataloader(self):
        self.train_dataset.set_transform(self._default_transforms())
        loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self._shuffle_train,
            num_workers=19,
            drop_last=True,
            pin_memory=False
        )
        
        return loader

    def val_dataloader(self):
        self.valid_dataset.set_transform(self._default_transforms())

        loader = DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=19,
            drop_last=True,
            pin_memory=False
        )
    
        return loader

    def test_dataloader(self):
        self.test_dataset.set_transform(self._default_transforms())

        loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=19,
            drop_last=True,
            pin_memory=False
        )

        return loader
    
    def total_dataloader(self):
        self.total_dataset.set_transform(self._default_transforms())

        loader = DataLoader(
            self.total_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=19,
            drop_last=True,
            pin_memory=False
        )

        return loader

    def _default_transforms(self):
        if self.scaler is not None:
            return transform_lib.Compose(
                [lambda x: self.scaler.transform(x.reshape(1, self.train_dataset.total_num_dim_x0)), ToTensor()]
            )
        else:
            return ToTensor()

    def get_attributes_dict(self):
        return self.train_dataset.get_attributes_dict()

    def get_normalized_X(self, mode='test'):
        if mode == 'train':
            return self.scaler.transform(self.train_dataset.X.copy())
        elif mode == 'test':
            return self.scaler.transform(self.test_dataset.X.copy())
        elif mode == 'valid':
            return self.scaler.transform(self.valid_dataset.X.copy())
        else:
            raise NotImplementedError
