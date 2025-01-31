import os
import numpy as np
from datasets.toy import ToySCM
from models.vaca import VACA
import utils.args_parser  as argtools
from data_modules.het_scm import HeterogeneousSCMDataModule


def generate_vaca(length, batch_size, device, is_training):

        scm = ToySCM(None, 'train', length)
        model_file = os.path.join('VACA', '_params', 'model_vaca.yaml')
        trainer_file = os.path.join('VACA', '_params', 'trainer.yaml')

        cfg = argtools.parse_args(model_file)
        cfg.update(argtools.parse_args(trainer_file))

        cfg['dataset'] = {
        'name': 'toy',
        'params1': {},
        'params2': {}
        }

        cfg['dataset']['params1']['batch_size'] = batch_size
        cfg['dataset']['params1']['num_samples_tr'] = length
        cfg['dataset']['params1']['num_workers'] = 0
        cfg['dataset']['params1']['equations_type'] = 'non-linear'
        cfg['dataset']['params1']['normalize'] = 'lik'
        cfg['dataset']['params1']['lambda_'] = 0.05
        cfg['dataset']['params1']['data_dir'] = '../Data'
        cfg['dataset']['params1']['device'] = device

        dataset_params = cfg['dataset']['params1']
        data_module = HeterogeneousSCMDataModule(**dataset_params)

        data_module.train_dataset._create_data()
        data_module.valid_dataset._create_data()
        data_module.test_dataset._create_data()
        data_module.prepare_data()
        data_module.train_dataloader()
        data_module.test_dataloader()
        data_module.val_dataloader()


        cfg['model']['params']['is_heterogeneous'] = scm.is_heterogeneous
        cfg['model']['params']['likelihood_x'] = scm.likelihood_list

        cfg['model']['params']['num_nodes'] = scm.num_nodes
        cfg['model']['params']['edge_dim'] = scm.edge_dimension

        model_params = cfg['model']['params']

        model_vaca = VACA(**model_params)
        model_vaca.set_random_train_sampler(data_module.get_random_train_sampler())

        model_vaca.set_optim_params(optim_params=cfg['optimizer'],
                                sched_params=cfg['scheduler'])

        from VACA.models._evaluator import MyEvaluator

        evaluator = MyEvaluator(model=model_vaca,
                                intervention_list=data_module.train_dataset.get_intervention_list(),
                                scaler=data_module.scaler
                                )
        model_vaca.set_my_evaluator(evaluator=evaluator)

        assert evaluator is not None
        del cfg['trainer']['progress_bar_refresh_rate']
        del cfg['trainer']['flush_logs_every_n_steps']
        del cfg['trainer']['terminate_on_nan']
        del cfg['trainer']['auto_select_gpus']
        del cfg['trainer']['weights_summary']
        cfg['trainer']['enable_model_summary'] = False
        del cfg['trainer']['gpus']
        del cfg['trainer']['track_grad_norm']

        from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
        from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
        import pytorch_lightning as pl

        yaml_file = ''
        if yaml_file == '':
                save_dir = argtools.mkdir(os.path.join(cfg['root_dir'],
                                                argtools.get_experiment_folder(cfg),
                                                str(cfg['seed'])))
        else:
                save_dir = os.path.join(yaml_file.split('/')[:-1])
        print(f'Save dir: {save_dir}')
        # trainer = pl.Trainer(**cfg['model'])
        logger = TensorBoardLogger(save_dir=save_dir, name='logs', default_hp_metric=False)
        out = logger.log_hyperparams(argtools.flatten_cfg(cfg))

        save_dir_ckpt = argtools.mkdir(os.path.join('vaca_tools', save_dir, 'ckpt'))
        ckpt_file = argtools.newest(save_dir_ckpt)
        callbacks = []

        if is_training == 1:
        
                checkpoint = ModelCheckpoint(monitor=model_vaca.monitor(),
                                                mode=model_vaca.monitor_mode(),
                                                save_top_k=1,
                                                save_last=True,
                                                filename='checkpoint-{epoch:02d}',
                                                dirpath=save_dir_ckpt)

                callbacks = [checkpoint]

                if cfg['early_stopping']:
                        early_stopping = EarlyStopping(model_vaca.monitor(), mode=model_vaca.monitor_mode(), min_delta=0.0,
                                                        patience=50)
                        callbacks.append(early_stopping)

                if ckpt_file is not None:
                        print(f'Loading model training: {ckpt_file}')
                        trainer = pl.Trainer(logger=logger, callbacks=callbacks, **cfg['trainer'], devices='auto', accelerator='gpu')
                else:

                        trainer = pl.Trainer(logger=logger, callbacks=callbacks, **cfg['trainer'], devices='auto', accelerator='gpu')
                
                trainer.fit(model=model_vaca, train_dataloaders = data_module.train_dataloader(), val_dataloaders = data_module.val_dataloader())
                trainer.validate(ckpt_path=ckpt_file, dataloaders=data_module.val_dataloader())
                trainer.test(ckpt_path=ckpt_file, dataloaders=data_module.test_dataloader())
                argtools.save_yaml(cfg, file_path=os.path.join(save_dir, 'hparams_full.yaml'))
        
        else:
                model_vaca = VACA.load_from_checkpoint(ckpt_file, **model_params)


        evaluator.set_model(model_vaca)
        model_vaca.set_my_evaluator(evaluator=evaluator)
        model_parameters = filter(lambda p: p.requires_grad, model_vaca.parameters())
        params = int(sum([np.prod(p.size()) for p in model_parameters]))
        model_vaca.eval()
        model_vaca.freeze()

        return model_vaca, data_module