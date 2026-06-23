import jittor as jt
jt.flags.use_cuda = 1

from omegaconf import OmegaConf
from tqdm import tqdm
from typing import Dict, List

import argparse
import numpy as np
import os
import random

from src.data.asset import Asset, Exporter
from src.data.dataset import DatasetConfig, DatasetConfig, PCDatasetModule
from src.data.transform import Transform
from src.model.parse import get_model
from src.system.parse import get_system, get_writer

def load(task: str, path: str) -> Dict:
    if path.endswith('.yaml'):
        path = path.removesuffix('.yaml')
    path += '.yaml'
    print(f"\033[92mload {task} config: {path}\033[0m")
    return OmegaConf.to_container(OmegaConf.load(path)) # type: ignore

def debug_fn(data: PCDatasetModule):
    train_dataloader = data.train_dataloader()
    assert train_dataloader is not None, "train_dataloader is None, cannot debug"
    for batch in tqdm(train_dataloader):
        batch: List[Asset]
        # for asset in batch:
        #     Exporter.export_obj(asset.sampled_vertices, "debug.obj")
        #     Exporter.export_obj(asset.sampled_vertices_noisy, "debug_noisy.obj")
        #     exit()

if __name__ == "__main__":
        
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--seed", type=int, required=False, default=123)
    args = parser.parse_args()
    
    # seed all
    jt.set_global_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    task = load('task', args.task)
    mode = task['mode']
    assert mode in ['train', 'predict', 'debug', 'validate']
    components = task['components']
    
    # get train/validate/predict data
    data_config = load('data', os.path.join('configs/data', components['data']))
    
    # get train dataset
    _train_dataset_config = data_config.get('train_dataset', None)
    if _train_dataset_config is not None:
        train_dataset_config = DatasetConfig.parse(**_train_dataset_config)
    else:
        train_dataset_config = None
    
    # get validate dataset
    _validate_dataset_config = data_config.get('validate_dataset', None)
    if _validate_dataset_config is not None:
        validate_dataset_config = DatasetConfig.parse(**_validate_dataset_config).split_by_cls()
    else:
        validate_dataset_config = None
        
    # get predict dataset
    _predict_dataset_config = data_config.get('predict_dataset', None)
    if _predict_dataset_config is not None:
        predict_dataset_config = DatasetConfig.parse(**_predict_dataset_config).split_by_cls()
    else:
        predict_dataset_config = None
    
    # get transform
    transform_config = load('transform', os.path.join('configs/transform', components['transform']))

    # get model
    model_config = components.get('model', None)
    if model_config is None:
        model = None
    else:
        model_config = load('model', os.path.join('configs/model', model_config))
        model = get_model(model_config=model_config, transform_config=transform_config)
    
    train_transform = (Transform.parse(**transform_config.get('train_transform', {}))) if model is None else model.get_train_transform()
    validate_transform = (Transform.parse(**transform_config.get('validate_transform', {}))) if model is None else model.get_validate_transform()
    predict_transform = (Transform.parse(**transform_config.get('predict_transform', {}))) if model is None else model.get_predict_transform()
    dataset_module = PCDatasetModule(
        process_fn=None if model is None else model._process_fn,
        train_dataset_config=train_dataset_config,
        validate_dataset_config=validate_dataset_config,
        predict_dataset_config=predict_dataset_config,
        train_transform=train_transform,
        validate_transform=validate_transform,
        predict_transform=predict_transform,
        debug=task.get('debug', False),
    )
    
    optimizer_config = task.get('optimizer', None)
    loss_config = task.get('loss', None)
    trainer_config = task.get('trainer', None)
    
    # load ckpt
    load_ckpt = task.get('load_ckpt', None)
    
    if load_ckpt is not None and model is not None:
        model.load(load_ckpt)
    
    # get writer
    writer_config = task.get('writer', None)
    
    # get system
    system_config = components.get('system', None)
    if system_config is not None:
        system_config = load('system', os.path.join('configs/system', system_config))
        system = get_system(
            dataset_module=dataset_module,
            model=model,
            optimizer_config=optimizer_config,
            loss_config=loss_config,
            trainer_config=trainer_config,
            writer=get_writer(**writer_config) if writer_config is not None else None,
            **system_config,
        )
    else:
        system = None
    
    if mode == 'debug':
        debug_fn(data=dataset_module)
    elif mode == 'train':
        assert system is not None, "system is None, cannot train"
        system.train()
    elif mode == 'predict':
        assert system is not None, "system is None, cannot predict"
        system.predict()
    else:
        assert 0