import yaml
import pandas as pd
import pathlib
import torch
import torch.nn as nn
import shutil
import random
import numpy as np

from src.datasets.celeba_dataset import ImageData
from src.models.cnn import create_model
from src.utils.logging import setup_logging
from src.training.trainer import Trainer
from src.utils.device import get_device

from torch.utils.data import Subset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from pathlib import Path
from datetime import datetime


def main():
    # create unique folder for each experiment
    # and save hyperparam yaml file
    run_id = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )
    
    run_dir = Path(
        f'artifacts/train/{run_id}'
    )
    
    run_dir.mkdir(
        parents=True,
        exist_ok=True
    )
    
    shutil.copy(
        'configs/train.yaml',
        run_dir / 'train.yaml'
    )
    
    
    # --------------------
    # load config
    # --------------------
    with open("configs/train.yaml") as f:
        cfg = yaml.safe_load(f)
    
    
    # --------------------
    # setup seed (reproducibility)
    # --------------------
    seed = cfg['reproducibility']['random_state']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    
    # --------------------
    # setup config
    # --------------------
    log_file_name = 'train.log'
    logger = setup_logging(logging_path=run_dir / log_file_name)
    
    # --------------------
    # log run directory
    # --------------------
    logger.info('=' * 80)
    logger.info('RUN DIRECTORY')
    logger.info('=' * 80)
    logger.info(str(run_dir))
    logger.info('\n')
    
    
    # --------------------
    # log hyperparameters
    # --------------------
    logger.info("=" * 80)
    logger.info("TRAINING CONFIGURATION")
    logger.info("=" * 80)
    
    for key, value in cfg.items():
        logger.info(f"{key}: {value}")
    logger.info('\n')
    
    # --------------------
    # create model
    # --------------------
    model = create_model(dropout_rate=cfg['model']['dropout'])
    
    device = get_device(logger=logger)
    
    model.to(device) # put model on GPU (if available)
    
    
    # --------------------
    # log architecture
    # --------------------
    logger.info("=" * 80)
    logger.info("MODEL ARCHITECTURE")
    logger.info("=" * 80)
    logger.info(str(model))
    logger.info('\n')
    
    
    # --------------------
    # log parameters
    # --------------------
    logger.info("=" * 80)
    logger.info("PARAMETERS")
    logger.info("=" * 80)
    total_params = sum( p.numel() for p in model.parameters() )
    trainable_params = sum( p.numel() for p in model.parameters() if p.requires_grad )
    
    logger.info(
        f"Total parameters: "
        f"{total_params:,}"
    )
    
    logger.info(
        f"Trainable parameters: "
        f"{trainable_params:,}"
    )
    logger.info('\n')
    
    
    # --------------------
    # load dataset
    # --------------------
    
    # obtain target labels
    data_attr = pd.read_csv(cfg['dataset']['attributes_file'])
    targets = data_attr[cfg['dataset']['target_name']].values
    
    # convert targets into required range for training loop
    # we are doing boolen outputs 0/1
    # currently, not smiling = -1; smiling = 1
    # convert to => not smiling = 0; smiling = 1
    targets[targets == -1] = 0
    
    
    # obtain image data
    datadir_path = pathlib.Path(cfg['dataset']['image_dir'])
    data_file_list = sorted( str(path) for path in datadir_path.glob('*.jpg'))
    
    # split data into train/test/val (80/10/10) stratified
    indices = list(range(len(data_file_list)))
    train_idx, temp_idx = train_test_split(
        indices,
        test_size=0.2,
        stratify=targets,
        random_state=cfg['reproducibility']['random_state']
    )
    
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.5,
        stratify=targets[temp_idx],
        random_state=cfg['reproducibility']['random_state']
    )
    
    # extract names of image files for each data subset
    train_files = [
        data_file_list[i] for i in train_idx
    ]
    
    valid_files = [
        data_file_list[i] for i in val_idx
    ]
    
    test_files = [
        data_file_list[i] for i in test_idx
    ]
    
    # extract targets for each data subset
    train_labels = targets[train_idx]
    valid_labels = targets[val_idx]
    test_labels = targets[test_idx]
    
    # setup transformations
    transform_train = transforms.Compose([
        transforms.RandomCrop([178, 178]),
        transforms.RandomHorizontalFlip(),
        transforms.Resize([64, 64]),
        transforms.ToTensor()
    ])
    
    transform_eval = transforms.Compose([
        transforms.CenterCrop([178, 178]),
        transforms.Resize([64, 64]),
        transforms.ToTensor()
    ])
    
    
    # create datasets with applied transformations for each data subset
    train_ds = ImageData(
        file_list=train_files,
        labels=train_labels,
        transform=transform_train
    )
    
    valid_ds = ImageData(
        file_list=valid_files,
        labels=valid_labels,
        transform=transform_eval
    )
    
    test_ds = ImageData(
        file_list=test_files,
        labels=test_labels,
        transform=transform_eval
    )
    
    # safeguard train/val/test size
    train_size = min( cfg['dataset']['train_size'], len(train_ds) )
    valid_size = min( cfg['dataset']['valid_size'], len(valid_ds) )
    test_size = min( cfg['dataset']['test_size'], len(test_ds) )
    
    # take only a small sample size for train/val/test
    train_ds_subset = Subset(train_ds, torch.arange(train_size))
    valid_ds_subset = Subset(valid_ds, torch.arange(valid_size))
    test_ds_subset = Subset(test_ds, torch.arange(test_size))
    
    # create data loaders
    # train_dl = DataLoader(train_ds_subset, batch_size = cfg['training']['batch_size'], shuffle=True, num_workers = cfg['dataset']['num_workers'])
    train_dl = DataLoader(train_ds_subset, batch_size = cfg['training']['batch_size'], shuffle=True)
    # valid_dl = DataLoader(valid_ds_subset, batch_size = cfg['training']['batch_size'], shuffle=False, num_workers = cfg['dataset']['num_workers'])
    valid_dl = DataLoader(valid_ds_subset, batch_size = cfg['training']['batch_size'], shuffle=False)
    # test_dl = DataLoader(test_ds_subset, batch_size = cfg['training']['batch_size'], shuffle=False, num_workers = cfg['dataset']['num_workers'])
    test_dl = DataLoader(test_ds_subset, batch_size = cfg['training']['batch_size'], shuffle=False)
    
    
    # --------------------
    # log dataset sizes
    # --------------------
    logger.info("=" * 80)
    logger.info("DATASET SIZE(S)")
    logger.info("=" * 80)
    
    logger.info(
        f'Full dataset size: '
        f'{len(data_file_list)}'
    )
    
    logger.info(
        f"Training samples: "
        f"{len(train_dl.dataset)}"
    )
    
    logger.info(
        f"Validation samples: "
        f"{len(valid_dl.dataset)}"
    )
    
    logger.info(
        f"Test samples: "
        f"{len(test_dl.dataset)}"
    )
    logger.info('\n')
    
    
    
    # --------------------
    # log data class dist
    # --------------------
    logger.info('=' * 80)
    logger.info('CLASS DISTRIBUTION')
    logger.info('=' * 80)
    logger.info(
        f'Train dataset:\n'
        f'{pd.Series(train_labels).value_counts()}'
    )
    
    logger.info(
        f'Validation dataset:\n'
        f'{pd.Series(valid_labels).value_counts()}'
    )
    
    logger.info(
        f'Test dataset:\n'
        f'{pd.Series(test_labels).value_counts()}'
    )
    logger.info('\n')
    
    
    # --------------------
    # setup optimizer
    # --------------------
    optimizer_type = cfg['optimizer']['type']
    
    if optimizer_type == 'Adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg['optimizer']['learning_rate']
        )
    
    elif optimizer_type == 'SGD':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=cfg['optimizer']['learning_rate']
        )
    else:
        raise ValueError(
            f'Unsupported optimizer: {optimizer_type}'
        )
    
    logger.info('=' * 80)
    logger.info('OPTIMIZER')
    logger.info('=' * 80)
    logger.info(str(optimizer))
    logger.info('\n')
    
    
    # --------------------
    # setup loss fn
    # --------------------
    loss_type = cfg['loss']['type']
    
    if loss_type == 'BCELoss':
        loss_fn = nn.BCELoss()
    elif loss_type == 'BCEWithLogitsLoss':
        loss_fn = nn.BCEWithLogitsLoss()
    else:
        raise ValueError(
            f'Unsupported loss: {loss_type}'
        )
    
    logger.info('=' * 80)
    logger.info('LOSS FUNCTION')
    logger.info('=' * 80)
    logger.info(str(loss_fn))
    logger.info('\n')
    
    
    # --------------------
    # create trainer & train
    # --------------------
    
    # for reproducibility (and not overwriting)
    # create unique path for model (full) and weights to save
    # model_path = run_dir / cfg['checkpointing']['model_name']
    # weights_path = run_dir / cfg['checkpointing']['model_weights']
    
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        out_path = str(run_dir),
        device=device,
        logger=logger,
        cfg=cfg,
    )
    
    history = trainer.train(
        train_dl=train_dl,
        valid_dl=valid_dl,
    )
    
    
    # --------------------
    # log final metrics
    # --------------------
    logger.info('=' * 80)
    logger.info('FINAL RESULTS')
    logger.info('=' * 80)
    
    logger.info(
        f"Best Train Accuracy: "
        f"{max(history['train_acc']):.4f}"
    )
    
    logger.info(
        f"Best Validation Accuracy: "
        f"{max(history['val_acc']):.4f}"
    )
    
    logger.info(
        f"Final Validation Loss: "
        f"{history['val_loss'][-1]:.4f}"
    )
    
    
    # --------------------
    # save training history
    # --------------------
    history_df = pd.DataFrame(history)
    # history_df.to_csv(f"{cfg['logging']['metrics']}", index=False)
    history_df.to_csv(run_dir / 'history.csv', index=False)
    
    logger.info(
        f"Results saved to: "
        f"{str(run_dir / 'history.csv')}"
    )
    logger.info('\n')



if __name__ == "__main__":
    main()