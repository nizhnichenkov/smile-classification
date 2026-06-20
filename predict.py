import yaml
import torch
import pandas as pd
import random
import numpy as np
import pathlib
import shutil
import json

from src.inference.predictor import Predictor
from src.utils.device import get_device
from src.utils.logging import setup_logging
from src.datasets.celeba_dataset import ImageData

from torchvision import transforms
from torch.utils.data import DataLoader
from datetime import datetime
from pathlib import Path


def main():

    # create unique folder for each experiment
    # and save hyperparam yaml file
    run_id = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )
    
    run_dir = Path(
        f'artifacts/inference/{run_id}'
    )
    
    run_dir.mkdir(
        parents=True,
        exist_ok=True
    )
    
    # save yaml file to know inference config
    shutil.copy(
        'configs/inference.yaml',
        run_dir / 'inference.yaml'
    )


    # --------------------
    # load config
    # --------------------
    with open('configs/inference.yaml') as f:
        cfg = yaml.safe_load(f)
    

    # --------------------
    # setup config
    # --------------------
    log_file_name = 'inference.log'
    logger = setup_logging(logging_path=run_dir / log_file_name)


    # --------------------
    # setup seed (reproducibility)
    # --------------------
    seed = cfg['reproducibility']['random_state']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    
    # --------------------
    # load model
    # --------------------
    model = torch.load(cfg['model']['path'], weights_only=False)


    # --------------------
    # get device 
    # --------------------
    device = get_device(logger=logger)
    model.to(device) # put model on GPU (if available)


    # --------------------
    # load dataset
    # --------------------

    # obtain target labels
    data_attr = pd.read_csv(cfg['dataset']['attributes_file'])

    # sort alphabetically (that's how we filtered our data)
    data_attr = data_attr.sort_values(f"{cfg['dataset']['name_column']}")

    # take last 20,000 labels as these correspond to our test data samples
    targets = data_attr[cfg['dataset']['target_column']].tail(20000).values
    
    # convert targets into required range for training loop
    # we are doing boolen outputs 0/1
    # currently, not smiling = -1; smiling = 1
    # convert to => not smiling = 0; smiling = 1
    targets[targets == -1] = 0

    # obtain image data
    datadir_path = pathlib.Path(cfg['dataset']['data_dir'])
    data_file_list = sorted( str(path) for path in datadir_path.glob('*.jpg'))

    # define transformations for test data
    # random transformations were applied to training data
    # to keep it consistent - we apply to test data as well
    transform_eval = transforms.Compose([
        transforms.CenterCrop([178, 178]),
        transforms.Resize([64, 64]),
        transforms.ToTensor()
    ])

    test_ds = ImageData(
        file_list=data_file_list,
        labels=targets,
        transform=transform_eval
    )

    test_dl = DataLoader(test_ds, batch_size=cfg['model']['batch_size'], shuffle=False)


    # --------------------
    # run inference
    # --------------------

    # create predictor
    predictor = Predictor(
        model=model,
        device=device,
        threshold=cfg['model']['threshold']
    )

    results = []
    sample_idx = 0
    batch_count = 0

    logger.info('='*80)
    logger.info('INFERENCE STARTED')
    logger.info('='*80)
    logger.info(f"Model: {cfg['model']['path']}")
    logger.info(f"Batch size: {cfg['model']['batch_size']}")
    logger.info(f"Threshold: {cfg['model']['threshold']}")
    logger.info(f"Dataset size: {len(test_dl.dataset)}")


    for x_batch, y_batch in test_dl:
        # inference (forward pass)
        logits, preds = predictor.predict_batch(x_batch)

        # obtain file names of data samples in batch
        file_names = data_file_list[sample_idx : sample_idx + cfg['model']['batch_size']]

        for fname, target, logit, pred in zip(
            file_names,
            y_batch,
            logits,
            preds
        ):
            results.append({
                'file': fname,
                'ground_truth': int(target), # 1.0/0.0 => 1/0
                'logit': logit.item(), # tensor(float_value) => float_value
                'pred': int(pred), # 1.0/0.0 => 1/0
            })

        sample_idx += cfg['model']['batch_size']

        batch_count += 1
        if batch_count % 50 == 0:
            logger.info(f'Batch {batch_count} processed...')

    
    # --------------------
    # save predictions
    # --------------------
    df = pd.DataFrame(results)
    df.to_csv(run_dir / 'predictions.csv', index=False)
    logger.info(f"Predictions saved to {run_dir / 'predictions.csv'}")


    # --------------------
    # build & save metadata 
    # --------------------
    metadata = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "model_path": cfg['model']['path'],
        "dataset_path": cfg['dataset']['data_dir'],
        "dataset_size": len(test_dl.dataset),
        "batch_size": cfg['model']['batch_size'],
        "threshold": cfg['model']['threshold'],
        "device": str(device)
    }

    # metrics
    correct = (df['ground_truth'] == df['pred']).sum() # number of correct predictions
    accuracy = (correct / len(df))                     # accuracy

    metadata['accuracy'] = float(accuracy)
    metadata['correct_predictions'] = int(correct)
    metadata['incorrect_predictions'] = int(len(df) - correct)

    with open(run_dir / "prediction_metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)
    
    logger.info(f"Prediction metadata saved to {run_dir / 'prediction_metadata.json'}")







if __name__ == "__main__":
    main()