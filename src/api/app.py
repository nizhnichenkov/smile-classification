import yaml
import torch
import shutil
import io
import time

from pathlib import Path
from fastapi import FastAPI
from fastapi import UploadFile
from PIL import Image
from torchvision import transforms
from datetime import datetime

from src.inference.predictor import Predictor
from src.utils.device import get_device
from src.utils.logging import setup_logging

app = FastAPI(
    title='Smile Classification API'
)

# ------------------
# startup event - load model once
# ------------------
@app.on_event("startup")
def startup():
    # NOTE: using "app.state.<var_name>" saves the variables in the app memory
    #       so they can be reused globally across other API endpoints

    # ------------------
    # save deployment run
    # ------------------
    # create unique folder for each deployment run
    run_id = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )
    
    run_dir = Path(
        f'artifacts/deploy/{run_id}'
    )
    
    run_dir.mkdir(
        parents=True,
        exist_ok=True
    )
    
    # save yaml file to know deployment config
    shutil.copy(
        'configs/deploy.yaml',
        run_dir / 'deploy.yaml'
    )

    # save Dockerfile to know containerization config
    shutil.copy(
        'Dockerfile.deploy',
        run_dir / 'Dockerfile.deploy' 
    )

    # save pyproject.toml to know python version and dependencies
    shutil.copy(
        'pyproject.toml',
        run_dir / 'pyproject.toml'
    )

    # ----------------
    # load config
    # ----------------
    with open("configs/deploy.yaml") as f:
        app.state.cfg = yaml.safe_load(f)
    
    # ----------------
    # setup logger
    # ----------------
    app.state.logger = setup_logging(logging_path=run_dir / app.state.cfg['log_file'])

    app.state.logger.info("="*80)
    app.state.logger.info("STARTING INFERENCE SERVICE")
    app.state.logger.info("="*80)
    app.state.logger.info("\n")

    # ----------------
    # get device
    # ----------------
    app.state.device = get_device(logger=app.state.logger)

    # ----------------
    # load model
    # ----------------
    app.state.logger.info(f"Model path: {app.state.cfg['model_path']}")

    app.state.model = torch.load(app.state.cfg['model_path'], weights_only=False)
    app.state.model.eval() # disable dropout

    # ----------------
    # model to predictor
    # ----------------
    app.state.predictor = Predictor(
        model=app.state.model,
        device=app.state.device,
        threshold=app.state.cfg['threshold']
    )

    # ----------------
    # define transformations
    # ----------------
    app.state.transform_eval = transforms.Compose([
        transforms.CenterCrop([178, 178]),
        transforms.Resize([64, 64]),
        transforms.ToTensor()
    ])

    app.state.logger.info('Model loaded successfully')



# ----------------
# prediction endpoint
# ----------------
@app.post("/predict")
async def predict(file: UploadFile):

    try:

        app.state.logger.info(
            f'Prediction request received: '
            f'{file.filename}'
        )
    
        # read bytes
        contents = await file.read()
    
        # convert to PIL
        image = Image.open(io.BytesIO(contents))
    
        # apply same transformation for val/test subsets used in training
        x = app.state.transform_eval(image)
    
        # add one more dimension at idx 0
        #  Predictor().predict_batch() expects batches in shape [[sample1], [sample2], ...]
        #  our single image for predicting is of shape [sample] - we add one more dimension
        #  to get it in shape - [[sample]]
        x = x.unsqueeze(0)
    
        start = time.perf_counter() # calc time for prediction
        logits, pred = app.state.predictor.predict_batch(x)
        elapsed = time.perf_counter() - start
    
        app.state.logger.info(
            f'Prediction completed | '
            f'file={file.filename} | '
            f'probability={logits:.4f} | '
            f'prediction={pred} | '
            f'latency={elapsed:.3f}s'
        )
    
        return {
            "logit": logits.item(),
            "prediction": int(pred)
        }

    except Exception as e:
        app.state.logger.exception(
            f'Prediction failed for '
            f'{file.filename}'
        )



# ----------------
# health check endpoint (useful for Docker/Kubernetes/Load Balancers)
# ----------------
@app.get("/health")
def health():
    app.state.logger.info("Health check: OK")

    return{
        "status": "healthy"
    }