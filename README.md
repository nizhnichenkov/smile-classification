# Smile Classification App
This repository contains code that does training, inferencing, and deployment of a CNN smile classifier that takes you from initial development to deployment.

## Notes & Requirements
- [Requirement] This repository only needs an installation of [Docker](https://www.docker.com/) on local machine - rest is done in a container.
- [Requirement] If you want to run the training pipeline (`./train.py`, train a CNN model from scratch) you need the training image data
    - Download and unzip image data from https://www.kaggle.com/datasets/jessicali9530/celeba-dataset into `/data/images/` (create folder if not exists); then build and run `Dockerfile.train` accordingly (shown below).
    - NOTE: This is only if you want to run `train` pipelines; for `inference` and `deploy` is not necessary.
- Python version used: `3.10.19`
- Project metadata and dependencies in `pyproject.toml`
    - `Docker build` will install these when creating an image.
- ML pipeline hyperparameters and settings in `./configs/train.yaml`, `./configs/inference.yaml` and `./configs/deploy.yaml`.
    - `./train.py` (training), `./predict.py` (inference), and `./src/api/app.py` (deployment) use these to fetch hyperparameters.

## Containerizing training pipeline
#### 0. Setup YAML file with hyperparameters and other configurations 
- YAML file under `configs/train.yaml`

#### 1. Build image
-   `docker build -f Dockerfile.train -t smile-training:latest .`
    - Installs Linux + Python + Python packages, and copies the data from local machine root directory (via `COPY . .` in Dockerfile.train)
    - NOTE: It ignores the files/directories written in `.dockerignore`

#### 2. Run training pipeline
- `docker run --rm -v $(pwd)/data:/app/data -v $(pwd)/artifacts:/app/artifacts smile-training:latest`
    - This runs the training pipeline and writes files to the local machine.
    - We use `-v <local_host_location>:<container_location>` to give the Docker container access to the data (in `$(pwd)/data`) and output (in `$(pwd)/artifacts`) folders that sit on the local machine - so Docker can read (the data) and write (model/metris/etc) to the local machine. This keeps the Docker image size small - as we don't copy over the data/outputs.
        - NOTE: `--rm` deletes the image after it has finished - so we need to save outputs outside of container.
        - NOTE: I use `$(pwd)/data` as the dataset resides in a folder named `data/` on my local machine. Same goes for `artifacts/` for output.

## Containerizing inference pipeline
#### 0. Setup YAML file with hyperparameters and other configurations
- YAML file under `configs/inference.yaml`

#### 1. Build image
- `docker build -f Dockerfile.predict -t smile-inference:latest .`

#### 2. Run inference pipeline
- `docker run --rm -v $(pwd)/data:/app/data -v $(pwd)/artifacts:/app/artifacts smile-inference:latest`


## Containerizing deployment inference pipeline
#### 0. Setup YAML file with hyperparams and other configuration
- YAML file under `configs/deploy.yaml`

#### 1. Build image
- `docker build -f Dockerfile.deploy -t smile-api:latest .`

#### 2. Run image
- `docker run --rm -v $(pwd)/artifacts:/app/artifacts -p 8000:8000 smile-api:latest`

#### 3. Go to Swagger UI
- Open up a web browser and type http://localhost:8000/docs

#### 4. Go to `predict` endpoint, upload an image file and click `Execute`
- NOTE: The image file should have dimensions 178x218 (width_x_height) and be of a persons face (profile). The image extension should be `.jpg/.jpeg`.
    - Can use sample images from `/data/test/`
- After `Execute`, you will receive a JSON output of the form `{"logit": float, "prediction": int}`
    - `logit`: float, raw probability
    - `prediction`: integer, (1) means "Smiling"; (0) means "Not Smiling".