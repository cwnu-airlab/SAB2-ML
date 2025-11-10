## Index
1. Directory Structure
2. Experiment Settings
3. Running the Code

## Directory Structure

### Tree
```bash
├── configs
│   ├── config.yaml
│   └── task
├── data
│   └── GoEmotions
├── requirements
│   ├── requirements_hydra.txt
│   └── requirements_transformers.txt
├── run.py
└── src
    ├── agent
    ├── datamodule
    ├── eval
    ├── models
    └── tokenizer
```

### Directory Description
#### configs 
Experiment settings (global defaults in config.yaml); task-specific overrides in configs/task/.

#### data
##### GoEmotions
-  GoEmotions multi-label dataset (labels and train/val/test splits used for training & evaluation).

#### requirements : Environment specs.

- requirements_hydra.txt: core + Hydra-based config stack

- requirements_transformers.txt: Transformers-based stack

#### run.py
Entry point to launch experiments.

#### src
Training pipeline (Trainer) and core modules:
- agent : Experiment runner/orchestration utilities.

- datamodule : Dataset & DataLoader setup.

- eval : Evaluation scripts and metrics.

- models : Model architectures and training loop logic.

- tokenizer : Tokenizer setup and preprocessing utilities.

## Experiment Settings
#### Environment
- **Python version:** 3.6  
- **Frameworks:** PyTorch 1.10.1, Transformers 4.18.0  
- **Configuration Management:** Hydra 1.2  
- **Hardware:** NVIDIA RTX TITAN (24 GB) × 1  
- **Operating System:** Ubuntu 18.04 LTS

#### example
Create a Virtual Environment

```
# using virtualenv (as requested)
virtualenv venv --python=python3.6
source venv/bin/activate        # Linux/macOS
```
Install pip

```
pip install -r requirements/requirements_transformers.txt
pip install -r requirements/requirements_hydra.txt
```

## Running the Code

```
python run.py task=multi_label/sab save_dir=checkpoint/go_emo/
```

