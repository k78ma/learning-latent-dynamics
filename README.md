# Embed to Control

## Instructions
Install requirements and activate virtual environment with:
```
uv sync
source .venv/bin/activate
```

Generate trajectories with:
```
python -m sim.mujoco_data
```

Train with:
```
python train.py --config <config_name>
```

![env_1](media/env_1.jpeg)