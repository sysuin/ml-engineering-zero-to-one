"""
Foresight's training as a pipeline. Chapter 21 writes it.

    make train MODEL=renewal        python -m foresight.pipeline.run

model.py      v0.6 as one scikit-learn object
features.py   one definition of Chapter 4's columns, and the skew test
evaluation.py the run's metrics and the model card's numbers
artifact.py   the model, its card and its manifest, in one folder
registry.py   staged, production, archived
settings.py   configs/<model>.toml
run.py        the training job, step by step
"""
