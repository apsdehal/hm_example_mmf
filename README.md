# Hateful Memes Example using MMF

This repository serves as an example of how to use MMF as a library in your projects and build on top of it.

The example tries to replicate the model developed in DrivenData's [blog post](https://www.drivendata.co/blog/hateful-memes-benchmark/) on the Hateful Memes.


## Installation

Preferably, create your own conda environment before following the steps below:

```
git clone https://github.com/apsdehal/hm_example_mmf
cd hm_example_mmf
pip install -r requirements.txt
```

## Running

Run training with the following command on the Hateful Memes dataset:

```
MMF_USER_DIR="." mmf_run config="configs/experiments/defaults.yaml"  model=concat_vl dataset=hateful_memes training.num_workers=0
```

We set `training.num_workers=0` here to avoid memory leaks with fasttext.
Please follow [configuration](https://mmf.readthedocs.io/en/latest/notes/configuration_system.html) document to understand how to use MMF's configuration system to update parameters.

## Directory Structure

```
├── configs
│   ├── experiments
│   │   └── defaults.yaml
│   └── models
│       └── concat_vl.yaml
├── __init__.py
├── models
│   ├── concat_vl.py
├── processors
│   ├── processors.py
├── README.md
└── requirements.txt
```

Some notes:

1. Configs have been divided into `experiments` and `models` where experiments will contain training configs while models will contain model specific config we implmented.
2. `__init__.py` imports all of the relevant files so that MMF can find them. This is what `env.user_dir` actually looks for.
3. `models` directory contains our model implementation, in this case specifically `concat_vl`.
4. `processors` contains our project specific processors implementation, in this case, we implemented FastText processor for Sentence Vectors.

## Issues/Feedback/Questions

Please open up issues related to this repository directly on [MMF](https://github.com/facebookresearch/mmf/issues/new/choose).


