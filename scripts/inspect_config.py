"""Compose release configuration without importing CUDA, Ray, or the trainer."""
import argparse
from pathlib import Path
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


def read_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='repo')
    args, overrides = parser.parse_known_args()
    directory = Path(__file__).resolve().parents[1] / 'verl/trainer/config'
    with initialize_config_dir(config_dir=str(directory), version_base=None):
        return compose(config_name=args.config, overrides=overrides)


if __name__ == '__main__':
    print(OmegaConf.to_yaml(read_config(), resolve=True))
