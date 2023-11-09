import argparse

from typing import Optional
import yaml

from dvsb.data import Dataset, DATASET_REGISTRY


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("run_benchmark")
    parser.add_argument("-n", "--name", help="config name", required=False, default="default")
    parser.add_argument("--no-cache", action="store_true")
    return parser


def load_dataset(dataset_config: dict, cache: bool) -> Dataset:
    dataset_config = dataset_config.copy() | {"cache": cache}
    name = dataset_config["name"]
    del dataset_config["name"]
    return DATASET_REGISTRY[name](**dataset_config)


def main() -> None:
    parser = get_parser()
    args = parser.parse_args()
    config_file = f"configs/{args.name}.yml"
    with open(config_file) as fin:
        config = yaml.safe_load(fin)

    cache = not args.no_cache
    datasets = [load_dataset(dataset_config, cache=cache) for dataset_config in config["datasets"]]
    for dataset in datasets:
        print(dataset.get_stats())


if __name__ == "__main__":
    main()
