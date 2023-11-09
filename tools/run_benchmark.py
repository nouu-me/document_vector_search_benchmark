import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import numpy.typing as npt
import pandas as pd
import yaml
from dvsb.data import DATASET_REGISTRY, Dataset
from dvsb.embedding import EMBEDDING_REGISTRY, Embedding
from dvsb.metric import METRIC_REGISTRY, Metric
from dvsb.relevance import RELEVANCE_REGISTRY, Relevance
from loguru import logger
from tqdm import tqdm


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


def load_embedding(embedding_config: dict) -> Embedding:
    embedding_config = embedding_config.copy()
    name = embedding_config["name"]
    del embedding_config["name"]
    return EMBEDDING_REGISTRY[name](**embedding_config)


def load_relevance(relevance_config: dict) -> Relevance:
    relevance_config = relevance_config.copy()
    name = relevance_config["name"]
    del relevance_config["name"]
    return RELEVANCE_REGISTRY[name](**relevance_config)


def load_metric(metric_config: dict) -> Metric:
    metric_config = metric_config.copy()
    name = metric_config["name"]
    del metric_config["name"]
    return METRIC_REGISTRY[name](**metric_config)


def make_batch(iterables: Iterable, batch_size: int) -> Iterable:
    batch = []
    for i, elem in enumerate(iterables, 1):
        batch.append(elem)
        if i % batch_size == 0:
            yield batch
            batch = []
    if batch:
        yield batch


def get_embeddings_cache_dir(dataset: Dataset, embedding: Embedding) -> Path:
    return Path(f"~/.dvsb/embeddings/{dataset.get_name()}/{embedding.get_name()}").expanduser()


def load_embeddings_from_cache(cache_dir: Path) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    logger.info(f"load embeddings from {str(cache_dir)}")
    query_embeddings = np.load(cache_dir / "query_embeddings.npy")
    context_embeddings = np.load(cache_dir / "context_embeddings.npy")
    return query_embeddings, context_embeddings


def cache_embeddings(
    query_embeddings: npt.NDArray[np.float64],
    context_embeddings: npt.NDArray[np.float64],
    cache_dir: Path,
) -> None:
    logger.info(f"save embeddings in {str(cache_dir)}")
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache_dir / "query_embeddings.npy", query_embeddings)
    np.save(cache_dir / "context_embeddings.npy", context_embeddings)


def main() -> None:
    parser = get_parser()
    args = parser.parse_args()
    config_file = f"configs/{args.name}.yml"
    with open(config_file) as fin:
        config = yaml.safe_load(fin)

    cache = not args.no_cache
    datasets = [load_dataset(dataset_config, cache=cache) for dataset_config in config["datasets"]]
    logger.info("datasets:")
    for dataset in datasets:
        logger.info("- " + json.dumps(dataset.get_stats()))

    embeddings = [load_embedding(embedding_config) for embedding_config in config["embeddings"]]
    logger.info("embeddings:")
    for embedding in embeddings:
        logger.info("- " + embedding.get_name())

    relevances = [load_relevance(relevance_config) for relevance_config in config["relevances"]]
    logger.info("relevances:")
    for relevance in relevances:
        logger.info("- " + relevance.get_name())

    metrics = [load_metric(metric_config) for metric_config in config["metrics"]]
    logger.info("metrics:")
    for metric in metrics:
        logger.info("- " + metric.get_name())

    records = []
    for dataset in datasets:
        logger.info(f"dataset: {dataset.get_name()}")
        for embedding in embeddings:
            logger.info(f"embedding: {embedding.get_name()}")
            cache_dir = get_embeddings_cache_dir(dataset, embedding)
            if cache and cache_dir.exists():
                query_embeddings, context_embeddings = load_embeddings_from_cache(cache_dir)
            else:
                batch_query_embeddings = []
                for queries in tqdm(list(make_batch(dataset.get_queries(), batch_size=8))):
                    batch_query_embeddings.append(embedding.get_embeddings(queries, mode="query"))
                query_embeddings = np.concatenate(batch_query_embeddings, axis=0)
                batch_context_embeddings = []
                for contexts in tqdm(list(make_batch(dataset.get_contexts(), batch_size=8))):
                    batch_context_embeddings.append(embedding.get_embeddings(contexts, mode="context"))
                context_embeddings = np.concatenate(batch_context_embeddings, axis=0)
                if cache:
                    cache_embeddings(query_embeddings, context_embeddings, cache_dir)
            for relevance in relevances:
                logger.info(f"relevance: {relevance.get_name()}")
                relevance_scores = relevance.compute(query_embeddings, context_embeddings)
                for metric in metrics:
                    metric_score = metric.compute(dataset.get_related_context_locations(), relevance_scores)
                    logger.info(f"{metric.get_name()}: {round(metric_score, 5)}")
                    records.append(
                        {
                            "dataset": dataset.get_name(),
                            "embedding": dataset.get_name(),
                            "relevance": relevance.get_name(),
                            "metric": metric.get_name(),
                        }
                    )
    result = pd.DataFrame(records)
    result.to_csv("result.csv", index=False)


if __name__ == "__main__":
    main()
