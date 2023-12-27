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

from dvsb.late_interaction_retrievers import (
    LATE_INTERACTION_REGISTRY,
    LateInteractionRetriever,
)
from dvsb.late_interaction_retrievers._colbert import Run, RunConfig
from dvsb.metric import METRIC_REGISTRY, Metric
from dvsb.relevance import RELEVANCE_REGISTRY, Relevance
from loguru import logger
from tqdm import tqdm
import torch


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("run_benchmark")
    parser.add_argument(
        "-n", "--name", help="config name", required=False, default="default"
    )
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


def load_late_interaction_retriever(
    late_interaction_retriever_config: dict,
) -> LateInteractionRetriever:
    late_interaction_retriever_config = late_interaction_retriever_config.copy()
    name = late_interaction_retriever_config["name"]
    del late_interaction_retriever_config["name"]
    return LATE_INTERACTION_REGISTRY[name](**late_interaction_retriever_config)


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
    return Path(
        f"~/.dvsb/embeddings/{dataset.get_name()}/{embedding.get_name()}"
    ).expanduser()


def load_embeddings_from_cache(
    cache_dir: Path,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
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
    logger.info(f"load config from {config_file}")
    with open(config_file) as fin:
        config = yaml.safe_load(fin)

    cache = not args.no_cache
    datasets = [
        load_dataset(dataset_config, cache=cache)
        for dataset_config in config["datasets"]
    ]
    logger.info("datasets:")
    for dataset in datasets:
        logger.info("- " + json.dumps(dataset.get_stats()))

    embeddings = [
        load_embedding(embedding_config) for embedding_config in config["embeddings"]
    ]
    logger.info("embeddings:")
    for embedding in embeddings:
        logger.info("- " + embedding.get_name())

    if "late_interaction_retrievers" in config:
        late_interaction_retrievers = [
            load_late_interaction_retriever(late_interaction_retriever_config)
            for late_interaction_retriever_config in config[
                "late_interaction_retrievers"
            ]
        ]
    else:
        late_interaction_retrievers = []
    logger.info("late_interaction_retrievers:")
    for late_interaction_retriever in late_interaction_retrievers:
        logger.info("- " + late_interaction_retriever.get_name())

    relevances = [
        load_relevance(relevance_config) for relevance_config in config["relevances"]
    ]
    logger.info("relevances:")
    for relevance in relevances:
        logger.info("- " + relevance.get_name())

    metrics = [load_metric(metric_config) for metric_config in config["metrics"]]
    logger.info("metrics:")
    for metric in metrics:
        logger.info("- " + metric.get_name())

    records = []
    for embedding in embeddings:
        embedding.load(has_cuda=torch.cuda.is_available())
        logger.info(f"embedding: {embedding.get_name()}")
        for dataset in datasets:
            logger.info(f"dataset: {dataset.get_name()}")
            cache_dir = get_embeddings_cache_dir(dataset, embedding)
            if cache and cache_dir.exists():
                query_embeddings, context_embeddings = load_embeddings_from_cache(
                    cache_dir
                )
            else:
                batch_query_embeddings = []
                for queries in tqdm(
                    list(make_batch(dataset.get_queries(), batch_size=128))
                ):
                    batch_query_embeddings.append(
                        embedding.get_embeddings(queries, mode="query")
                    )
                query_embeddings = np.concatenate(batch_query_embeddings, axis=0)
                batch_context_embeddings = []
                for contexts in tqdm(
                    list(make_batch(dataset.get_contexts(), batch_size=128))
                ):
                    batch_context_embeddings.append(
                        embedding.get_embeddings(contexts, mode="context")
                    )
                context_embeddings = np.concatenate(batch_context_embeddings, axis=0)
                if cache:
                    cache_embeddings(query_embeddings, context_embeddings, cache_dir)
            for relevance in relevances:
                logger.info(f"relevance: {relevance.get_name()}")
                relevance_scores = relevance.compute(
                    query_embeddings, context_embeddings
                )
                metric_scores = {}
                for metric in metrics:
                    metric_score = metric.compute(
                        dataset.get_related_context_locations(), scores=relevance_scores
                    )
                    logger.info(f"{metric.get_name()}: {round(metric_score, 5)}")
                    metric_scores[metric.get_name()] = metric_score
                records.append(
                    {
                        "dataset": dataset.get_name(),
                        "embedding": embedding.get_name(),
                        "relevance": relevance.get_name(),
                    }
                    | metric_scores
                )
        del embedding

    for late_interaction_retriever in late_interaction_retrievers:
        model_name = late_interaction_retriever.get_name()
        with Run().context(
            RunConfig(
                nranks=late_interaction_retriever.n_gpu,
                experiment=f"benchmark_{model_name}",
            )
        ):
            for dataset in datasets:
                late_interaction_retriever.index(
                    index_name=f"{model_name}_{dataset.get_name()}",
                    documents=dataset.get_contexts(),
                )
                queries = dataset.get_queries()
                y_pred = late_interaction_retriever.query(queries, k=10)
                relevance = "Cosine"
                metric_scores = {}
                for metric in metrics:
                    metric_score = metric.compute(
                        dataset.get_related_context_locations(), y_pred=y_pred
                    )
                    logger.info(f"{metric.get_name()}: {round(metric_score, 5)}")
                    metric_scores[metric.get_name()] = metric_score
                records.append(
                    {
                        "dataset": dataset.get_name(),
                        "embedding": late_interaction_retriever.get_name(),
                        "relevance": relevance,
                    }
                    | metric_scores
                )
    result = pd.DataFrame(records)
    result.to_csv("result.csv", index=False)


if __name__ == "__main__":
    main()
