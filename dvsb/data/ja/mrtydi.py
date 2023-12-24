import json
from pathlib import Path

import datasets
import requests
import random
from dvsb.data.dataset import DATASET_REGISTRY, Dataset
from loguru import logger


@DATASET_REGISTRY.register
class MrTyDi(Dataset):
    def __init__(
        self,
        version: str = "1.0",
        split: str = "test",
        cache: bool = True,
        corpus_sample: int = 0,
    ) -> None:
        self.version = version
        self.split = split
        self.name = f"MrTyDi-v{version}-{split}"
        self.titles: list[str] = []
        self.queries: list[str] = []
        self.contexts: list[str] = []
        self.related_context_locations: list[list[int]] = []
        self.load_data(version, split, cache, corpus_sample)

    def get_name(self) -> str:
        return self.name

    def get_titles(self) -> list[str]:
        return self.titles

    def get_queries(self) -> list[str]:
        return self.queries

    def get_contexts(self) -> list[str]:
        return self.contexts

    def get_related_context_locations(self) -> list[list[int]]:
        return self.related_context_locations

    def get_cache_dir(self) -> Path:
        return Path(
            f"~/.dvsb/dataset/ja/mrtydi-v{self.version}-{self.split}"
        ).expanduser()

    def __save_cache(self) -> None:
        cache_dir = self.get_cache_dir()
        logger.info(f"saving {self.name} in {str(cache_dir)}")
        cache_dir.mkdir(parents=True, exist_ok=True)
        with open(cache_dir / "titles.json", "w") as fout:
            json.dump(self.titles, fout)
        with open(cache_dir / "queries.json", "w") as fout:
            json.dump(self.queries, fout)
        with open(cache_dir / "contexts.json", "w") as fout:
            json.dump(self.contexts, fout)
        with open(cache_dir / "related_context_locations.json", "w") as fout:
            json.dump(self.related_context_locations, fout)

    def __load_cache(self) -> None:
        cache_dir = self.get_cache_dir()
        logger.info(f"loading {self.name} from {str(cache_dir)}")
        with open(cache_dir / "titles.json") as fin:
            self.titles = json.load(fin)
        with open(cache_dir / "queries.json") as fin:
            self.queries = json.load(fin)
        with open(cache_dir / "contexts.json") as fin:
            self.contexts = json.load(fin)
        with open(cache_dir / "related_context_locations.json") as fin:
            self.related_context_locations = json.load(fin)

    def load_data(
        self, version: str, split: str, cache: bool, corpus_sample: int
    ) -> None:
        if cache:
            cache_dir = self.get_cache_dir()
            if cache_dir.exists():
                self.__load_cache()
                return
        logger.info(f"loading Mr.TyDi dataset (version: {version}, split: {split})")
        self.titles = []  # Unused
        self.queries = []
        self.contexts = []
        self.related_context_locations = []
        data = datasets.load_dataset("castorini/mr-tydi", "japanese", split)[split]
        context_to_index = {}  # Map context text to its index

        corpus = datasets.load_dataset("castorini/mr-tydi-corpus", "japanese")

        # Generate 3000 random negatives
        docids = corpus["train"]["docid"]
        if corpus_sample > 0:
            random.seed(42)
            passage_ids = set(random.sample(docids, corpus_sample))
        else:
            passage_ids = set()

        del docids

        for entry in data:
            for doc in entry["positive_passages"]:
                passage_ids.add(doc["docid"])
            for doc in entry["negative_passages"]:
                passage_ids.add(doc["docid"])

        corpus = corpus["train"].filter(lambda x: x["docid"] in passage_ids)
        passages_map = {x["docid"]: x["text"] for x in corpus}
        del corpus

        for d in data:
            cur_queries = [d["query"]]
            self.queries.extend(cur_queries)

            positive_indices = []
            for paragraph in d["positive_passages"]:
                cur_context = passages_map[paragraph["docid"]]
                if cur_context not in context_to_index:
                    context_to_index[cur_context] = len(self.contexts)
                    self.contexts.append(cur_context)
                positive_indices.append(context_to_index[cur_context])

            for paragraph in d["negative_passages"]:
                cur_context = passages_map[paragraph["docid"]]
                if cur_context not in context_to_index:
                    context_to_index[cur_context] = len(self.contexts)
                    self.contexts.append(cur_context)

            # Link each query to its relevant (positive) contexts
            self.related_context_locations.extend([positive_indices] * len(cur_queries))

        for _, passage in passages_map.items():
            if passage not in context_to_index:
                context_to_index[passage] = len(self.contexts)
                self.contexts.append(passage)

        del passages_map

        if cache:
            self.__save_cache()
