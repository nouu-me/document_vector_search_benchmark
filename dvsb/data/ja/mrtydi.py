import json
import os
import random
from pathlib import Path
from typing import Literal, Optional

import datasets
import requests
from dvsb.data.dataset import DATASET_REGISTRY, Dataset
from loguru import logger


@DATASET_REGISTRY.register
class MrTyDi(Dataset):
    def __init__(
        self,
        version: str = "1.0",
        split: str = "test",
        cache: bool = True,
        sampling_method: Optional[Literal["hard_negative", "random"]] = None,
        corpus_sample: int = 0,
    ) -> None:
        self.version = version
        self.split = split
        self.name = f"MrTyDi-v{version}-{split}"
        self.titles: list[str] = []
        self.queries: list[str] = []
        self.contexts: list[str] = []
        self.related_context_locations: list[list[int]] = []
        self.load_data(version, split, cache, sampling_method, corpus_sample)

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
        root_cache_dir = Path(os.getenv("DVSB_CACHE_DIR", "~/.dvsb"))
        return (root_cache_dir / "dataset" / "ja" / f"mrtydi-v{self.version}-{self.split}").expanduser()

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
        self,
        version: str,
        split: str,
        cache: bool,
        sampling_method: Optional[Literal["hard_negative", "random"]] = None,
        corpus_sample: int = 0,
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

        if sampling_method is not None:
            corpus = datasets.load_dataset("castorini/mr-tydi-corpus", "japanese")["train"]
            if sampling_method == "hard_negative":
                if corpus_sample < 0:
                    corpus_sample = 100
                hard_neg_urls = "https://ben.clavie.eu/retrieval/tydi_ja_bm25_top1000.json"
                hard_negs = requests.get(hard_neg_urls).json()
                print("SAMPLED :", corpus_sample)
                passage_ids = set([docid for x in hard_negs.values() for docid in x[:corpus_sample]])

            elif sampling_method == "random":
                if corpus_sample > 0:
                    random.seed(42)
                    passage_ids = set(random.sample(corpus["docid"], corpus_sample))

            passages_map = {x["docid"]: {"title": x["title"], "text": x["text"]} for x in corpus}
        else:
            raise ValueError("Must specify a valid sampling method.")

        for d in data:
            cur_queries = [d["query"]]
            self.queries.extend(cur_queries)

            positive_indices = []
            for paragraph in d["positive_passages"]:
                cur_context_map = passages_map[paragraph["docid"]]
                cur_context = f"{cur_context_map['title']} {cur_context_map['text']}"
                if cur_context not in context_to_index:
                    context_to_index[cur_context] = len(self.contexts)
                    self.contexts.append(cur_context)
                positive_indices.append(context_to_index[cur_context])

            for paragraph in d["negative_passages"]:
                cur_context_map = passages_map[paragraph["docid"]]
                cur_context = f"{cur_context_map['title']} {cur_context_map['text']}"
                if cur_context not in context_to_index:
                    context_to_index[cur_context] = len(self.contexts)
                    self.contexts.append(cur_context)

            # Link each query to its relevant (positive) contexts
            self.related_context_locations.extend([positive_indices] * len(cur_queries))

        for pid, passage_map in passages_map.items():
            passage = f"{passage_map['title']} {passage_map['text']}"
            if pid not in passage_ids:
                continue
            if passage not in context_to_index:
                context_to_index[passage] = len(self.contexts)
                self.contexts.append(passage)

        del passages_map

        if cache:
            self.__save_cache()
