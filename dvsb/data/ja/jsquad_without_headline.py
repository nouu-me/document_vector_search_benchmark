import json
import os
from pathlib import Path

import requests
from dvsb.data.dataset import DATASET_REGISTRY, Dataset
from loguru import logger


@DATASET_REGISTRY.register
class JSQuADWithoutHeadline(Dataset):
    URLS = {
        "1.1": {
            "train": "https://github.com/yahoojapan/JGLUE/raw/main/datasets/jsquad-v1.1/train-v1.1.json",
            "valid": "https://github.com/yahoojapan/JGLUE/raw/main/datasets/jsquad-v1.1/valid-v1.1.json",
        }
    }

    def __init__(self, version: str = "1.1", split: str = "train", cache: bool = True) -> None:
        self.version = version
        self.split = split
        self.name = f"JSQuAD-v{version}-without-headline-{split}"
        self.titles: list[str] = []
        self.queries: list[str] = []
        self.contexts: list[str] = []
        self.related_context_locations: list[list[int]] = []
        self.load_data(version, split, cache)

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
        return (
            root_cache_dir / "dataset" / "ja" / f"jsquad-v{self.version}-without-headline-{self.split}"
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

    def load_data(self, version: str, split: str, cache: bool) -> None:
        if cache:
            cache_dir = self.get_cache_dir()
            if cache_dir.exists():
                self.__load_cache()
                return
        logger.info(f"loading JSQuAD-without-headline dataset (version: {version}, split: {split})")
        self.titles = []
        self.queries = []
        self.contexts = []
        self.related_context_locations = []
        url = self.URLS[version][split]
        logger.info(f"downloading data from {url}")
        res = requests.get(url)
        data = json.loads(res.content)["data"]
        for d in data:
            title = d["title"]
            for paragraph in d["paragraphs"]:
                cur_queries = [qa["question"] for qa in paragraph["qas"]]
                cur_context = paragraph["context"]
                first_sep_pos = cur_context.find("[SEP]")
                if first_sep_pos >= 0:
                    cur_context = cur_context[first_sep_pos + len("[SEP]") :].lstrip()
                self.titles.extend([title] * len(cur_queries))
                self.queries.extend(cur_queries)
                context_location = len(self.contexts)
                self.contexts.append(cur_context)
                self.related_context_locations.extend([[context_location]] * len(cur_queries))
        if cache:
            self.__save_cache()
