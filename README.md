# Document Vector Search Benchmark

Benchmark for Japanese document embedding and vector search

# setup

requirements:

- python >=3.9,<3.12
- poetry

required environmental variables:

- `OPENAI_API_KEY` (to use OpenAI's embedding API)
- `COHERE_API_KEY` (to use Cohere's embedding API)

to install all dependencies, perform

```bash
$ make install
```

# run benchmark

Benchmark settings can be controlled through a yaml file. The default configuration can be found in [configs/default.yml](./configs/default.yml).
You can use your own config file in the configs directry by setting the `DVSB_CONFIG_NAME` environmental variable (ex. `default`).

To run benchmark,

```
$ make run_benchmark
```

# result

| dataset          | embedding                                                   | relevance | metric    | value     |
|------------------|------------------------------------------------------------|-----------|-----------|-----------|
| JSQuAD-v1.1-valid | CohereEmbedding-embed-multilingual-v3.0                    | Cosine    | Recall@1  | 0.773075  |
| JSQuAD-v1.1-valid | CohereEmbedding-embed-multilingual-v3.0                    | Cosine    | Recall@3  | 0.902972  |
| JSQuAD-v1.1-valid | CohereEmbedding-embed-multilingual-v3.0                    | Cosine    | Recall@5  | 0.932238  |
| JSQuAD-v1.1-valid | CohereEmbedding-embed-multilingual-v3.0                    | Cosine    | Recall@10 | 0.959253  |
| JSQuAD-v1.1-valid | OpenAIEmbedding-text-embedding-ada-002                     | Cosine    | Recall@1  | 0.753940  |
| JSQuAD-v1.1-valid | OpenAIEmbedding-text-embedding-ada-002                     | Cosine    | Recall@3  | 0.874606  |
| JSQuAD-v1.1-valid | OpenAIEmbedding-text-embedding-ada-002                     | Cosine    | Recall@5  | 0.906799  |
| JSQuAD-v1.1-valid | OpenAIEmbedding-text-embedding-ada-002                     | Cosine    | Recall@10 | 0.937866  |
| JSQuAD-v1.1-valid | SonoisaSentenceBertJapanese-sonoisa/sentence-bert-base-ja-mean-tokens-v2 | Cosine    | Recall@1  | 0.654210  |
| JSQuAD-v1.1-valid | SonoisaSentenceBertJapanese-sonoisa/sentence-bert-base-ja-mean-tokens-v2 | Cosine    | Recall@3  | 0.810671  |
| JSQuAD-v1.1-valid | SonoisaSentenceBertJapanese-sonoisa/sentence-bert-base-ja-mean-tokens-v2 | Cosine    | Recall@5  | 0.862900  |
| JSQuAD-v1.1-valid | SonoisaSentenceBertJapanese-sonoisa/sentence-bert-base-ja-mean-tokens-v2 | Cosine    | Recall@10 | 0.914228  |
| JSQuAD-v1.1-valid | SonoisaSentenceLukeJapanese-sonoisa/sentence-luke-japanese-base-lite   | Cosine    | Recall@1  | 0.652634  |
| JSQuAD-v1.1-valid | SonoisaSentenceLukeJapanese-sonoisa/sentence-luke-japanese-base-lite   | Cosine    | Recall@3  | 0.813147  |
| JSQuAD-v1.1-valid | SonoisaSentenceLukeJapanese-sonoisa/sentence-luke-japanese-base-lite   | Cosine    | Recall@5  | 0.861324  |
| JSQuAD-v1.1-valid | SonoisaSentenceLukeJapanese-sonoisa/sentence-luke-japanese-base-lite   | Cosine    | Recall@10 | 0.908825  |
| JSQuAD-v1.1-valid | E5Embedding-intfloat/multilingual-e5-small                            | Cosine    | Recall@1  | 0.840387  |
| JSQuAD-v1.1-valid | E5Embedding-intfloat/multilingual-e5-small                            | Cosine    | Recall@3  | 0.933814  |
| JSQuAD-v1.1-valid | E5Embedding-intfloat/multilingual-e5-small                            | Cosine    | Recall@5  | 0.953850  |
| JSQuAD-v1.1-valid | E5Embedding-intfloat/multilingual-e5-small                            | Cosine    | Recall@10 | 0.972985  |
| JSQuAD-v1.1-valid | E5Embedding-intfloat/multilingual-e5-base                             | Cosine    | Recall@1  | 0.838361  |
| JSQuAD-v1.1-valid | E5Embedding-intfloat/multilingual-e5-base                             | Cosine    | Recall@3  | 0.934039  |
| JSQuAD-v1.1-valid | E5Embedding-intfloat/multilingual-e5-base                             | Cosine    | Recall@5  | 0.954975  |
| JSQuAD-v1.1-valid | E5Embedding-intfloat/multilingual-e5-base                             | Cosine    | Recall@10 | 0.972535  |
| JSQuAD-v1.1-valid | E5Embedding-intfloat/multilingual-e5-large                            | Cosine    | Recall@1  | 0.864926  |
| JSQuAD-v1.1-valid | E5Embedding-intfloat/multilingual-e5-large                            | Cosine    | Recall@3  | 0.952949  |
| JSQuAD-v1.1-valid | E5Embedding-intfloat/multilingual-e5-large                            | Cosine    | Recall@5  | 0.965781  |
| JSQuAD-v1.1-valid | E5Embedding-intfloat/multilingual-e5-large                            | Cosine    | Recall@10 | 0.977488  |
