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

|    | dataset           | embedding                                                                 | relevance   |   Recall@1 |   Recall@3 |   Recall@5 |   Recall@10 |
|----|-------------------|---------------------------------------------------------------------------|-------------|------------|------------|------------|-------------|
|  0 | JSQuAD-v1.1-valid | E5Embedding-intfloat/multilingual-e5-large                                | Cosine      |   0.864926 |   0.952949 |   0.965781 |    0.977488 |
|  1 | JSQuAD-v1.1-valid | E5Embedding-intfloat/multilingual-e5-small                                | Cosine      |   0.840387 |   0.933814 |   0.95385  |    0.972985 |
|  2 | JSQuAD-v1.1-valid | E5Embedding-intfloat/multilingual-e5-base                                 | Cosine      |   0.838361 |   0.934039 |   0.954975 |    0.972535 |
|  3 | JSQuAD-v1.1-valid | VertexAITextEmbedding-textembedding-gecko-multilingual@001                | Cosine      |   0.780729 |   0.904322 |   0.932463 |    0.961054 |
|  4 | JSQuAD-v1.1-valid | VertexAITextEmbedding-textembedding-gecko-multilingual@latest             | Cosine      |   0.780729 |   0.904548 |   0.932238 |    0.960603 |
|  5 | JSQuAD-v1.1-valid | OpenAIEmbedding-text-embedding-ada-002                                    | Cosine      |   0.75394  |   0.874606 |   0.906799 |    0.937866 |
|  6 | JSQuAD-v1.1-valid | SonoisaSentenceBertJapanese-sonoisa/sentence-bert-base-ja-mean-tokens-v2  | Cosine      |   0.65421  |   0.810671 |   0.8629   |    0.914228 |
|  7 | JSQuAD-v1.1-valid | SonoisaSentenceLukeJapanese-sonoisa/sentence-luke-japanese-base-lite      | Cosine      |   0.652634 |   0.813147 |   0.861324 |    0.908825 |
|  8 | JSQuAD-v1.1-valid | SentenceTransformerEmbedding-pkshatech/GLuCoSE-base-ja                    | Cosine      |   0.644755 |   0.798064 |   0.846466 |    0.896668 |
|  9 | JSQuAD-v1.1-valid | SentenceTransformerEmbedding-sonoisa/sentence-bert-base-ja-mean-tokens-v2 | Cosine      |   0.639802 |   0.782981 |   0.841288 |    0.894867 |
| 10 | JSQuAD-v1.1-valid | SentenceTransformerEmbedding-cl-nagoya/sup-simcse-ja-base                 | Cosine      |   0.631923 |   0.792661 |   0.848942 |    0.897118 |
| 11 | JSQuAD-v1.1-valid | SentenceTransformerEmbedding-cl-nagoya/sup-simcse-ja-large                | Cosine      |   0.603107 |   0.776452 |   0.833408 |    0.889239 |
| 12 | JSQuAD-v1.1-valid | SentenceTransformerEmbedding-cl-nagoya/unsup-simcse-ja-large              | Cosine      |   0.594777 |   0.755966 |   0.8181   |    0.879559 |
| 13 | JSQuAD-v1.1-valid | SentenceTransformerEmbedding-cl-nagoya/unsup-simcse-ja-base               | Cosine      |   0.577217 |   0.746961 |   0.804142 |    0.870779 |
