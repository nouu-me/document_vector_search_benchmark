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

|    | Model                                                                     |   JSQuAD-v1.1-valid recall@1 |   JSQuAD-v1.1-valid recall@3 |   JSQuAD-v1.1-valid recall@5 |   JSQuAD-v1.1-valid recall@10 | MrTyDi-v1.0-test recall@1   | MrTyDi-v1.0-test recall@3   | MrTyDi-v1.0-test recall@5   | MrTyDi-v1.0-test recall@10   | MIRACL-v1.0-dev recall@1   | MIRACL-v1.0-dev recall@3   | MIRACL-v1.0-dev recall@5   | MIRACL-v1.0-dev recall@10   |
|----|---------------------------------------------------------------------------|------------------------------|------------------------------|------------------------------|-------------------------------|-----------------------------|-----------------------------|-----------------------------|------------------------------|----------------------------|----------------------------|----------------------------|-----------------------------|
|  0 | E5Embedding-intfloat/multilingual-e5-large                                |                     0.86493  |                     0.95295  |                     0.96578  |                      0.97749  | 0.61157                     | 0.81273                     | 0.85625                     | 0.89306                      | 0.3348                     | 0.52176                    | 0.59992                    | 0.69678                     |
|  1 | E5Embedding-intfloat/multilingual-e5-small                                |                     0.84039  |                     0.93381  |                     0.95385  |                      0.97299  | 0.57106                     | 0.76366                     | 0.79421                     | 0.84444                      | 0.28925                    | 0.46448                    | 0.54013                    | 0.64014                     |
|  2 | ColBERTRetriever-bclavie/JaColBERT                                                         |                     0.90567  |                     0.9588   |                     0.96826  |                      0.97794  | 0.56134                     | 0.74444                     | 0.78079                     | 0.82083                      | 0.27699                    | 0.46369                    | 0.54606                    | 0.64494                     |
|  3 | E5Embedding-intfloat/multilingual-e5-base                                 |                     0.83836  |                     0.93404  |                     0.95498  |                      0.97253  | 0.57917                     | 0.77708                     | 0.81505                     | 0.85718                      | 0.31226                    | 0.48195                    | 0.55337                    | 0.63197                     |
|  4 | VertexAITextEmbedding-textembedding-gecko-multilingual@001                |                     0.780729 |                     0.904322 |                     0.932463 |                      0.961054 | ---                         | ---                         | ---                         | ---                          | ---                        | ---                        | ---                        | ---                         |
|  5 | VertexAITextEmbedding-textembedding-gecko-multilingual@latest             |                     0.780729 |                     0.904548 |                     0.932238 |                      0.960603 | ---                         | ---                         | ---                         | ---                          | ---                        | ---                        | ---                        | ---                         |
|  6 | OpenAIEmbedding-text-embedding-ada-002                                    |                     0.75394  |                     0.874606 |                     0.906799 |                      0.937866 | ---                         | ---                         | ---                         | ---                          | ---                        | ---                        | ---                        | ---                         |
|  7 | SentenceTransformerEmbedding-bclavie/fio-base-japanese-v0.1               |                     0.69968  |                     0.84151  |                     0.87933  |                      0.92436  | 0.3912                      | 0.58241                     | 0.64907                     | 0.71157                      | 0.14853                    | 0.27872                    | 0.35786                    | 0.46295                     |
|  8 | SonoisaSentenceBertJapanese-sonoisa/sentence-bert-base-ja-mean-tokens-v2  |                     0.65421  |                     0.81067  |                     0.8629   |                      0.91423  | 0.30116                     | 0.4875                      | 0.54884                     | 0.61157                      | 0.07305                    | 0.17223                    | 0.224                      | 0.33838                     |
|  9 | SonoisaSentenceLukeJapanese-sonoisa/sentence-luke-japanese-base-lite      |                     0.65263  |                     0.81315  |                     0.86132  |                      0.90882  | 0.24282                     | 0.39722                     | 0.47037                     | 0.54352                      | 0.07116                    | 0.15504                    | 0.21505                    | 0.30468                     |
| 10 | SentenceTransformerEmbedding-pkshatech/GLuCoSE-base-ja                    |                     0.64475  |                     0.79806  |                     0.84647  |                      0.89667  | 0.46968                     | 0.61713                     | 0.66921                     | 0.73542                      | 0.2124                     | 0.36852                    | 0.43225                    | 0.51465                     |
| 11 | SentenceTransformerEmbedding-sonoisa/sentence-bert-base-ja-mean-tokens-v2 |                     0.6398   |                     0.78298  |                     0.84129  |                      0.89487  | 0.30417                     | 0.45417                     | 0.525                       | 0.59861                      | 0.09431                    | 0.1966                     | 0.25401                    | 0.3543                      |
| 12 | SentenceTransformerEmbedding-cl-nagoya/sup-simcse-ja-base                 |                     0.63192  |                     0.79266  |                     0.84894  |                      0.89712  | 0.28866                     | 0.45417                     | 0.51435                     | 0.58032                      | 0.05196                    | 0.13299                    | 0.17717                    | 0.26391                     |
| 13 | SentenceTransformerEmbedding-cl-nagoya/sup-simcse-ja-large                |                     0.60311  |                     0.77645  |                     0.83341  |                      0.88924  | 0.29329                     | 0.45856                     | 0.51736                     | 0.58102                      | 0.07263                    | 0.15921                    | 0.21258                    | 0.29529                     |
| 14 | SentenceTransformerEmbedding-cl-nagoya/unsup-simcse-ja-large              |                     0.59478  |                     0.75597  |                     0.8181   |                      0.87956  | 0.21574                     | 0.36134                     | 0.40556                     | 0.48287                      | 0.05774                    | 0.11392                    | 0.15792                    | 0.23963                     |
| 15 | SentenceTransformerEmbedding-cl-nagoya/unsup-simcse-ja-base               |                     0.57722  |                     0.74696  |                     0.80414  |                      0.87078  | 0.18495                     | 0.31528                     | 0.38264                     | 0.46875                      | 0.04061                    | 0.11226                    | 0.15183                    | 0.22628                     |