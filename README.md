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


|    | Model                                                                     |   Recall@1 JSQuAD-v1.1-valid |   Recall@3 JSQuAD-v1.1-valid |   Recall@5 JSQuAD-v1.1-valid |   Recall@10 JSQuAD-v1.1-valid |   Recall@3 MIRACL-v1.0-dev |   Recall@5 MIRACL-v1.0-dev |   Recall@10 MIRACL-v1.0-dev |   Recall@100 MIRACL-v1.0-dev |
|----|---------------------------------------------------------------------------|------------------------------|------------------------------|------------------------------|-------------------------------|----------------------------|----------------------------|-----------------------------|------------------------------|
|  0 | ColBERTRetriever-bclavie/JaColBERTv2                                      |                     0.920756 |                     0.967807 |                     0.976587 |                      0.982665 |                  0.622286  |                   0.713644 |                    0.82424  |                     0.970855 |
|  1 | SentenceTransformerEmbedding-BAAI/bge-m3                                  |                     0.849842 |                     0.939217 |                     0.958577 |                      0.975912 |                  0.686322  |                   0.769516 |                    0.85376  |                     0.980984 |
|  2 | E5Embedding-intfloat/multilingual-e5-large                                |                     0.864926 |                     0.952949 |                     0.965781 |                      0.977488 |                  0.658555  |                   0.741041 |                    0.835273 |                     0.982559 |
|  3 | ColBERTRetriever-bclavie/JaColBERT                                        |                     0.911301 |                     0.961054 |                     0.970059 |                      0.977262 |                  0.555464  |                   0.639378 |                    0.748628 |                     0.933103 |
|  4 | E5Embedding-intfloat/multilingual-e5-base                                 |                     0.838361 |                     0.934039 |                     0.954975 |                      0.972535 |                  0.612025  |                   0.687829 |                    0.798884 |                     0.976868 |
|  5 | E5Embedding-intfloat/multilingual-e5-small                                |                     0.840387 |                     0.933814 |                     0.95385  |                      0.972985 |                  0.59911   |                   0.688636 |                    0.783391 |                     0.972148 |
|  6 | VertexAITextEmbedding-textembedding-gecko-multilingual@001                |                     0.780729 |                     0.904322 |                     0.932463 |                      0.961054 |                  N/A  |                   N/A |                    N/A |                     N/A |
|  7 | VertexAITextEmbedding-textembedding-gecko-multilingual@latest             |                     0.780729 |                     0.904548 |                     0.932238 |                      0.960603 |                  N/A  |                   N/A |                    N/A |                     N/A |
|  8 | OpenAIEmbedding-text-embedding-ada-002                                    |                     0.75394  |                     0.874606 |                     0.906799 |                      0.937866 |                  N/A  |                   N/A |                    N/A |                     N/A |
|  9 | SonoisaSentenceLukeJapanese-sonoisa/sentence-luke-japanese-base-lite      |                     0.652634 |                     0.813147 |                     0.861324 |                      0.908825 |                  0.144617  |                   0.211484 |                    0.297427 |                     0.622732 |
|  10 | SonoisaSentenceBertJapanese-sonoisa/sentence-bert-base-ja-mean-tokens-v2  |                     0.65421  |                     0.810671 |                     0.8629   |                      0.914228 |                  0.170388  |                   0.221441 |                    0.314252 |                     0.660344 |
|  11 | SentenceTransformerEmbedding-pkshatech/GLuCoSE-base-ja                    |                     0.644755 |                     0.798064 |                     0.846466 |                      0.896668 |                  0.472039  |                   0.546962 |                    0.645746 |                     0.861757 |
|  12 | SentenceTransformerEmbedding-cl-nagoya/sup-simcse-ja-base                 |                     0.631923 |                     0.792661 |                     0.848942 |                      0.897118 |                  0.136905  |                   0.185814 |                    0.26734  |                     0.590408 |
| 13 | SentenceTransformerEmbedding-sonoisa/sentence-bert-base-ja-mean-tokens-v2 |                     0.639802 |                     0.782981 |                     0.841288 |                      0.894867 |                  0.203452  |                   0.26437  |                    0.357611 |                     0.708596 |
| 14 | SentenceTransformerEmbedding-cl-nagoya/sup-simcse-ja-large                |                     0.603107 |                     0.776452 |                     0.833408 |                      0.889239 |                  0.154618  |                   0.202999 |                    0.293636 |                     0.584585 |
| 15 | SentenceTransformerEmbedding-cl-nagoya/unsup-simcse-ja-large              |                     0.594777 |                     0.755966 |                     0.8181   |                      0.879559 |                  0.102252  |                   0.142104 |                    0.218686 |                     0.52806  |
| 16 | SentenceTransformerEmbedding-cl-nagoya/unsup-simcse-ja-base               |                     0.577217 |                     0.746961 |                     0.804142 |                      0.870779 |                  0.0963266 |                   0.121559 |                    0.195299 |                     0.500001 |