# Clusterly - Semantic Idea Mapper

Often we have many thoughts about what things we need to do now.
Especially, when it's weekend and we wish to be productive.
However, we often struggle to pick the most prioritized tasks,
that will make us enjoy and thrive at the same time.

Clusterly will help you to automatically organize your thoughts in the vector space,
group, band and label the groups using NLP techniques.

The following libraries were used: `nltp`, `spacy`, `sklearn`, `numpy`, `matplotlib`, `tkinter`, `sentence_transformers`, `umap`, `loguru`.

## The process

1. **Preprocessing** (lemmatization and normalization)
2. **Embedding** - `all-MiniLM-L6-v2` model
3. **Non-linear Dimensionality Reduction** - UMAP
4. **Define best number of clusters** with **Silhoutte score**
5. **Clustering** - KMeans
6. **Labeling clusters** - KeyBERT

![alt text](images/image.png)

## How to run?

**1. Clone:**

```
git clone git@github.com:montarelab/clusterly.git
```

**2. Install requirements**

```
pip install -r requirements.txt
```

**3. Go to `src`:**

```
cd src
```

**4. Run app**

```
python app.py
```
