# contextual translation similarity

This document demonstrates how to implement *Contextual Translation Similarity* (CTS) using data from the *CRITT Translation Process Research Database* (TPR-DB).

# materials

**1. pre-trained word vectors**

|filename|size of vocabulary|source|dimension|model type|window|
|--|--|--|--|--|--|
|eswiki_20180420_300d.pkl|1828809|Wikipedia (Spanish)|300|skip-gram|5|

See also https://wikipedia2vec.github.io/wikipedia2vec/

**2. translation process data**

|study|SL|TL|participants|source texts|source tokens|target segments|
|--|--|--|--|--|--|--|
|BML12|English|Spanish|31|6|847|25936|

See also https://sites.google.com/site/centretranslationinnovation/tpr-db/public-studies

# how-to

**1. load pre-trained word vectors and translation process data**

use the wikipedia2vec module to load pre-trained vectors:

```python
import pandas as pd
from wikipedia2vec import Wikipedia2Vec

wiki2vec = Wikipedia2Vec.load('eswiki_20180420_300d.pkl')
BML12 = pd.read_csv('BML12_ST.csv')
```

**2. obtain word vectors for target segments**

use the `merge_ST_TT()` function (see `contextual translation similarity.py`, the same below):

```python
ST_TT = merge_ST_TT(BML12)
```

*Output: dict*

```python
{
  (Text1, Id1, SToken1): [vec1, vec2, ...],
  ...
  }
```

**3. compute similarity values**

use the `contextual_translation_similarity()` function:

```python
CTS = contextual_translation_similarity(ST_TT)
```

*Output: pd.DataFrame*

|Text|Id|SToken|CTS_EUC|CTS_COS|CTS_MV|
|--|--|--|--|--|--|
|5|5|new|-4.2312057|0.88711752|9.16550539|
|5|6|academic|-5.5407048|0.81836244|9.82061641|
|...|...|...|...|...|...|

**4. add other features (optional)**

use `add_entropy()` and `add_mean_dur()` functions to add *word translation entropy* (HTra) values and *mean duration* (mDur) to the table obtained in step 3:

```python
CTS = add_entropy(BML12, CTS, 'HTra')
CTS = add_entropy(BML12, CTS, 'HCross')
CTS = add_mean_dur(BML12, CTS)
```

*Output: pd.DataFrame*

|Text|Id|SToken|CTS_EUC|CTS_COS|CTS_MV|HTra|HCross|mDur|
|--|--|--|--|--|--|--|--|--|
|5|5|new|-4.2312057|0.88711752|9.16550539|0.6907|0.6907|302.310345|
|5|6|academic|-5.5407048|0.81836244|9.82061641|1.0351|0.6907|482.275862|
|...|...|...|...|...|...|...|...|...|

mDur: the mean of durations (of different participants), computed for each source token

**5. export and append**

export the table obtained in step 3 or 4 to local environments or append it to the original translation process data:

```python
# export
CTS.to_csv('CTS.csv')

# append
BML12_CTS = append_CTS(BML12, CTS)
```

