# contextual translation similarity

This document demonstrates how to implement *Contextual Translation Similarity* (CTS) using data from the *CRITT Translation Process Research Database* (TPR-DB).

# materials

**1. pre-trained word vectors**

|filename|size|source|dimension|model type|window|
|--|--|--|--|--|--|
|eswiki_20180420_300d.pkl|4.47 GB|Wikipedia (Spanish)|300|skip-gram|5|

See also https://wikipedia2vec.github.io/wikipedia2vec/

**2. translation process data**

|study|SL|TL|participants|source texts|source tokens|target segments|
|--|--|--|--|--|--|--|
|BML12|English|Spanish|31|6|847|25937|

See also https://sites.google.com/site/centretranslationinnovation/tpr-db/public-studies

# how to

