# Insurance Topic Analysis
Repository contains code regarding LDA Analysis of insurance related news 

# Examples
```python

from pipelines import Docs2Topics
from preprocessing_utils import Preprocessing

texts = """To be, or not to be, that is the question:
          Whether 'tis nobler in the mind to suffer
          The slings and arrows of outrageous fortune,
          Or to take Arms against a Sea of troubles,
          And by opposing end them: to die, to sleep;
          No more; and by a sleep, to say we end
          The heart-ache, and the thousand natural shocks
          That Flesh is heir to? 'Tis a consummation
          Devoutly to be wished. To die, to sleep,
          To sleep, perchance to Dream; aye, there's the rub,
          For in that sleep of death, what dreams may come,
          When we have shuffled off this mortal coil,
          Must give us pause."""

feature_extraction_args = {"max_df":1.0,
                           "min_df":0,
                           "stop_words":[],

                           # keep defaults
                           "analyzer":"word",
                           "tokenizer":None,
                           "lowcase":False}

lda_args = {"n_topics": 2,
            "max_iter": 10,
            "n_jobs": -1,

            # Note: thold is more prior than n_topic_words
            # But works slower
            "n_topic_words": 4,
            "word_thold": None}

preprocessing_pipeline = Preprocessing()
doc2topic = Docs2Topics(feature_extraction_args, lda_args)

preprocessed_text = preprocessing_pipeline.preprocess_batch([texts])
topics, words_per_topic = doc2topic.get_topics(preprocessed_text)
```

It provides output: 
```python

# topics
{0: [(',', 0.014686180021571887),
     ('to', 0.014184309385795015),
     ('sleep', 0.013349824349856078),
     ('the', 0.013255000918955656)],
 1: [(',', 0.023162549551458914),
     ('to', 0.018766991048079128),
     ('sleep', 0.0146796338376476),
     ('the', 0.013795378848642228)]}

# words_per_topic
[[',', 'to', 'sleep', 'the'], [',', 'to', 'sleep', 'the']]

```
