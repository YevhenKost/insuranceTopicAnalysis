# Insurance Topic Analysis
Repository contains code regarding LDA Analysis of insurance related news 

# Examples
```python

from pipelines import Docs2Topics
from preprocessing_utils import Preprocessing

texts = """Зродились ми великої години
З пожеж війни, із полум’я вогнів,
Плекав нас біль по втраті України,
Кормив нас гнів і злість на ворогів.

І ось ми йдем у бою життєвому –
Тверді, міцні, незламні, мов граніт,
Бо плач не дав свободи ще нікому,
А хто борець, той здобуває світ.

Не хочемо ні слави, ні заплати.
Заплатою нам – розкіш боротьби!
Солодше нам у бою умирати,
Ніж жити в путах, мов німі раби.

Доволі нам руїни і незгоди,
Не сміє брат на брата йти у бій!
Під синьо-жовтим прапором свободи
З’єднаєм весь великий нарід свій.

Велику правду – для усіх єдину,
Наш гордий клич народові несе!
Вітчизні ти будь вірний до загину,
Нам Україна вище понад усе!

Веде нас в бій борців упавших слава.
Для нас закон найвищий – то приказ:
Соборна Українська держава –
Вільна й міцна, від Сяну по Кавказ"""

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

preprocessed_text = preprocessing_pipeline.preprocess_batch(texts)
topics = doc2topic.get_topics(preprocessed_text)
```

It provides output: 
```python
{0: [(',', 0.009193804823965581),
     ('ми', 0.009181293571095492),
     ('.', 0.009160791355235289),
     ('–', 0.00915213723975985)],
 1: [(',', 0.018401715777324638),
     ('ми', 0.014509728716192957),
     ('.', 0.011925993263786154),
     ('–', 0.011283228533043869)]}
```