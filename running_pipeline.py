from pipelines import Docs2Topics
from tqdm import tqdm
import json
import pprint
import os
import pandas as pd

num_topics_range = [5, 10, 20, 30, 40, 50]
max_df_range = [0.4, 0.5, 0.6, 0.7, 0.8]

save_dir = "grid_topics"
os.makedirs(save_dir, exist_ok=True)


df = pd.read_pickle("Dataset_with_lemmas_04cut.pkl")
df["is_general"] = df["from"].apply(lambda x: "general" in x)
df["is_axa"] = df["from"].apply(lambda x: "axa" in x)

axa_texts = df[df["is_axa"] == True]["lemmas"].values.tolist()
gen_texts = df[df["is_general"] == True]["lemmas"].values.tolist()


for n_topics in tqdm(num_topics_range):
    for mdf in max_df_range:

        print(n_topics, mdf)

        lda_args = {"n_topics": n_topics,
                    "max_iter": 50,
                    "n_jobs": -1,

                    # Note: thold is more prior than n_topic_words
                    # But works slower
                    "n_topic_words": 11,
                    "word_thold": None}
        feature_extraction_args = {"max_df": mdf,
                                   "min_df": 0,
                                   "stop_words": [],

                                   # keep defaults
                                   "analyzer": "word",
                                   "tokenizer": None,
                                   "lowcase": False}

        doc2topic = Docs2Topics(feature_extraction_args, lda_args)
        extracted_topics_axa = doc2topic.get_topics(axa_texts)
        extracted_topics_gen = doc2topic.get_topics(gen_texts)


        with open(os.path.join(save_dir, f"axa_ntop{n_topics}_mdf{mdf}.json"), 'w') as f:
            json.dump(extracted_topics_axa, f)
        with open(os.path.join(save_dir, f"gen_ntop{n_topics}_mdf{mdf}.json"), 'w') as f:
            json.dump(extracted_topics_gen, f)