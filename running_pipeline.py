from pipelines import Docs2Topics
from tqdm import tqdm
import json
import pprint
import os
import pandas as pd
from lda_utils import TopicNumEvaluation
import scipy.stats as ss
import numpy as np
import matplotlib.pyplot as plt


def grid_search(save_dir = "grid_topics_minToken5_"):
    num_topics_range = [20, 30, 40, 50, 60, 70]
    max_df_range = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]


    os.makedirs(save_dir, exist_ok=True)


    df = pd.read_pickle("Dataset_with_lemmas_04cut.pkl")
    df["lemmas"] = df["lemmas"].apply(lambda x: [t for t in x if len(t) >= 5])
    df["is_general"] = df["from"].apply(lambda x: "general" in x)
    df["is_axa"] = df["from"].apply(lambda x: "axa" in x)

    axa_texts = df[df["is_axa"] == True]["lemmas"].values.tolist()
    gen_texts = df[df["is_general"] == True]["lemmas"].values.tolist()


    evaluator = TopicNumEvaluation()
    evaluator_results = {}

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
            extracted_topics_axa, extracted_topics_axa_words = doc2topic.fit_get_topics(axa_texts)
            extracted_topics_gen, extracted_topics_gen_words = doc2topic.fit_get_topics(gen_texts)


            # validation
            axa_coh = evaluator.calculate_model_coherence(extracted_topics_axa_words)
            gen_coh = evaluator.calculate_model_coherence(extracted_topics_gen_words)

            print(n_topics, mdf, axa_coh, gen_coh)

            df = pd.DataFrame()
            df["topics"] = [v for v in extracted_topics_axa.values()]
            df.to_csv(os.path.join(save_dir, f"axa_ntop{n_topics}_mdf{mdf}.csv"))

            df = pd.DataFrame()
            df["topics"] = [v for v in extracted_topics_gen.values()]
            df.to_csv(os.path.join(save_dir, f"gen_ntop{n_topics}_mdf{mdf}.csv"))

            evaluator_results[f"axa_ntop{n_topics}_mdf{mdf}"] = axa_coh
            evaluator_results[f"gen_ntop{n_topics}_mdf{mdf}"] = gen_coh

    with open(os.path.join(save_dir, "eval_coh_scores.json"), "w") as f:
        json.dump(evaluator_results, f)

    gen_scores = {x:v for x,v in evaluator_results.items() if "gen_" in x}
    axa_scores = {x:v for x,v in evaluator_results.items() if "axa_" in x}

    print(sorted(gen_scores.items(), key=lambda x: x[1])[-1])
    print(sorted(axa_scores.items(), key=lambda x: x[1])[-1])


def calcualte_corr(df_path="Dataset_with_lemmas_04cut_rating.pkl",
                   max_df=0.6,
                   n_topics=60,
                   news_type="general",
                   split_by=["Період", "Рік"],
                   rating_cols =
                   ['Агрострахование', 'Активы', 'Гарантийный фонд', 'ДСАГО',
                    'Инвестиции в ОВГЗ', 'КАСКО', 'Медицинское страхование (ДМС)',
                    'Нераспределенная прибыль', 'ОСАГО', 'Премии от физических лиц',
                    'Премии от юридических лиц', 'Прямое страхование', 'Собственный капитал',
                    'Страхование грузов и багажа', 'Страхование здоровья на случай болезни',
                    'Страхование имущества', 'Страхование туристов (медрасходов)', 'Страховые выплаты',
                    'Страховые премии', 'Страховые резервы', 'Уровень выплат', 'Чистые страховые премии'],
                   save_dir_summary=""):

    # reading data and loading lemmatized texts
    df = pd.read_pickle(df_path)
    df["lemmas"] = df["lemmas"].apply(lambda x: [t for t in x if len(t) >= 5])
    df[f"is_{news_type}"] = df["from"].apply(lambda x: news_type in x)
    lemmatized_texts = df[df[f"is_{news_type}"] == True]["lemmas"].values.tolist()

    # init parameters and models
    lda_args = {"n_topics": n_topics,
                "max_iter": 50,
                "n_jobs": -1,

                # Note: thold is more prior than n_topic_words
                # But works slower
                "n_topic_words": 11,
                "word_thold": None}
    feature_extraction_args = {"max_df": max_df,
                               "min_df": 0,
                               "stop_words": [],

                               # keep defaults
                               "analyzer": "word",
                               "tokenizer": None,
                               "lowcase": False}

    print("Loaded data")
    # fitting LDA
    d2t = Docs2Topics(feature_extraction_args, lda_args)
    prob_topics, word_topics = d2t.fit_get_topics(lemmatized_texts)

    print("Fitted LDA")
    # calculating probs and ranks for each topic in texts
    # merging in dataframe
    topics_probs = d2t.predict_topics(lemmatized_texts)
    topic_cols = [f"topic_{i}" for i in range(len(word_topics))]
    prob_topic_df = pd.DataFrame(topics_probs, columns=topic_cols)
    rating_df = df[rating_cols + split_by].join(prob_topic_df)

    print("Predicted topics")

    # for each split calculating sum of probs for each topic and calculating rank
    summary_df = []
    for p, period_df in rating_df.groupby(split_by):

        summ_topics_probs = period_df[topic_cols].sum(axis=0).values
        ranks_topics = ss.rankdata(-summ_topics_probs, "min")
        ranks_topics = {topic_cols[k]:v for k,v in enumerate(ranks_topics)}
        for col in rating_cols:
            ranks_topics[col] = period_df[col].values.tolist()[0]
        ranks_topics[" ".join(split_by)] = p

        summary_df.append(ranks_topics)

    # calculating corrs
    print("Starting Corr")
    summary_df = pd.DataFrame.from_records(summary_df)

    summary_df = summary_df[topic_cols + rating_cols]
    summary_df.to_csv(os.path.join(save_dir_summary, f"Ntop{n_topics}_TypE{news_type}_maxdf{max_df}__summary.csv"))
    corr_summary_df = summary_df.corr()[topic_cols].iloc[len(topic_cols):]
    corr_summary_df.to_csv(os.path.join(save_dir_summary, f"Ntop{n_topics}_TypE{news_type}_maxdf{max_df}__summaryCORR.csv"))

    # plotting corr
    f = plt.figure(figsize=(15, 15))
    plt.matshow(corr_summary_df, fignum=f.number)
    plt.yticks(range(len(rating_cols)), rating_cols, fontsize=14)
    plt.xticks(range(len(topic_cols)), topic_cols, fontsize=14, rotation=90)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    plt.title('Correlation Matrix', fontsize=16)

    plt.show()








if __name__ == '__main__':
    calcualte_corr()