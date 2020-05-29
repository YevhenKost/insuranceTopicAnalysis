from sklearn.decomposition import LatentDirichletAllocation
import numpy as np
from itertools import combinations
from fasttext import load_model
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

class LDA:

    def __init__(self, n_topics=10, max_iter=10,n_jobs=-1, n_topic_words=10, word_thold=None):
        self.model = LatentDirichletAllocation(n_components=n_topics,max_iter=max_iter, n_jobs=n_jobs, random_state=2)
        self.extracted_topics = {}
        self.topic_num_words = n_topic_words
        self.thold = word_thold
        self.coherence_score = 0
        self.extracted_topics_words = []

    def fit(self, features, feature_index_dict):
        self.model.fit(features)
        self._update_topics(feature_index_dict)

    def _update_topics(self, feature_index_dict):

        for topic_idx, topic in enumerate(self.model.components_/self.model.components_.sum(axis=1)[:, np.newaxis]):

            if not self.thold:
                self.extracted_topics[topic_idx] = [(str(feature_index_dict[i]), topic[i])
                                 for i in topic.argsort()[:-self.topic_num_words - 1:-1]]
            else:
                self.extracted_topics[topic_idx] = [(feature_index_dict[i], topic[i])
                                                    for i in topic if i >= self.thold]
        self.extracted_topics_words = []
        for top_probs in self.extracted_topics.values():
            self.extracted_topics_words.append([x[0] for x in top_probs])

    def transform(self, texts):
        return self.model.transform(texts)

    def to_csv(self, path):
        df = pd.DataFrame()
        for i, t_words in enumerate(self.extracted_topics_words):

            df[f"topic_{i}"] = t_words
        df.to_csv(path)

class TopicNumEvaluation:

    def __init__(self, fasttext_path=r"D:\fasstText_models\ukrainian\cc.uk.300.bin"):

        self.w2v_model = load_model(fasttext_path)


    def calculate_topic_coherence(self, topic_words):

        # check each pair of terms
        pair_scores = []

        embedded_tokens = [self.w2v_model.get_word_vector(x) for x in topic_words]

        for pair in combinations(list(range(len(topic_words))), 2):

            w1 = embedded_tokens[pair[0]].reshape(1,-1)
            w2 = embedded_tokens[pair[1]].reshape(1, -1)
            pair_scores.append(cosine_similarity(w1, w2).item())

        # get the mean for all pairs in this topic
        topic_score = sum(pair_scores) / len(pair_scores)

        return topic_score

    def calculate_model_coherence(self, topics):

        scores = []
        for topic in topics:
            scores.append(self.calculate_topic_coherence(topic))

        return np.mean(scores)

if __name__ == '__main__':

    import os
    import pandas as pd
    from tqdm import tqdm

    top_gen, top_axa = {}, {}
    ev = TopicNumEvaluation()

    for t in tqdm(os.listdir("grid_topics_minToken5")):
        df = pd.read_csv(os.path.join("grid_topics_minToken5", t))
        topics = df["topics"].apply(lambda x: [y[0] for y in x]).values.tolist()

        sc = ev.calculate_model_coherence(topics)
        if "gen_" in t:
            top_gen[t] = sc
        if "axa_" in t:
            top_axa[t] = sc
    print(sorted(top_gen.items(), key=lambda x: x[1]))
    print(sorted(top_axa.items(), key=lambda x: x[1]))