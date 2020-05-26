from sklearn.decomposition import NMF, LatentDirichletAllocation
import numpy as np

class LDA:

    def __init__(self, n_topics=10, max_iter=10,n_jobs=-1, n_topic_words=10, word_thold=None):
        self.model = LatentDirichletAllocation(n_components=n_topics,max_iter=max_iter, n_jobs=n_jobs, random_state=2)
        self.extracted_topics = {}
        self.topic_num_words = n_topic_words
        self.thold = word_thold

    def fit(self, features, feature_index_dict=None):
        self.model.fit(features)
        self._update_topics(feature_index_dict)

    def _update_topics(self, feature_index_dict):

        for topic_idx, topic in enumerate(self.model.components_/self.model.components_.sum(axis=1)[:, np.newaxis]):

            if not self.thold:
                self.extracted_topics[topic_idx] = [(feature_index_dict[i], topic[i])
                                 for i in topic.argsort()[:-self.topic_num_words - 1:-1]]
            else:
                self.extracted_topics[topic_idx] = [(feature_index_dict[i], topic[i])
                                                    for i in topic if i >= self.thold]

if __name__ == '__main__':

    import numpy as np

    model = LatentDirichletAllocation()
    model.fit(np.random.randint(0,100, (10, 20)))
    c = model.components_/model.components_.sum(axis=1)[:, np.newaxis]
    print()