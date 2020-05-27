from sklearn.feature_extraction.text import TfidfVectorizer
import stanza

class Preprocessing:

    def __init__(self):

        self.stanford_pipeline = stanza.Pipeline("uk", use_gpu=True)

    def preprocess_text(self, text):

        doc = self.stanford_pipeline(text)

        lemmatized_output = []
        for s in doc.sentences:
            for t in s.words:
                lemmatized_output.append(t.lemma)
        del doc
        return lemmatized_output

    def preprocess_batch(self, texts):
        return list(map(self.preprocess_text, texts))

class FeatureExtraction:

    def __init__(self, max_df, min_df,stop_words, analyzer="word", tokenizer=None,lowcase=False):

        if not tokenizer:
            tokenizer = self._default_tokenizer
        self.tf_idf = TfidfVectorizer(max_df=max_df, min_df=min_df, lowercase=lowcase,
                                      stop_words=stop_words,analyzer=analyzer,tokenizer=tokenizer)
        self.feature_index_dict = {}

    def _default_tokenizer(self, x):
        return x

    def fit_transform(self, tokenized_texts):
        """tokenized_texts - list of preprocessed stings"""
        features = self.tf_idf.fit_transform(tokenized_texts)
        self._update_feature_index_dict(self.tf_idf.vocabulary_)
        return features

    def _update_feature_index_dict(self, vocabulary_tfifd):
        self.feature_index_dict = {v:k for k,v in vocabulary_tfifd.items()}


