from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from stemmer import UkrainianStemmer

class Preprocessing:

    @classmethod
    def _tokenize(cls, text):
        return word_tokenize(text)

    @classmethod
    def _preprocess_token(cls, token):

        token = token.lower()
        token = token.strip()
        token = UkrainianStemmer(token).stem_word()

        return token

    @classmethod
    def preprocess_text(cls, text):

        tokens = cls._tokenize(text)
        tokens = list(map(cls._preprocess_token, tokens))
        return tokens

    @classmethod
    def preprocess_batch(cls, texts):
        return list(map(cls.preprocess_text, texts))

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


