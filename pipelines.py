from preprocessing_utils import Preprocessing, FeatureExtraction
from lda_utils import LDA


class Docs2Topics:

    def __init__(self, feature_extraction_args, lda_args):
        self.feature_extractor = FeatureExtraction(**feature_extraction_args)
        # self.preprocessor = Preprocessing()
        self.lda_model = LDA(**lda_args)

    def get_topics(self, texts):

        # texts = self.preprocessor.preprocess_batch(texts)
        features = self.feature_extractor.fit_transform(texts)
        feature_index_dict = self.feature_extractor.feature_index_dict

        print("Vocabulary Length: ", len(feature_index_dict))


        self.lda_model.fit(features, feature_index_dict)
        topics = self.lda_model.extracted_topics

        return topics


if __name__ == '__main__':

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
Вільна й міцна, від Сяну по Кавказ""".split("\n")

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

    topics = Docs2Topics.get_topics(texts, feature_extraction_args, lda_args)

    # import pprint
    # pprint.pprint(topics)
    # {0: [(',', 0.01992596015018256),
    #      ('велик', 0.01615288636099073),
    #      ('свобод', 0.01593267663712953),
    #      ('бо', 0.015625739990102007)],
    #  1: [(',', 0.026545867700971584),
    #      ('–', 0.02237089104273583),
    #      ('нас', 0.018479716599540626),
    #      ('.', 0.01700120425530488)]}
