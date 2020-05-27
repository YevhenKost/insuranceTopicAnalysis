import pandas as pd
import numpy as np
from tqdm import tqdm
# from preprocessing_utils import Preprocessing
#
# df = pd.read_pickle("Dataset.pkl")
#
# df["is_general"] = df["from"].apply(lambda x: "general" in x)
# df["is_axa"] = df["from"].apply(lambda x: "axa" in x)
#
# axa_texts = df[df["is_axa"] == True]["translated"].values.tolist()
# gen_texts = df[df["is_general"] == True]["translated"].values.tolist()
#
# preprocessing_pipeline = Preprocessing()
# print("start preprocessing")
# preprocessed_texts_axa = preprocessing_pipeline.preprocess_batch(axa_texts)
# print("done axa")
# preprocessed_texts_gen = preprocessing_pipeline.preprocess_batch(gen_texts)
# print("done gen")
#
# with open("preprocessed_data.json", "w") as f:
#     json.dump({"axa":preprocessed_texts_axa, "gen":preprocessed_texts_gen}, f)

df = pd.read_pickle("Dataset_with_lemmas.pkl")

def get_length_quantile(df_, level = 0.4):

    lens = df_["lemmas"].apply(lambda x: len(x))
    return np.quantile(lens,level)

max_cut = get_length_quantile(df)
cutted_lemmas_data = []
keep_cols = list(df.columns)
keep_cols = keep_cols.remove("lemmas")

for _, row in tqdm(df.iterrows()):

    kept_data = row.to_dict()

    if len(row["lemmas"]) > max_cut:
        for cutted_ in np.array_split(row["lemmas"], len(row["lemmas"]) // max_cut):
            kept_data["lemmas"] = cutted_.tolist()
            cutted_lemmas_data.append(kept_data)


    else:
        cutted_lemmas_data.append(kept_data)

new_df = pd.DataFrame.from_records(cutted_lemmas_data)
new_df.to_pickle("Dataset_with_lemmas_04cut.pkl")





