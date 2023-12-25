import pandas as pd
import numpy as np
import re
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

def pre_process(data_df, train_or_test="Train", outliers_words_location=None, outliers_words_keyword=None):
    if train_or_test == "Train":
        y = data_df['target']
        data_df.drop(['target', 'id'], axis=1, inplace=True)
    else:
        y = None
    # extract one hot encoding for location and keyword
    data_df, words_location, words_keyword = one_hot_encoding(data_df, outliers_words_location, outliers_words_keyword)
    # extract features from text
    embeddings = text_embedding(data_df, train_or_test=train_or_test)
    # add the embeddings to the data frame
    data_df = pd.concat([data_df, pd.DataFrame(embeddings, columns=[f"text_embedding{i}" for i in range(embeddings.shape[1])])], axis=1)
    columns_to_drop = ['location', 'keyword', 'text']
    data_df.drop(columns_to_drop, axis=1, inplace=True)
    # normalize except the columns that are one hot encoded
    one_hot_len = len(words_location) + len(words_keyword)
    data_df = data_df.iloc[:, :one_hot_len].join(data_df.iloc[:, one_hot_len:].apply(lambda x: (x - x.mean()) / x.std(), axis=1))
    return data_df, y, words_location, words_keyword


def one_hot_encoding(df, valid_words_location=None, valid_words_keyword=None):
    dummy_location, words_location = get_dummies(df["location"], valid_words_location)
    dummy_keyword, words_keyword = get_dummies(df["keyword"], valid_words_keyword)
    df = pd.concat([df, dummy_location, dummy_keyword], axis=1)
    return df, words_location, words_keyword


def get_dummies(column, valid_words=None):
    if valid_words is None:
        valid_words = get_valid_words(column)
    # create new data frame with one hot encoding
    one_hot_df = pd.DataFrame(index=column.index, columns=valid_words)
    one_hot_df = one_hot_df.fillna(False)
    # add the words that are in the column
    for word in valid_words:
        one_hot_df[word] = column.str.contains(word, regex=False)
    # replace nan with False
    one_hot_df = one_hot_df.fillna(False)
    return one_hot_df, valid_words


def text_embedding(df, train_or_test="train", load=True):
    if load:
        try:
            embeddings = np.load(f"Results/{train_or_test}_text_embedding.npy", allow_pickle=True)
            embeddings = np.stack(embeddings)
            return embeddings
        except:
            print("No embeddings file found, generating new embeddings")
    embeddings = df["text"].apply(lambda x: model.encode(x))
    embeddings = np.stack(embeddings.values)
    # save embeddings
    np.save(f"Results/{train_or_test}_text_embedding.npy", embeddings)
    return embeddings


def get_valid_words(column):
    column = column.dropna()
    # split the text to words
    column = column.apply(lambda x: re.sub(r'[^\w\s]', ',', x))
    column = column.str.split(",")
    column = column.explode()
    # remove spaces and hashtags
    column = column.str.replace(" ", "")
    column = column.str.replace("#", "")
    # detect none words like ? valid words are words that contain only english letters and some special characters
    valid_letters = "-" + " " + "'" + "." + 'abcdefghijklmnopqrstuvwxyz' + 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    valid_words = column[column.str.contains(f"[^{valid_letters}]", regex=True) == False]
    # remove empty valid words
    valid_words = valid_words[valid_words != ""]
    # get the words with the max occurrences in the column
    valid_words_count = valid_words.value_counts()
    # drop the words with less than 0.001% occurrences
    valid_words = list(valid_words_count[valid_words_count >= 0.001 * len(valid_words)].index)
    return valid_words


if __name__ == '__main__':
    pass

