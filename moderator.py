import sys
import gensim
import pandas as pd
from models_trainer import load_model
from extractor import file_to_df, comments_to_df
from tokenizer import count_and_clean_df
from models_trainer import cv, tfidf, get_word2vec_embeddings


def prepare_data(model_name, data):
    if model_name == "bow":
        return cv(data, True)
    elif model_name == "tfidf":
        return tfidf(data, True)


def classify(data, model_name=None, output=None):
    if model_name is None:
        classify(data, "bow")
        classify(data, "tfidf")
        classify(data, "w2v")
        data.to_csv(output)

    else:
        list_corpus = data["Comment text"].values.astype("U").tolist()
        predictions = None

        if model_name == "bow":
            bow_model = load_model("bow")
            bow_data = prepare_data("bow", list_corpus)
            predictions = bow_model.predict(bow_data)

        elif model_name == "tfidf":
            tfidf_model = load_model("tfidf")
            tfidf_data = prepare_data("tfidf", list_corpus)
            predictions = tfidf_model.predict(tfidf_data)

        else:
            word2vec_path = "GoogleNews-vectors-negative300.bin"
            word2vec = gensim.models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True, limit=500000)

            w2v_model = load_model("w2v")
            w2v_data = get_word2vec_embeddings(word2vec, data)
            predictions = w2v_model.predict(w2v_data)

        for i in range(len(predictions)):
            data.loc[i, model_name] = predictions[i]
            if predictions[i] == 1:
                print("'", data.loc[i, "Comment text"], "' DETECTED AS SPAM BY ", model_name, ".", sep="")

        if output is not None:
            data.to_csv(output)


def main():
    print(len(sys.argv))
    if len(sys.argv) < 4:
        print("Missing arguments.\nUsage: python moderator.py mode[-file <path to file with comments>, -id <youtube video "
              "url>, -string <single string in quotation marks>] <output file name> optional:model name[bow, tfidf, w2v]")
        return -1

    data = None
    model_name = None

    if sys.argv[1] == "-file":
        filepath = sys.argv[2]
        data = file_to_df(filepath)
    elif sys.argv[1] == "-id":
        video_id = sys.argv[2]
        data = comments_to_df(video_id)
    elif sys.argv[1] == "-string":
        data = pd.DataFrame({"Unnamed: 0": 0, "Comment text": sys.argv[2]})
        count_and_clean_df(data)
    else:
        print("Wrong argument:", sys.argv[1])
        exit(-1)

    output = sys.argv[3]
    if len(sys.argv) > 4 and sys.argv[4] in ["bow", "tfidf", "w2v"]:
        model_name = sys.argv[4]

    classify(data, model_name, output)


if __name__ == "__main__":
    main()
