import itertools
import gensim
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA, TruncatedSVD
import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pickle


def cv(data, transform=False):
    count_vectorizer = CountVectorizer()
    if not transform:
        res = count_vectorizer.fit_transform(data)
        save_vectorizer("bow", count_vectorizer)
        return res
    else:
        return load_vectorizer("bow").transform(data)


def tfidf(data, transform=False):
    tfidf_vectorizer = TfidfVectorizer()
    if not transform:
        res = tfidf_vectorizer.fit_transform(data)
        save_vectorizer("tfidf", tfidf_vectorizer)
        return res
    else:
        return load_vectorizer("tfidf").transform(data)


def save_vectorizer(model_name, vectorizer):
    pickle.dump(vectorizer, open("../models/" + model_name + "_vectorizer.save", 'wb'))


def load_vectorizer(model_name):
    if model_name not in ['bow', 'tfidf', 'w2v']:
        raise ValueError("Unsupported model name.")

    return pickle.load(open("../models/" + model_name + "_vectorizer.save", 'rb'))


def save_model(output, model):
    pickle.dump(model, open("../models/" + output, 'wb'))


def load_model(model_name, test=False):
    if model_name not in ['bow', 'tfidf', 'w2v']:
        raise ValueError("Unsupported model name.")

    path = "../models/"
    if test:
        path += "test_"

    return pickle.load(open(path + model_name + "_model.save", 'rb'))


def _train_model(corpus, labels, model_name, dataframe=None):
    if model_name == "bow":
        corpus_counts = cv(corpus)
        bow_classifier = LogisticRegression(C=30.0, class_weight='balanced', solver='newton-cg',
                                            multi_class='multinomial', random_state=40)
        bow_classifier.fit(corpus_counts, labels)

        return bow_classifier
    elif model_name == "tfidf":
        corpus_tfidf = tfidf(corpus)
        tfidf_classifier = LogisticRegression(C=30.0, class_weight='balanced', solver='newton-cg',
                                              multi_class='multinomial', random_state=40)
        tfidf_classifier.fit(corpus_tfidf, labels)

        return tfidf_classifier
    else:
        word2vec_path = "../GoogleNews-vectors-negative300.bin"
        word2vec = gensim.models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True)

        embeddings = get_word2vec_embeddings(word2vec, dataframe)
        X_train_word2vec, X_test_word2vec, y_train_word2vec, y_test_word2vec = train_test_split(embeddings, labels,
                                                                                                test_size=0.0,
                                                                                                random_state=40)

        w2v_classifier = LogisticRegression(C=30.0, class_weight='balanced', solver='newton-cg',
                                            multi_class='multinomial', random_state=40)
        w2v_classifier.fit(X_train_word2vec, y_train_word2vec)

        return w2v_classifier


def train_model(dataframe, model_name=None, save=False):
    list_corpus = dataframe["Comment text"].values.astype("U").tolist()
    list_labels = dataframe["Class"].tolist()

    corpus, X_test, labels, y_test = train_test_split(list_corpus, list_labels, test_size=0.0,
                                                        random_state=40)

    if model_name not in ['bow', 'tfidf', 'w2v', None]:
        raise ValueError("model_name must be either 'bow', 'tfidf' or 'w2v")

    if model_name is None:
        bow_classifier = _train_model(corpus, labels, "bow")
        tfidf_classifier = _train_model(corpus, labels, "tfidf")
        w2v_classifier = _train_model(corpus, labels, "w2v", dataframe)

        if save:
            save_model("bow_model.save", bow_classifier)
            save_model("tfidf_model.save", tfidf_classifier)
            save_model("w2v_model.save", w2v_classifier)

        return bow_classifier, tfidf_classifier, w2v_classifier

    elif model_name == "bow":
        bow_classifier = _train_model(corpus, labels, "bow")
        if save:
            save_model("bow_model.save", bow_classifier)
        return bow_classifier

    elif model_name == "tfidf":
        tfidf_classifier = _train_model(corpus, labels, "tfidf")
        if save:
            save_model("tfidf_model.save", tfidf_classifier)
        return tfidf_classifier

    elif model_name == "w2v":
        w2v_classifier = _train_model(corpus, labels, "w2v", dataframe)
        if save:
            save_model("w2v_model.save", w2v_classifier)
        return w2v_classifier


def plot_LSA(test_data, test_labels, savepath="PCA_demo.csv", plot=True):
    lsa = TruncatedSVD(n_components=2)
    lsa.fit(test_data)
    lsa_scores = lsa.transform(test_data)
    color_mapper = {label: idx for idx, label in enumerate(set(test_labels))}
    color_column = [color_mapper[label] for label in test_labels]
    colors = ['orange', 'blue', 'blue']
    if plot:
        plt.scatter(lsa_scores[:, 0], lsa_scores[:, 1], s=8, alpha=.8, c=test_labels,
                    cmap=matplotlib.colors.ListedColormap(colors))
        red_patch = mpatches.Patch(color='orange', label='0')
        green_patch = mpatches.Patch(color='blue', label='1')
        plt.legend(handles=[red_patch, green_patch], prop={'size': 30})


def get_metrics(y_test, y_predicted):
    # true positives / (true positives+false positives)
    precision = precision_score(y_test, y_predicted, pos_label=None,
                                average='weighted')
    # true positives / (true positives + false negatives)
    recall = recall_score(y_test, y_predicted, pos_label=None,
                          average='weighted')

    # harmonic mean of precision and recall
    f1 = f1_score(y_test, y_predicted, pos_label=None, average='weighted')

    # true positives + true negatives/ total
    accuracy = accuracy_score(y_test, y_predicted)
    return accuracy, precision, recall, f1


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.winter):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=30)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=20)
    plt.yticks(tick_marks, classes, fontsize=20)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center",
                 color="white" if cm[i, j] < thresh else "black", fontsize=40)

    plt.tight_layout()
    plt.ylabel('True label', fontsize=30)
    plt.xlabel('Predicted label', fontsize=30)

    return plt


def get_most_important_features(vectorizer, model, n=5):
    index_to_word = {v: k for k, v in vectorizer.vocabulary_.items()}

    # loop for each class
    classes = {}
    print(model.coef_)
    for class_index in range(model.coef_.shape[0]):
        word_importances = [(el, index_to_word[i]) for i, el in enumerate(model.coef_[class_index])]
        sorted_coeff = sorted(word_importances, key=lambda x: x[0], reverse=True)
        tops = sorted(sorted_coeff[:n], key=lambda x: x[0])
        bottom = sorted_coeff[-n:]
        classes[class_index] = {
            'tops': tops,
            'bottom': bottom
        }
    return classes


def plot_important_words(top_scores, top_words, bottom_scores, bottom_words, name):
    y_pos = np.arange(len(top_words))
    top_pairs = [(a, b) for a, b in zip(top_words, top_scores)]
    top_pairs = sorted(top_pairs, key=lambda x: x[1])

    bottom_pairs = [(a, b) for a, b in zip(bottom_words, bottom_scores)]
    bottom_pairs = sorted(bottom_pairs, key=lambda x: x[1], reverse=True)

    top_words = [a[0] for a in top_pairs]
    top_scores = [a[1] for a in top_pairs]

    bottom_words = [a[0] for a in bottom_pairs]
    bottom_scores = [a[1] for a in bottom_pairs]

    fig = plt.figure(figsize=(10, 10))

    plt.subplot(121)
    plt.barh(y_pos, bottom_scores, align='center', alpha=0.5)
    plt.title('We good', fontsize=20)
    plt.yticks(y_pos, bottom_words, fontsize=14)
    plt.suptitle('Key words', fontsize=16)
    plt.xlabel('Importance', fontsize=20)

    plt.subplot(122)
    plt.barh(y_pos, top_scores, align='center', alpha=0.5)
    plt.title('Spam', fontsize=20)
    plt.yticks(y_pos, top_words, fontsize=14)
    plt.suptitle(name, fontsize=16)
    plt.xlabel('Importance', fontsize=20)

    plt.subplots_adjust(wspace=0.8)
    plt.show()


def get_average_word2vec(tokens_list, vector, generate_missing=False, k=300):
    if len(tokens_list) < 1:
        return np.zeros(k)
    if generate_missing:
        vectorized = [vector[word] if word in vector else np.random.rand(k) for word in tokens_list]
    else:
        vectorized = [vector[word] if word in vector else np.zeros(k) for word in tokens_list]
    length = len(vectorized)
    summed = np.sum(vectorized, axis=0)
    averaged = np.divide(summed, length)
    return averaged


def get_word2vec_embeddings(vectors, clean_questions, generate_missing=False):
    embeddings = clean_questions['Tokens'].apply(lambda x: get_average_word2vec(x, vectors,
                                                                                generate_missing=generate_missing))
    return list(embeddings)


def main():
    clean_questions = pd.read_csv("clear_data.csv")

    train_model(clean_questions, save=False)


if __name__ == "__main__":
    main()
