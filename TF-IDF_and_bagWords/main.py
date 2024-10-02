import json
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pandas as pd

# Загрузка данных из JSON файла
def load_data(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        articles = json.load(f)
    return articles


def build_bow_tfidf(articles):
    texts = [article['text'] for article in articles]

    # Мешок слов
    vectorizer_bow = CountVectorizer()
    X_bow = vectorizer_bow.fit_transform(texts)

    # TF-IDF
    vectorizer_tfidf = TfidfVectorizer()
    X_tfidf = vectorizer_tfidf.fit_transform(texts)

    return X_bow, X_tfidf, vectorizer_bow, vectorizer_tfidf


def save_data(sorted_bow, sorted_tfidf, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("Top 20 Bag of Words (BoW) Features:\n")
        for word, freq in sorted_bow:
            f.write(f"{word}: {freq}\n")

        f.write("\nTop 20 TF-IDF Features:\n")
        for word, freq in sorted_tfidf:
            f.write(f"{word}: {freq}\n")



def main(name):
    articles = load_data(name)

    X_bow, X_tfidf, vectorizer_bow, vectorizer_tfidf = build_bow_tfidf(articles)

    # Получение частот признаков
    bow_frequencies = X_bow.sum(axis=0).A1
    tfidf_frequencies = X_tfidf.sum(axis=0).A1

    # Создание словарей признаков и их частот
    bow_features = vectorizer_bow.get_feature_names_out()
    tfidf_features = vectorizer_tfidf.get_feature_names_out()

    bow_freq_dict = dict(zip(bow_features, bow_frequencies))
    tfidf_freq_dict = dict(zip(tfidf_features, tfidf_frequencies))

    # Сортировка признаков по частоте
    sorted_bow = sorted(bow_freq_dict.items(), key=lambda x: x[1], reverse=True)[:20]
    sorted_tfidf = sorted(tfidf_freq_dict.items(), key=lambda x: x[1], reverse=True)[:20]

    save_data(sorted_bow, sorted_tfidf, name+'.txt')


if __name__ == "__main__":
    filenames = ['newsCryptoProcessed.json', 'newsFootballProcessed.json', 'newsHockeyProcessed.json']
    for name in filenames:
        main(name)