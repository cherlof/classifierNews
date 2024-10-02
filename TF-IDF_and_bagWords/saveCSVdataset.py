import json
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Загрузка данных из JSON файла
def load_data(filename, label):
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            articles = json.load(f)
        for article in articles:
            article['label'] = label
        return articles
    except Exception as e:
        print(f"Ошибка при загрузке данных: {e}")
        return []

# Построение мешка слов и TF-IDF
def build_bow_tfidf(articles):
    texts = [article['text'] for article in articles]

    # Мешок слов
    vectorizer_bow = CountVectorizer()
    X_bow = vectorizer_bow.fit_transform(texts)

    # TF-IDF
    vectorizer_tfidf = TfidfVectorizer()
    X_tfidf = vectorizer_tfidf.fit_transform(texts)

    return X_bow, X_tfidf, vectorizer_bow, vectorizer_tfidf

# Сохранение признаковых описаний в CSV файл
def save_features_to_csv(X_bow, X_tfidf, vectorizer_bow, vectorizer_tfidf, labels, filename):
    df_bow = pd.DataFrame(X_bow.toarray(), columns=vectorizer_bow.get_feature_names_out())
    df_tfidf = pd.DataFrame(X_tfidf.toarray(), columns=vectorizer_tfidf.get_feature_names_out())

    df_bow['label'] = labels
    df_tfidf['label'] = labels

    df_combined = pd.concat([df_bow, df_tfidf], axis=1)
    df_combined.to_csv(filename, index=False)

def main():
    filenames = [
        ('newsCryptoProcessed.json', 'crypto'),
        ('newsFootballProcessed.json', 'football'),
        ('newsHockeyProcessed.json', 'hockey')
    ]

    all_articles = []
    for name, label in filenames:
        articles = load_data(name, label)
        all_articles.extend(articles)

    if all_articles:
        X_bow, X_tfidf, vectorizer_bow, vectorizer_tfidf = build_bow_tfidf(all_articles)
        labels = [article['label'] for article in all_articles]
        save_features_to_csv(X_bow, X_tfidf, vectorizer_bow, vectorizer_tfidf, labels, 'combined_features.csv')

if __name__ == "__main__":
    main()