import json
import re
from nltk.corpus import stopwords
import pymorphy2


# Загрузка данных из JSON файла
def load_data(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        articles = json.load(f)
    return articles


# Очистка текста
def clean_text(text):
    text = re.sub(r'\s+', ' ', text)  # Удаление лишних пробелов
    text = re.sub(r'[^\w\s]', '', text)  # Удаление пунктуации
    return text.lower()


# Удаление стоп-слов
def remove_stopwords(text, stop_words):
    words = text.split()
    return ' '.join([word for word in words if word not in stop_words])


# Лемматизация текста
def lemmatize_text(text, morph):
    words = text.split()
    return ' '.join([morph.parse(word)[0].normal_form for word in words])


def preprocess_articles(articles, remove_stop_words=False, use_lemmatization=False):
    stop_words = set(stopwords.words('russian'))
    morph = pymorphy2.MorphAnalyzer()

    for article in articles:
        article['text'] = clean_text(article['text'])
        article['title'] = clean_text(article['title'])

        if remove_stop_words:
            article['text'] = remove_stopwords(article['text'], stop_words)
            article['title'] = remove_stopwords(article['title'], stop_words)

        if use_lemmatization:
            article['text'] = lemmatize_text(article['text'], morph)
            article['title'] = lemmatize_text(article['title'], morph)

    return articles


def save_data(articles, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(articles, f, ensure_ascii=False, indent=4)


def main():
    name = 'newsCryptoProcessed'
    articles = load_data('newsCrypto.json')
    processed_articles = preprocess_articles(articles, remove_stop_words=True, use_lemmatization=True)
    save_data(processed_articles, name+'.json')

    name = 'newsFootballProcessed'
    articles = load_data('newsFootball.json')
    processed_articles = preprocess_articles(articles, remove_stop_words=True, use_lemmatization=True)
    save_data(processed_articles, name+'.json')

    name = 'newsHockeyProcessed'
    articles = load_data('newsHockey.json')
    processed_articles = preprocess_articles(articles, remove_stop_words=True, use_lemmatization=True)
    save_data(processed_articles, name+'.json')


if __name__ == "__main__":
    main()