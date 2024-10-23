import json
import nltk
from nltk.util import ngrams
from collections import Counter

# Загрузка необходимых данных для nltk
nltk.download('punkt')
nltk.download('stopwords')


# Загрузка данных
def load_json(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        return json.load(file)


# Преобразование документов в список строк
def prepare_documents(data):
    documents = []
    for item in data:
        if isinstance(item, dict):
            doc = ' '.join([item.get("title", ""), item.get("text", "")])
            documents.append(doc)
    return documents


# Функция для извлечения биграмм и триграмм
def extract_ngrams(documents, n):
    ngrams_list = []
    for doc in documents:
        words = nltk.word_tokenize(doc, language='russian')
        ngrams_list.extend(list(ngrams(words, n)))
    return ngrams_list


# Функция для сохранения n-грамм в файл
def save_ngrams_to_file(filename, ngrams):
    with open(filename, 'w', encoding='utf-8') as file:
        for ngram in ngrams:
            file.write(' '.join(ngram) + '\n')


# Функция для сохранения результатов классификации в файл
def save_classification_results(filename, topic, bigram_freq, trigram_freq):
    with open(filename, 'w', encoding='utf-8') as file:
        file.write(f"Тема: {topic}\n")
        file.write("Топ-10 биграмм:\n")
        for bigram, freq in bigram_freq.most_common(10):
            file.write(f"{bigram}: {freq}\n")
        file.write("\nТоп-10 триграмм:\n")
        for trigram, freq in trigram_freq.most_common(10):
            file.write(f"{trigram}: {freq}\n")


def main():
    # Загрузка данных из всех файлов
    filenames = {
        'newsCryptoProcessed.json': 'Криптовалюты',
        'newsFootballProcessed.json': 'Футбол',
        'newsHockeyProcessed.json': 'Хоккей'
    }

    for filename, topic in filenames.items():
        data = load_json(filename)
        documents = prepare_documents(data)
        print(f"Для документа {filename} (Тема: {topic})")

        # Извлечение биграмм
        bigrams = extract_ngrams(documents, 2)
        # Подсчет биграмм
        bigram_freq = Counter(bigrams)
        # Сохранение биграмм в файл
        bigram_output_filename = filename.replace('.json', '_bigrams.txt')
        save_ngrams_to_file(bigram_output_filename, bigrams)
        print("Топ-10 биграмм:")
        for bigram, freq in bigram_freq.most_common(10):
            print(bigram, freq)

        # Извлечение триграмм
        trigrams = extract_ngrams(documents, 3)
        # Подсчет триграмм
        trigram_freq = Counter(trigrams)
        # Сохранение триграмм в файл
        trigram_output_filename = filename.replace('.json', '_trigrams.txt')
        save_ngrams_to_file(trigram_output_filename, trigrams)
        print("Топ-10 триграмм:")
        for trigram, freq in trigram_freq.most_common(10):
            print(trigram, freq)

        # Сохранение результатов классификации в файл
        classification_results_filename = filename.replace('.json', '_classification_results.txt')
        save_classification_results(classification_results_filename, topic, bigram_freq, trigram_freq)
        print(f"Результаты классификации сохранены в файл: {classification_results_filename}")


if __name__ == "__main__":
    main()
