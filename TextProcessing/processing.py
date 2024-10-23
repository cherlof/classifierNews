import json
import re
from pymystem3 import Mystem
from nltk.corpus import stopwords

# Загрузка необходимых данных для nltk
import nltk
nltk.download('stopwords')

# Инициализация объектов
stop_words = set(stopwords.words('russian'))
mystem = Mystem()

# Функция для очистки и нормализации текста
def preprocess_text(text):
    # Удаление цифр и специальных символов
    text = re.sub(r'[^\w\s]', '', text)
    # Приведение к нижнему регистру
    text = text.lower()
    # Лемматизация текста
    lemmas = mystem.lemmatize(text)
    # Удаление стоп-слов и очистка от пустых строк
    cleaned_words = [lemma for lemma in lemmas if lemma.strip() and lemma not in stop_words]
    return ' '.join(cleaned_words)

# Функция для обработки JSON данных
def process_json(input_filename, output_filename):
    with open(input_filename, 'r', encoding='utf-8') as file:
        data = json.load(file)

    processed_data = []
    for item in data:
        processed_article = {}
        for key, value in item.items():
            if key in ["title", "text"]:
                processed_article[key] = preprocess_text(value)
            else:
                processed_article[key] = value
        processed_data.append(processed_article)

    # Принудительное завершение работы mystem
    mystem.close()

    # Сохранение обработанных данных обратно в JSON
    with open(output_filename, 'w', encoding='utf-8') as file:
        json.dump(processed_data, file, ensure_ascii=False, indent=4)
def main():

    process_json('newsCrypto.json', 'newsCryptoProcessed.json')
    process_json('newsFootball.json', 'newsFootballProcessed.json')
    process_json('newsHockey.json', 'newsHockeyProcessed.json')

if __name__ == "__main__":
    main()