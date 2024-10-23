import json
import matplotlib.pyplot as plt
from collections import Counter


def zipf_law(words, name):
    frequency = Counter(words)
    sorted_words = sorted(frequency, key=frequency.get, reverse=True)
    sorted_frequencies = [frequency[word] for word in sorted_words]

    # График закона Ципфа
    plt.figure(figsize=(12, 8))
    plt.bar(sorted_words[:50], sorted_frequencies[:50])  # Показываем топ-50 слов для удобства отображения
    plt.xticks(rotation=90)
    plt.xlabel('Слова')
    plt.ylabel('Частота встречаемости')
    plt.title('Закон Ципфа')
    plt.savefig(name + '.png')  # Сохранение графика
    plt.show()


def process_json(input_filename):
    with open(input_filename, 'r', encoding='utf-8') as file:
        data = json.load(file)

    all_words = []
    for item in data:
        if isinstance(item, dict):
            for key in ["title", "text"]:
                if key in item:
                    words = item[key].split()  # Простое разбиение текста на слова
                    all_words.extend(words)

    return all_words


def main():
    filenames = ['newsCryptoProcessed.json', 'newsFootballProcessed.json', 'newsHockeyProcessed.json']
    for filename in filenames:
        words = process_json(filename)
        print(f"Обрабатываем файл: {filename}, количество слов: {len(words)}")
        name = filename.split('.')[0]
        zipf_law(words, name)


if __name__ == "__main__":
    main()
