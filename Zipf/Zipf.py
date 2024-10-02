import json
import matplotlib.pyplot as plt
from collections import Counter
import re


def load_data(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        articles = json.load(f)
    return articles


def get_word_frequencies(articles):
    text = " ".join(article['text'] for article in articles)
    words = re.findall(r'\w+', text.lower())
    return Counter(words)


def plot_zipf_law(frequencies, name):
    ranks = range(1, len(frequencies) + 1)
    frequencies = [freq for word, freq in frequencies]

    plt.figure(figsize=(10, 6))
    plt.plot(ranks, frequencies, marker='o')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Ранг')
    plt.ylabel('Частота')
    plt.title('Закон Ципфа')
    plt.grid(True)
    plt.savefig(name + '.png')

def save_top_words(frequencies, filename, top_n=20):
    with open(filename, 'w', encoding='utf-8') as f:
        for word, freq in frequencies[:top_n]:
            f.write(f"{word}: {freq}\n")


def main():
    name = 'newsCrypto'
    articles = load_data('newsCrypto.json')
    word_frequencies = get_word_frequencies(articles)
    sorted_frequencies = word_frequencies.most_common()
    plot_zipf_law(sorted_frequencies, name)
    save_top_words(sorted_frequencies, name+'.txt')

    name = 'newsFootball'
    articles = load_data('newsFootball.json')
    word_frequencies = get_word_frequencies(articles)
    sorted_frequencies = word_frequencies.most_common()
    plot_zipf_law(sorted_frequencies, name)
    save_top_words(sorted_frequencies, name + '.txt')

    name = 'newsHockey'
    articles = load_data('newsHockey.json')
    word_frequencies = get_word_frequencies(articles)
    sorted_frequencies = word_frequencies.most_common()
    plot_zipf_law(sorted_frequencies, name)
    save_top_words(sorted_frequencies, name + '.txt')


if __name__ == "__main__":
    main()
