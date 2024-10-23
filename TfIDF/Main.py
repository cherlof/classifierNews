import json
import math
import pandas as pd
from collections import defaultdict

# Загрузка JSON данных
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

# Создание словаря терминов
def create_term_dict(documents):
    term_dict = defaultdict(int)
    for doc in documents:
        words = doc.split()
        for word in words:
            term_dict[word] += 1
    return term_dict

# Подсчет Term Frequency (TF)
def compute_tf(doc, term_dict):
    words = doc.split()
    tf = defaultdict(float)
    for word in words:
        tf[word] += 1 / len(words)  # нормализованный TF
    return tf

# Подсчет Inverse Document Frequency (IDF)
def compute_idf(documents, term_dict):
    idf = defaultdict(float)
    n_docs = len(documents)
    for term in term_dict:
        containing_docs = sum(1 for doc in documents if term in doc.split())
        idf[term] = math.log(n_docs / (1 + containing_docs)) + 1
    return idf

# Подсчет Tf-Idf
def compute_tfidf(tf, idf):
    tfidf = {term: tf[term] * idf[term] for term in tf}
    return tfidf

# Создание матрицы Tf-Idf
def create_tfidf_matrix(documents, term_dict):
    idf = compute_idf(documents, term_dict)
    matrix = []
    term_list = list(term_dict.keys())
    for doc in documents:
        tf = compute_tf(doc, term_dict)
        tfidf = compute_tfidf(tf, idf)
        row = [tfidf.get(term, 0) for term in term_list]
        matrix.append(row)
    return pd.DataFrame(matrix, columns=term_list)





def main():
    # Загрузка данных из всех файлов
    filenames = ['newsCryptoProcessed.json', 'newsFootballProcessed.json', 'newsHockeyProcessed.json']
    all_documents = []
    for filename in filenames:
        data = load_json(filename)
        documents = prepare_documents(data)
        all_documents.extend(documents)

    # Создание словаря терминов
    term_dict = create_term_dict(all_documents)

    # Создание матрицы Tf-Idf
    tfidf_matrix = create_tfidf_matrix(all_documents, term_dict)

    # Сохранение матрицы в CSV файл
    tfidf_matrix.to_csv('tfidf_matrix.csv', index=False)

    # Отображение матрицы
    print(tfidf_matrix)

if __name__ == "__main__":
    main()