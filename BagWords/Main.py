import json
import pandas as pd
import numpy as np
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

# Создание матрицы терм-документ
def create_term_document_matrix(documents, term_dict):
    matrix = []
    term_list = list(term_dict.keys())
    for doc in documents:
        words = doc.split()
        word_count = defaultdict(int)
        for word in words:
            word_count[word] += 1
        row = [word_count.get(term, 0) for term in term_list]
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

    # Создание матрицы терм-документ
    term_dict = create_term_dict(all_documents)

    # Создание матрицы терм-документ
    term_document_matrix = create_term_document_matrix(all_documents, term_dict)
    term_document_matrix.to_csv('term_document_matrix.csv', index=False)
    print(term_document_matrix)

if __name__ == "__main__":
    main()
