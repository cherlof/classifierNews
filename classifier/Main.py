import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay

# Загрузка матриц
M1 = pd.read_csv('term_document_matrix.csv')  # Мешок слов
M2 = pd.read_csv('tfidf_matrix.csv')  # Tf-Idf

# Создание меток (примерные метки, замените на ваши метки)
# Предполагаем, что у нас три класса: 0 - Crypto, 1 - Football, 2 - Hockey
num_samples = M1.shape[0]  # Общее количество образцов
labels = [0] * (num_samples // 3) + [1] * (num_samples // 3) + [2] * (num_samples // 3)

# Убедимся, что количество меток соответствует количеству образцов
assert len(labels) == num_samples, "Количество меток не соответствует количеству образцов"


# Применение PCA
def apply_pca(matrix, n_components=2):
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(matrix)
    return principal_components


# Применение PCA на матрицах M1 и M2
M1_pca = apply_pca(M1, n_components=5)
M2_pca = apply_pca(M2, n_components=5)

# Разделение данных на тренировочные и тестовые
X_train_M1, X_test_M1, y_train, y_test = train_test_split(M1_pca, labels, test_size=0.2, random_state=42)
X_train_M2, X_test_M2, y_train, y_test = train_test_split(M2_pca, labels, test_size=0.2, random_state=42)

# Обучение модели Random Forest на матрице M1 с PCA
model_M1 = RandomForestClassifier(n_estimators=100, random_state=42)
model_M1.fit(X_train_M1, y_train)
y_pred_M1 = model_M1.predict(X_test_M1)


accuracy_M1 = accuracy_score(y_test, y_pred_M1)
report_M1 = classification_report(y_test, y_pred_M1)
# Оценка модели на матрице M1 с PCA
print("Матрица M1 с PCA (мешок слов) и Random Forest:")
print("Accuracy:", accuracy_M1)
print(report_M1)

with open('classification_results_M1.txt', 'w') as file:
    file.write(f"Матрица M1 с PCA (мешок слов) и Random Forest:\n")
    file.write(f"Accuracy: {accuracy_M1}\n")
    file.write(report_M1)

# Отображение и сохранение матрицы ошибок для M1 с PCA
disp_M1 = ConfusionMatrixDisplay.from_estimator(model_M1, X_test_M1, y_test,
                                                display_labels=['Crypto', 'Football', 'Hockey'], cmap=plt.cm.Blues)
disp_M1.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix for M1 with PCA and Random Forest")
plt.savefig('confusion_matrix_M1_PCA_RF.png')
plt.show()

# Обучение модели Random Forest на матрице M2 с PCA
model_M2 = RandomForestClassifier(n_estimators=100, random_state=42)
model_M2.fit(X_train_M2, y_train)
y_pred_M2 = model_M2.predict(X_test_M2)

accuracy_M2 = accuracy_score(y_test, y_pred_M2)
report_M2 = classification_report(y_test, y_pred_M2)
# Оценка модели на матрице M2 с PCA
print("Матрица M2 с PCA (Tf-Idf) и Random Forest:")
print("Accuracy:",accuracy_M2)
print(report_M2)

with open('classification_results_M2.txt', 'w') as file:
    file.write(f"Матрица M2 с PCA (Tf-Idf) и Random Forest:\n")
    file.write(f"Accuracy: {accuracy_M2}\n")
    file.write(report_M2)

# Отображение и сохранение матрицы ошибок для M2 с PCA
disp_M2 = ConfusionMatrixDisplay.from_estimator(model_M2, X_test_M2, y_test,
                                                display_labels=['Crypto', 'Football', 'Hockey'], cmap=plt.cm.Blues)
disp_M2.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix for M2 with PCA and Random Forest")
plt.savefig('confusion_matrix_M2_PCA_RF.png')
plt.show()


# Визуализация 2D карты классификации для обучающих и тестовых данных
def plot_classification_map(X_train, X_test, y_train, y_test, model, title, filename):
    plt.figure(figsize=(10, 6))
    markers = ['o', 's', '^']
    classes = ['Crypto', 'Football', 'Hockey']

    # Визуализация обучающих данных
    for i, class_label in enumerate(np.unique(y_train)):
        plt.scatter(X_train[y_train == class_label, 0], X_train[y_train == class_label, 1],
                    label=f'Train {classes[class_label]}', marker=markers[i], alpha=0.7)

    # Визуализация тестовых данных
    for i, class_label in enumerate(np.unique(y_test)):
        plt.scatter(X_test[y_test == class_label, 0], X_test[y_test == class_label, 1],
                    label=f'Test {classes[class_label]}', edgecolor='k', facecolor='none', marker=markers[i], alpha=1)

    # Предсказанные классы для тестовых данных
    y_pred = model.predict(X_test)
    for i, class_label in enumerate(np.unique(y_pred)):
        plt.scatter(X_test[y_pred == class_label, 0], X_test[y_pred == class_label, 1],
                    label=f'Pred {classes[class_label]}', edgecolor='red', facecolor='none', marker=markers[i], alpha=1,
                    linestyle='--')

    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
    plt.title(title)
    plt.savefig(filename)
    plt.show()


# Визуализация карты классификации для M1 и M2
plot_classification_map(X_train_M1, X_test_M1, y_train, y_test, model_M1,
                        'Classification Map for M1 with PCA and Random Forest', 'classification_map_M1_PCA_RF.png')
plot_classification_map(X_train_M2, X_test_M2, y_train, y_test, model_M2,
                        'Classification Map for M2 with PCA and Random Forest', 'classification_map_M2_PCA_RF.png')
