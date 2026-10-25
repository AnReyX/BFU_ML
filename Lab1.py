import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as ptc

file = input("Введите название .csv файла (lab1_DS.csv): ") # 1. Чтение данных из файла
data = pd.read_csv(file)

for col in data.columns: # 2. Вывод статистической информации
    print(f"\nНазвание столбца: {col}")
    print(f"Количество: {data[col].count()}")
    print(f"Минимум: {data[col].min()}")
    print(f"Максимум: {data[col].max()}")
    print(f"Среднее: {round(data[col].mean(), 4)}")

print("\nКакие столбцы доступны:")
for ind, col in enumerate(data.columns):
    print(f"{ind}: {col}")

x_col, y_col = int(input("Введите номер столбца для X: ")), int(input("Введите номер столбца для Y: "))
x, y = data.iloc[:, x_col], data.iloc[:, y_col]
x_mean, y_mean = x.mean(), y.mean() # 4. Метод наименьших квадратов
a = ((x - x_mean) * (y - y_mean)).sum() / ((x - x_mean) ** 2).sum()
b = y_mean - a * x_mean
y_predicted = a * x + b

print(f"\nИтоговое уравнение регрессии: y = {round(a, 4)}x + {round(b, 4)},")
print("Где параметры регрессионной прямой:")
print(f"a = {round(a, 4)}")
print(f"b = {round(b, 4)}")

# Создаём три графика и обозначаем оси
fig, axes = plt.subplots(1, 3, figsize=(18, 5)) # Графики для 3, 5, 6
for axe in axes:
    axe.grid(True)
    axe.set_xlabel(data.columns[x_col])
    axe.set_ylabel(data.columns[y_col])

# 3. Изображение исходных точек
axes[0].set_title("3. Исходные точки")
axes[0].scatter(x, y, color="blue")

x_line = np.linspace(x.min(), x.max(), 100)
y_line = a * x_line + b

# 5. Изображение графика регрессии
axes[1].set_title("4. Линейная регрессия по исходным точкам")
axes[1].scatter(x, y, color="blue") # Исходные точки
axes[1].plot(x_line, y_line, color="red") # Регрессионная прямая

# 6. Изображение квадратов ошибок
axes[2].set_title("5. Заштрихованные квадраты ошибок")
axes[2].scatter(x, y, color="blue")
axes[2].plot(x_line, y_line, color="red")

scale = (x.max() - x.min()) / (y.max() - y.min())
for xi, yi, ypi in zip(x, y, y_predicted):
    error = abs(yi - ypi)
    if not error: # Нет ошибки, не нужно рисовать квадрат
        continue
    length = error * scale
    square_x = xi - length if yi > ypi else xi # Если точка над линией — квадрат влево, иначе вправо (для красоты)
    square = ptc.Rectangle((square_x, min(yi, ypi)), length, error,
        facecolor="orange", edgecolor="yellow",
        alpha=0.4, hatch="///")
    axes[2].add_patch(square)
    axes[2].plot([xi, xi], [yi, ypi], color="orange", linewidth=1)

plt.tight_layout()
plt.show()