import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error

data = pd.read_csv('lab2_2_DS.csv')
print("Датасет lab2_2_DS.csv:")
print(data.head(), "\n")

X = data[['Temperature (°C)']]
y = data['Ice Cream Sales (units)']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

models = {'Linear Regression (deg=1)': LinearRegression()}
for d in [2, 3, 4, 5, 10]:
    models[f'Polynomial Regression (deg={d})'] = make_pipeline(PolynomialFeatures(degree=d, include_bias=False),
                                                               LinearRegression())

plt.figure(figsize=(11, 7))
plt.scatter(X_train['Temperature (°C)'], y_train, color='blue', s=50, alpha=0.8, label='Train set')
plt.scatter(X_test['Temperature (°C)'], y_test, color='red', s=65, marker='D', alpha=0.9, label='Test set')

x_grid = pd.DataFrame({
    'Temperature (°C)': np.linspace(X['Temperature (°C)'].min(), X['Temperature (°C)'].max(), 300)
})

colors = ['#d62728', '#1f77b4', '#2ca02c', '#ff7f0e', '#9467bd', '#8c564b']
linestyles = ['--', '-', '-', '-.', ':', ':']
linewidths = [2.0, 2.5, 2.5, 2.0, 1.8, 1.8]

results = []

for (name, model), color, ls, lw in zip(models.items(), colors, linestyles, linewidths):
    model.fit(X_train, y_train)
    y_pred_test = model.predict(X_test)
    y_pred_train = model.predict(X_train)
    mse_test = mean_squared_error(y_test, y_pred_test)
    r2_test = r2_score(y_test, y_pred_test)
    mape_test = mean_absolute_percentage_error(y_test, y_pred_test)
    mse_train = mean_squared_error(y_train, y_pred_train)
    r2_train = r2_score(y_train, y_pred_train)

    results += [{
        'Модель': name,
        'MSE (Test)': round(mse_test, 3),
        'R2 (Test)': round(r2_test, 4),
        'MAPE (Test), %': round(mape_test * 100, 2),
        'MSE (Train)': round(mse_train, 3),
        'R2 (Train)': round(r2_train, 4)
    }]

    y_grid = model.predict(x_grid)
    plt.plot(x_grid['Temperature (°C)'], y_grid, color=color, linestyle=ls, linewidth=lw, label=name)

plt.title('Полиномиальная и линейная регрессии (продажи мороженого от температуры)', fontsize=13, fontweight='bold')
plt.xlabel('Температура (°C)', fontsize=11)
plt.ylabel('Продажи мороженого (units)', fontsize=11)
plt.ylim(-5, 50)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(fontsize=9.5, loc='upper center')
plt.tight_layout()
plt.savefig('lab2_2_plot.png', dpi=150)

data_results = pd.DataFrame(results)
print("Таблица метрик моделей полиномиальной и линейной регрессии:")
print(data_results.to_string(index=False))

best_mse_model = data_results.sort_values(by='MSE (Test)', ascending=True).iloc[0]
print("\nВывод по лабораторной 2.2:")
print("1. Линейная регрессия не справляется с задачей (R2 на тесте = 0.0135, MSE = 131.08), "
      "так как зависимость в данных нелинейная (U-образная). Наблюдается сильное недообучение")
print(
    f"2. Полиномы 2-й и 3-й степени резко улучшают качество: на тестовой выборке лучшей по MSE и R2 является модель '{best_mse_model['Модель']}' "
    f"(MSE = {best_mse_model['MSE (Test)']}, R2 = {best_mse_model['R2 (Test)']}, MAPE = {best_mse_model['MAPE (Test), %']}%)."
    f" Полином 2-й степени даёт оптимальный результат (MSE = 13.360, R2 = 0.8995) при меньшей сложности модели")
print(
    "3. При увеличении степени полинома (4, 5, 10) ошибка на обучающей выборке продолжает падать (R2 Train растет до 0.9706),"
    " однако на тестовой выборке MSE начинает расти (с 13.134 до 18.766 при deg=5 и 17.164 при deg=10)."
    "Наблюдается переобучение: полином высокой степени начинает подстраиваться под случайный шум обучающей выборки. "
    "Лучший выбор — полином 2-й или 3-й степени")
print("\nГрафик сохранён в файле lab2_2_plot.png")