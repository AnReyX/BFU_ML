import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_absolute_error, r2_score, mean_absolute_percentage_error

data = pd.read_csv('lab2_1_DS.csv')
print("Датасет lab2_1_DS.csv:")
print(data.head(), "\n")

X_1d = data[['experience']]
y = data['income']
X_train_1d, X_test_1d, y_train, y_test = train_test_split(X_1d, y, test_size=0.3, random_state=42)

models_1d = {
    'Linear Regression': LinearRegression(),
    'Ridge (alpha=10.0)': Ridge(alpha=10.0, random_state=42),
    'Lasso (alpha=100.0)': Lasso(alpha=100.0, random_state=42),
    'Elastic Net (alpha=1.0, l1=0.5)': ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=42)
}

results_1d = []
x_line = pd.DataFrame({'experience': np.linspace(data['experience'].min(), data['experience'].max(), 200)})

plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.scatter(X_train_1d['experience'], y_train, color='blue', s=55, alpha=0.8, label='Train set')
plt.scatter(X_test_1d['experience'], y_test, color='red', s=65, marker='D', alpha=0.9, label='Test set')

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd']
linestyles = ['-', '--', '-.', ':']

print("Параметры моделей (признак experience):")
for (name, model), color, ls in zip(models_1d.items(), colors, linestyles):
    model.fit(X_train_1d, y_train)
    y_pred_test = model.predict(X_test_1d)
    y_pred_train = model.predict(X_train_1d)
    mae = mean_absolute_error(y_test, y_pred_test)
    r2 = r2_score(y_test, y_pred_test)
    mape = mean_absolute_percentage_error(y_test, y_pred_test)
    r2_train = r2_score(y_train, y_pred_train)

    print(f"{name:32s} | intercept = {model.intercept_:10.2f}, coef = {model.coef_[0]:8.2f}")
    results_1d += [{
        'Модель': name,
        'Intercept (b0)': round(model.intercept_, 2),
        'Coef (b1)': round(model.coef_[0], 2),
        'MAE (Test)': round(mae, 2),
        'R2 (Test)': round(r2, 4),
        'MAPE (Test), %': round(mape * 100, 2),
        'R2 (Train)': round(r2_train, 4)
    }]

    y_line = model.predict(x_line)
    plt.plot(x_line['experience'], y_line, color=color, linestyle=ls, linewidth=2.2, label=name)

plt.title('Линейная регрессия (доход от опыта работы)', fontsize=12, fontweight='bold')
plt.xlabel('Опыт работы (в годах)', fontsize=11)
plt.ylabel('Доход', fontsize=11)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(fontsize=9)


X_2d = data[['age', 'experience']]
X_train_2d, X_test_2d, y_train_2d, y_test_2d = train_test_split(X_2d, y, test_size=0.3, random_state=42)

models_2d = {
    'Linear Regression': LinearRegression(),
    'Ridge (alpha=1.0)': Ridge(alpha=1.0, random_state=42),
    'Lasso (alpha=1.0)': Lasso(alpha=1.0, random_state=42),
    'Elastic Net (alpha=1.0, l1=0.5)': ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=42)
}

results_2d = []
mean_age = data['age'].mean()
x_line_2d = pd.DataFrame({
    'age': np.full(200, mean_age),
    'experience': np.linspace(data['experience'].min(), data['experience'].max(), 200)
})

plt.subplot(1, 2, 2)
plt.scatter(X_train_2d['experience'], y_train_2d, color='blue', s=55, alpha=0.8, label='Train set')
plt.scatter(X_test_2d['experience'], y_test_2d, color='red', s=65, marker='D', alpha=0.9, label='Test set')

print("\nПараметры моделей (признаки age, experience):")
for (name, model), color, ls in zip(models_2d.items(), colors, linestyles):
    model.fit(X_train_2d, y_train_2d)
    y_pred_test = model.predict(X_test_2d)
    y_pred_train = model.predict(X_train_2d)
    mae = mean_absolute_error(y_test_2d, y_pred_test)
    r2 = r2_score(y_test_2d, y_pred_test)
    mape = mean_absolute_percentage_error(y_test_2d, y_pred_test)
    r2_train = r2_score(y_train_2d, y_pred_train)

    print(f"{name:32s} | intercept = {model.intercept_:10.2f}, coef = [{model.coef_[0]:7.2f}, {model.coef_[1]:8.2f}]")
    results_2d += [{
        'Модель': name,
        'Intercept (b0)': round(model.intercept_, 2),
        'Coef age': round(model.coef_[0], 2),
        'Coef exp': round(model.coef_[1], 2),
        'MAE (Test)': round(mae, 2),
        'R2 (Test)': round(r2, 4),
        'MAPE (Test), %': round(mape * 100, 2),
        'R2 (Train)': round(r2_train, 4)
    }]

    y_line_2d = model.predict(x_line_2d)
    plt.plot(x_line_2d['experience'], y_line_2d, color=color, linestyle=ls, linewidth=2.2,
             label=f"{name} (при ср. возрасте)")

plt.title('Многофакторная регрессия (проекция при среднем возрасте)', fontsize=12, fontweight='bold')
plt.xlabel('Опыт работы (в годах)', fontsize=11)
plt.ylabel('Доход', fontsize=11)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(fontsize=9)
plt.tight_layout()
plt.savefig('lab2_1_plot.png', dpi=150)

df_res_1d = pd.DataFrame(results_1d)
df_res_2d = pd.DataFrame(results_2d)

print("\nТаблица метрик (признак experience):")
print(df_res_1d.to_string(index=False))

print("\nТаблица метрик (признаки: age, experience):")
print(df_res_2d.to_string(index=False))

best_1d = df_res_1d.sort_values(by='R2 (Test)', ascending=False).iloc[0]
best_2d = df_res_2d.sort_values(by='R2 (Test)', ascending=False).iloc[0]
print("\nВывод по лабораторной 2.1:")
print(
    f"1. При использовании одного признака experience лучшей на тестовой выборке оказалась модель '{best_1d['Модель']}'"
    f" (MAE = {best_1d['MAE (Test)']}, R2 = {best_1d['R2 (Test)']}, MAPE = {best_1d['MAPE (Test), %']}%)")
print(
    f"2. При использовании обоих признаков age и experience качество моделей возросло: R2 увеличивается с 0.90-0.92 до "
    f"0.944-0.946, а ошибка MAE снижается с 1078-1247 до 834-840 ден. ед. Лучший R2 на тесте показала модель '{best_2d['Модель']}'"
    f" (R2 = {best_2d['R2 (Test)']}), а минимальную абсолютную ошибку MAE = 834.00 показала модель LASSO")
print(
    "3. Регуляризация (Ridge, Lasso, Elastic Net) штрафует модель за слишком большие веса, снижая риск переобучения на "
    "малых выборках и повышая обобщающую способность на новых тестовых данных")
print("\nГрафик сохранён в файле lab2_1_plot.png")