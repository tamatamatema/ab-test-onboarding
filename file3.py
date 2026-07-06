# 3. Статистический тест (Z-test для пропорций)
from statsmodels.stats.proportion import proportions_ztest

successes = metrics['conversions'].values
nobs = metrics['users'].values

stat, p_value = proportions_ztest(successes, nobs)

print(f"Z-statistic: {stat:.4f}")
print(f"P-value: {p_value:.4f}")

if p_value < 0.05:
    print("Вывод: Разница статистически значима. Отвергаем нулевую гипотезу.")
else:
    print("Вывод: Разница не значима. Нулевая гипотеза сохраняется.")