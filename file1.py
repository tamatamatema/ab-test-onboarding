import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

# Настройки графиков
sns.set_theme(style="whitegrid")

# 1. Генерация синтетических данных (симуляция логов приложения)
np.random.seed(42)
n_users = 5000

data = {
    'user_id': range(1, n_users + 1),
    'group': np.random.choice(['control', 'test'], size=n_users, p=[0.5, 0.5]),
    # Контрольная группа: базовая конверсия 12%. Тестовая: 15%.
    'is_converted': np.zeros(n_users, dtype=int),
    'retention_day_7': np.zeros(n_users, dtype=int)
}

df = pd.DataFrame(data)

# Симулируем поведение пользователей
control_mask = df['group'] == 'control'
test_mask = df['group'] == 'test'

df.loc[control_mask, 'is_converted'] = np.random.binomial(1, 0.12, size=control_mask.sum())
df.loc[test_mask, 'is_converted'] = np.random.binomial(1, 0.15, size=test_mask.sum())

# Retention (возвращаемость) зависит от того, сконвертировался ли юзер
df.loc[df['is_converted'] == 1, 'retention_day_7'] = np.random.binomial(1, 0.40, size=(df['is_converted'] == 1).sum())
df.loc[df['is_converted'] == 0, 'retention_day_7'] = np.random.binomial(1, 0.10, size=(df['is_converted'] == 0).sum())

print("Превью данных:")
display(df.head())