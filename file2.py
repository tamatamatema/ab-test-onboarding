# 2. Расчет базовых метрик по группам
metrics = df.groupby('group').agg(
    users=('user_id', 'count'),
    conversions=('is_converted', 'sum'),
    retained_users=('retention_day_7', 'sum')
).reset_index()

metrics['conversion_rate'] = metrics['conversions'] / metrics['users']
metrics['retention_rate_d7'] = metrics['retained_users'] / metrics['users']

print("Продуктовые метрики по группам:")
display(metrics.style.format({
    'conversion_rate': '{:.2%}',
    'retention_rate_d7': '{:.2%}'
}))

# Визуализация
plt.figure(figsize=(8, 5))
sns.barplot(data=metrics, x='group', y='conversion_rate', palette=['#ff9999', '#66b3ff'])
plt.title('Conversion Rate: Control vs Test')
plt.ylabel('Conversion Rate')
plt.show()