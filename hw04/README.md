# Домашнее задание #4 — Линейная регрессия

## Описание

Полный цикл построения ML-модели: EDA → предобработка → инженерия признаков → обучение → оценка.

**Датасет:** `housing_data.csv` — характеристики жилых блоков Калифорнии  
**Задача:** регрессия — предсказать медианную стоимость дома (`MedHouseVal`)

## Структура проекта

```
hw4/
├── hw4_linear_regression.ipynb   # Основной ноутбук с исследованием
├── housing_data.csv               # Датасет
├── plots/                         # Сохранённые графики
│   ├── distributions.png
│   ├── correlation_matrix.png
│   ├── target_distribution.png
│   ├── coefficients.png
│   ├── model_comparison.png
│   └── predictions.png
└── README.md
```

## Результаты

| Модель | Test R² | Test RMSE |
|---|---|---|
| LinearRegression | ~0.931 | ~0.101 |
| Ridge (best alpha) | ~0.931 | ~0.101 |
| Lasso (best alpha) | ~0.931 | ~0.101 |

## Как запустить

```bash
pip install pandas scikit-learn seaborn matplotlib jupyter
jupyter notebook hw4_linear_regression.ipynb
```

## Ключевые выводы

- Сильнейший предиктор цены — медианный доход района (`MedInc`, corr ≈ 0.89)
- Feature engineering (log-трансформация, расстояние до города) улучшил качество
- Δ R² (Train − Test) < 0.003 — переобучения нет
- Ridge незначительно лучше LinearRegression за счёт регуляризации
