import numpy as np
import pandas as pd


def prepare_dataset(
    dataset,
    target_col,
    key_cols,
    log_transform=True,
    corr_threshold=0.85,
    min_corr_target=0.05,
    classification=False,
):
    """
    :param dataset:     датасет
    :target_col:        название целевой переменной
    :key_cols:          список всех ключевых признаков (IC50, CC50, SI и derived)
    :log_transform:     логарифмирование целевой переменной (для регрессии)
    :corr_threshold:    порог корреляции между признаками
    :min_corr_target:  минимальная корреляция признака с target
    :classification:    если True, не фильтруем признаки по корреляции с target

    :return dataset_prepared:   датафрейм (признаки + target)
    """

    data = dataset.copy()

    print(
        f"\n=== Подготовка для {target_col} ({'classification' if classification else 'regression'}) ==="
    )

    # Удаление константных признаков
    nunique = data.nunique()
    constant_features = nunique[nunique == 1].index
    data = data.drop(columns=constant_features)
    print(f"Удалено константных признаков: {len(constant_features)}")

    # Удаление всех key_cols, кроме целевого
    drop_key = [c for c in key_cols if c in data.columns and c != target_col]
    data = data.drop(columns=drop_key)
    print(f"Удалены другие ключевые признаки: {drop_key}")

    # Удаление выбросов (только для регрессии)
    if not classification:
        Q1 = data[target_col].quantile(0.25)
        Q3 = data[target_col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        before = data.shape[0]
        data = data[
            (data[target_col] >= lower_bound) & (data[target_col] <= upper_bound)
        ]
        print(f"Удалено выбросов: {before - data.shape[0]} строк")

    # Целевая переменная
    if log_transform and not classification:
        data[f"log_{target_col}"] = np.log10(data[target_col] + 1e-6)
        target_name = f"log_{target_col}"
    else:
        target_name = target_col

    # Формирование признаков
    drop_cols = [
        c
        for c in [target_col, f"log_{target_col}"]
        if c in data.columns and c != target_name
    ]
    features = data.drop(columns=drop_cols)
    print(f"Признаков перед фильтрацией: {features.shape[1]}")

    # Очистка признаков до корреляции
    stds = features.std()
    zero_std = stds[stds == 0].index
    features = features.drop(columns=zero_std)
    features = features.dropna(axis=1)
    print(f"После предварительной очистки (NaN/нулевая дисперсия): {features.shape[1]}")

    # Отбор признаков
    if not classification:
        corrs = features.corrwith(data[target_name]).abs().dropna()
        selected_features = corrs[corrs > min_corr_target].index
        features = features[selected_features]
        print(f"После отбора по корреляции с target: {features.shape[1]}")

    # Корреляция между признаками
    corr_matrix = features.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    drop_high_corr = [
        column for column in upper.columns if any(upper[column] > corr_threshold)
    ]
    features = features.drop(columns=drop_high_corr)
    print(f"После удаления сильно коррелирующих признаков: {features.shape[1]}")

    # Итоговый датасет
    dataset_prepared = pd.concat([features, data[target_name]], axis=1)

    dataset_prepared = dataset_prepared.loc[:, ~dataset_prepared.columns.duplicated()]

    return dataset_prepared
