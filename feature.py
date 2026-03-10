"""
Feature extraction functions for hh.ru resume data.

Each private function transforms one raw text column into a numeric Series.
The public ``build_xy`` function assembles the full feature matrix and target.

Feature list (11 total)
-----------------------
is_male, age, town, full_time, has_car, higher_education,
remote_work, experience, desired_position, last_position, last_employer
"""

import numpy as np
import pandas as pd


def _extract_salary(series: pd.Series) -> pd.Series:
    """Extract the first numeric salary value from a text field.

    Takes only the first continuous digit sequence so that ranges like
    '50 000 — 80 000 руб.' yield 50000 rather than the concatenated
    monster '5000080000'.
    """
    return (
        series.astype(str)
        .str.extract(r"(\d[\d\s]*\d|\d+)")[0]
        .str.replace(r"\s+", "", regex=True)
        .replace("nan", pd.NA)
        .replace("", pd.NA)
        .astype("Int64")
    )


def _extract_age(series: pd.Series) -> pd.Series:
    """Extract age in years from a combined gender/age string."""
    return series.str.extract(r"(\d+)\s*год")[0].astype("Int64")


def _extract_time(series: pd.Series) -> pd.Series:
    """Return 1 if the employment type is full-time, else 0."""
    return series.str.contains("полная", case=False, na=False).astype(int)


def _extract_gender(series: pd.Series) -> pd.Series:
    """Return 1 if the candidate is male, else 0."""
    return series.str.contains("Мужчина", na=False).astype(int)


def _extract_town(series: pd.Series) -> pd.Series:
    """Encode city name as an integer category code."""
    return series.str.split(",").str[0].astype("category").cat.codes


def _extract_car(series: pd.Series) -> pd.Series:
    """Return 1 if the candidate mentions car ownership."""
    return series.str.contains("авто", case=False, na=False).astype(int)


def _extract_education(series: pd.Series) -> pd.Series:
    """Return 1 if the candidate has higher education."""
    return series.str.contains("Высшее", case=False, na=False).astype(int)


def _extract_remote(series: pd.Series) -> pd.Series:
    """Return 1 if the candidate is open to remote work."""
    return series.str.contains("удал", case=False, na=False).astype(int)


def _extract_experience(series: pd.Series) -> pd.Series:
    """Extract total years of experience; returns NA when not found."""
    return series.str.extract(r"(\d+)\s*лет")[0].astype("Int64")


def _encode_category(series: pd.Series) -> pd.Series:
    """Encode a free-text column as integer category codes.

    Normalises whitespace and lowercases before encoding so that
    'Программист' and 'программист ' map to the same code.
    Null and 'не указано' values get code -1.
    """
    normalised = (
        series.astype(str)
        .str.strip()
        .str.lower()
        .replace("nan", pd.NA)
        .replace("не указано", pd.NA)
    )
    return normalised.astype("category").cat.codes


def build_xy(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Build feature matrix X and target vector y from a cleaned DataFrame.

    Parameters
    ----------
    df:
        Cleaned resume DataFrame produced by the handler chain.

    Returns
    -------
    (X, y) as int64 NumPy arrays ready for model training.
    """
    y = _extract_salary(df["ЗП"])

    X = pd.DataFrame(
        {
            "is_male": _extract_gender(df["Пол, возраст"]),
            "age": _extract_age(df["Пол, возраст"]),
            "town": _extract_town(df["Город"]),
            "full_time": _extract_time(df["Занятость"]),
            "has_car": _extract_car(df["Авто"]),
            "higher_education": _extract_education(df["Образование и ВУЗ"]),
            "remote_work": _extract_remote(df["График"]),
            "experience": _extract_experience(
                df["Опыт (двойное нажатие для полной версии)"]
            ),
            # High-signal text columns — category codes.
            # 'desired_position' is the strongest predictor of salary
            # and should give the biggest R² lift.
            "desired_position": _encode_category(df["Ищет работу на должность:"]),
            "last_position": _encode_category(df["Последеняя/нынешняя должность"]),
            "last_employer": _encode_category(df["Последенее/нынешнее место работы"]),
        }
    )

    # Remove rows with missing or clearly invalid salaries.
    # Realistic hh.ru range: 10 000 – 1 000 000 RUB/month.
    mask = y.notna() & (y >= 10_000) & (y <= 1_000_000)
    X = X[mask].fillna(0)
    y = y[mask]

    return (
        X.astype("int64").to_numpy(),
        y.astype("int64").to_numpy(),
    )
