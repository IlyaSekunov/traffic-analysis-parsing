"""
Chain-of-responsibility handlers for DataFrame preprocessing.

Each handler performs one transformation step and passes the result to the
next handler in the chain.  Build the chain with ``set_next_handler`` and
trigger it by calling ``handle`` on the first handler.
"""

import re
from abc import ABC, abstractmethod

import pandas as pd


class DataFrameHandler(ABC):
    """Abstract base class for all DataFrame preprocessing handlers."""

    def __init__(self) -> None:
        self._next_handler: "DataFrameHandler | None" = None

    def set_next_handler(
            self, handler: "DataFrameHandler"
    ) -> "DataFrameHandler":
        """Attach the next handler and return it for chaining."""
        self._next_handler = handler
        return handler

    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process *df* and pass it down the chain."""
        df = self._process(df)
        if self._next_handler:
            return self._next_handler.handle(df)
        return df

    @abstractmethod
    def _process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Implement the actual transformation in subclasses."""


class CleanDataHandler(DataFrameHandler):
    """Strip invisible Unicode characters from all string columns."""

    _INVISIBLE = re.compile(r"[\u00a0\u200b\ufeff]")

    def _process(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        for col in result.select_dtypes(include="object").columns:
            result[col] = result[col].apply(
                lambda x: self._INVISIBLE.sub(" ", str(x)).strip()
                if pd.notna(x)
                else x
            )
        return result


class DropUselessDataHandler(DataFrameHandler):
    """Remove rows with any NA value and exact duplicates."""

    def _process(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.dropna().reset_index(drop=True)  # was incorrectly drop_na()
        df = df.drop_duplicates().reset_index(drop=True)
        return df


class DropEmptySalaryHandler(DataFrameHandler):
    """Keep only rows that have a non-null salary value."""

    def _process(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[df["ЗП"].notna()].reset_index(drop=True)
