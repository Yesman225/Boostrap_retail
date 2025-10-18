"""Typed accessors around Streamlit session state."""
from __future__ import annotations

from typing import Iterable

import numpy as np
import streamlit as st


def get_selected_tickers(default: Iterable[str] | None = None) -> list[str]:
    if "selected_tickers" not in st.session_state:
        st.session_state.selected_tickers = list(default or [])
    return st.session_state.selected_tickers


def set_selected_tickers(tickers: Iterable[str]) -> None:
    st.session_state.selected_tickers = list(tickers)


def get_weights(default: np.ndarray | None = None) -> np.ndarray | None:
    if "weights" not in st.session_state:
        st.session_state.weights = default.copy() if default is not None else None
    return st.session_state.weights


def set_weights(weights: np.ndarray | None) -> None:
    st.session_state.weights = None if weights is None else np.asarray(weights)
