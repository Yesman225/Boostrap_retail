"""Reusable Streamlit widgets."""
from __future__ import annotations

import streamlit as st

from src.config import HORIZONS


def horizon_selector(label: str = "Choose horizon:") -> str:
    return st.radio(label, HORIZONS, horizontal=True)


def multiselect_indices(options: list[str], label: str = "Indices") -> list[str]:
    return st.multiselect(label, options, default=options)
