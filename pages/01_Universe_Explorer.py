from __future__ import annotations

import streamlit as st

from src import config
from src.data import sources


@st.cache_data(ttl=config.CACHE_TTL)
def load_indices():
    return sources.load_indices(config.INDEX_URLS)


def main() -> None:
    st.title("Universe Explorer")
    st.write("Inspect the indices and constituents available to the optimiser.")

    df = load_indices()
    st.dataframe(df.sort_values(["index_name", "company"]).reset_index(drop=True))


if __name__ == "__main__":
    main()
