"""
Main entry point for the Portfolio Builder Agent application.
"""

import streamlit as st

from src.config import load_config
from src.dashboard import run_dashboard
from src.logging_config import configure_logging

if __name__ == "__main__":
    config = load_config()
    configure_logging(config.get("logging", {}))
    st.set_page_config(
        page_title=config["app"]["title"], layout=config["app"]["layout"], page_icon="img/finance_icon.png"
    )
    run_dashboard(config)
