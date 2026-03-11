"""Unit tests for main entrypoint."""

import runpy
import sys
from pathlib import Path


def test_main_runs(monkeypatch):
    class DummyStreamlit:
        def __init__(self) -> None:
            self.layout = None
            self.page_icon = None

        def set_page_config(self, layout: str, page_icon: str | None = None) -> None:
            self.layout = layout
            self.page_icon = page_icon

    dummy_st = DummyStreamlit()

    class DummyConfigModule:
        @staticmethod
        def load_config():
            return {"app": {"layout": "wide"}}

    class DummyDashboardModule:
        called_with = None

        @staticmethod
        def run_dashboard(config):
            DummyDashboardModule.called_with = config

    class DummyLoggingModule:
        called_with = None

        @staticmethod
        def configure_logging(config):
            DummyLoggingModule.called_with = config

    monkeypatch.setitem(sys.modules, "streamlit", dummy_st)
    monkeypatch.setitem(sys.modules, "src.config", DummyConfigModule)
    monkeypatch.setitem(sys.modules, "src.dashboard", DummyDashboardModule)
    monkeypatch.setitem(sys.modules, "src.logging_config", DummyLoggingModule)

    main_path = Path(__file__).resolve().parents[1] / "main.py"
    runpy.run_path(str(main_path), run_name="__main__")

    assert dummy_st.layout == "wide"
    assert dummy_st.page_icon == "img/finance_icon.png"
    assert DummyLoggingModule.called_with == {}
    assert DummyDashboardModule.called_with == {"app": {"layout": "wide"}}
