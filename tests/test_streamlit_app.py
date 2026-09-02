# Copyright 2026 Ahmad Mujtaba
# Licensed under the Apache License, Version 2.0 (the "License");

"""Headless smoke tests for the Streamlit entry point."""

from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_app_starts_with_demo_without_loading_model() -> None:
  app = Path(__file__).parents[1] / "streamlit_app.py"
  test_app = AppTest.from_file(str(app), default_timeout=20).run()
  assert not test_app.exception
  assert test_app.title[0].value == "TimesFM-3 explorer"
  assert test_app.dataframe
  assert not test_app.session_state["runs"]
  submit = next(button for button in test_app.button if button.label == "Run forecast")
  assert submit.disabled
  test_app.checkbox[0].set_value(True).run()
  submit = next(button for button in test_app.button if button.label == "Run forecast")
  assert not submit.disabled
