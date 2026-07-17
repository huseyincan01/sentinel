from src.ui.app import build_app

def test_ui():
    app = build_app()
    assert app is not None
