"""CSS for the floating chat button and the right-hand chat panel. Kept tiny and import-free so the closed state costs
nothing: the heavy chat code is only imported when the panel is opened."""
from __future__ import annotations

PANEL_W = "min(430px, 36vw)"

FAB_CSS = """
<style>
.st-key-chat_fab { position: fixed; right: 22px; bottom: 22px; z-index: 1000; width: auto !important; }
.st-key-chat_fab button { width: 58px; height: 58px; border-radius: 50%; font-size: 26px; padding: 0;
  box-shadow: 0 4px 14px rgba(0,0,0,.35); }
</style>
"""


def panel_css(dark: bool) -> str:
    bg, border = ("#0e1117", "rgba(250,250,250,.18)") if dark else ("#ffffff", "rgba(49,51,63,.2)")
    return f"""
<style>
.st-key-chat_panel {{ position: fixed; top: 3.6rem; right: 14px; width: {PANEL_W}; z-index: 900;
  max-height: calc(100vh - 4.6rem); overflow-y: auto; background: {bg}; border: 1px solid {border};
  border-radius: 14px; padding: 10px 12px 6px; box-shadow: 0 8px 28px rgba(0,0,0,.28); }}
[data-testid="stMainBlockContainer"] {{ padding-right: calc({PANEL_W} + 34px) !important; }}
@media (max-width: 900px) {{
  .st-key-chat_panel {{ left: 8px; right: 8px; width: auto; top: auto; bottom: 8px; max-height: 72vh; }}
  [data-testid="stMainBlockContainer"] {{ padding-right: 1rem !important; }}
}}
</style>
"""
