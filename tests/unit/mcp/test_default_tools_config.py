from pathlib import Path


def test_fetch_server_uses_mcp_one() -> None:
    root = Path(__file__).resolve().parents[3]
    config = (root / "microagents/default-tools.md").read_text()

    assert 'args: ["--with", "mcp<2", "mcp-server-fetch"]' in config
