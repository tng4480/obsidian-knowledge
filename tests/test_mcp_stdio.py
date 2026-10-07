"""End-to-end over a real stdio pipe: what Codex actually does."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

LAUNCHER = Path(__file__).parent / "_stdio_launcher.py"


def run_session(vault: Path, index_dir: Path, scenario, read_only: bool = False):
    env = {"ACADMEM_VAULT": str(vault), "ACADMEM_INDEX_DIR": str(index_dir), "PYTHONIOENCODING": "utf-8"}
    if read_only:
        env["STDIO_READ_ONLY"] = "1"
    params = StdioServerParameters(command=sys.executable, args=[str(LAUNCHER)], env=env)

    async def go():
        with open(index_dir.parent / "stderr.log", "w", encoding="utf-8") as errlog:
            async with stdio_client(params, errlog=errlog) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    return await scenario(session)

    return asyncio.run(asyncio.wait_for(go(), timeout=120))


def test_server_speaks_the_protocol_over_stdio(vault, config):
    async def scenario(session):
        tools = {t.name for t in (await session.list_tools()).tools}
        found = await session.call_tool("search_academic_memory", {"query": "posterior probability of disease", "module": "3B1"})
        created = await session.call_tool("create_note", {"path": "3B1/Notes/Over stdio.md", "content": "# Stdio\n\nwritten via the transport"})
        refused = await session.call_tool("create_note", {"path": "../nope.md", "content": "x"})
        return tools, found, created, refused

    tools, found, created, refused = run_session(vault, config.index_dir, scenario)
    assert {"search_academic_memory", "create_note", "get_context"} <= tools

    assert not found.is_error
    top = found.structured_content["results"][0]
    assert top["source"] == "3B1/Questions/Bayes.md" and top["text"]  # provenance survives the wire

    assert not created.is_error and (vault / "3B1" / "Notes" / "Over stdio.md").is_file()

    assert refused.is_error
    assert "PathNotAllowed" in refused.content[0].text  # the model can read why
    assert not (vault.parent / "nope.md").exists()


def test_read_only_server_over_stdio_has_no_write_tools(vault, config):
    async def scenario(session):
        return {t.name for t in (await session.list_tools()).tools}

    tools = run_session(vault, config.index_dir, scenario, read_only=True)
    assert "search_academic_memory" in tools
    assert not tools & {"create_note", "append_to_note", "replace_section"}


@pytest.mark.parametrize("flag", [[], ["--no-watch"]])
def test_main_refuses_a_missing_vault(tmp_path, flag, capsys):
    from acadmem.mcp_server import main

    assert main(["--vault", str(tmp_path / "does-not-exist"), "--index-dir", str(tmp_path / "i"), *flag]) == 2
    assert "vault not found" in capsys.readouterr().err
