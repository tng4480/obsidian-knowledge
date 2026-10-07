"""Real-vault smoke test: launch ``acadmem-mcp`` exactly as Codex would and ask it real questions.

Read-only: it never calls a write tool, so your vault is untouched.

    uv run python eval/smoke_mcp.py [--vault PATH] [--no-watch]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import sys
import time
from pathlib import Path

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

QUESTIONS = [
    ("what does the 3F7 material say about Huffman coding", {"module": "3F7"}),
    ("derive the Viterbi path metric update", {"module": "3F4"}),
    ("common emitter bias current and stability", {}),
    ("importance sampling optimal proposal", {"file_type": "pdf"}),
    ("Tripos 2023 question on aliasing", {"path_prefix": "3F1/Questions"}),
]


def brief(hit: dict) -> str:
    where = f"p.{hit['page']}" if hit.get("page") else "md"
    return f"  {hit['score']:.3f}  {hit['source']} [{where}]  {hit['section'][:60]!r}"


async def main(args: argparse.Namespace) -> int:
    exe = shutil.which("acadmem-mcp") or str(Path(sys.executable).with_name("acadmem-mcp.exe"))
    server_args = ["--no-watch"] if args.no_watch else []
    env = {"PYTHONIOENCODING": "utf-8", "ACADMEM_VAULT": args.vault or os.environ["ACADMEM_VAULT"]}
    params = StdioServerParameters(command=exe, args=server_args, env=env)
    t0 = time.perf_counter()
    async with stdio_client(params, errlog=open(os.devnull, "w")) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            print(f"initialize: {time.perf_counter() - t0:.1f}s  ({exe})")
            tools = [t.name for t in (await session.list_tools()).tools]
            print("tools:", ", ".join(tools))

            status = (await session.call_tool("index_status", {})).structured_content
            print(f"index: {status['documents']} documents, {status['chunks']} chunks, failed={len(status['failed_files'])}, "
                  f"modules={status['documents_by_module']}")

            if args.delay:
                print(f"waiting {args.delay:.0f}s (a person does not ask a question the instant the server starts) ...")
                await asyncio.sleep(args.delay)

            first_hit = None
            for query, filters in QUESTIONS:
                t1 = time.perf_counter()
                res = await session.call_tool("search_academic_memory", {"query": query, "top_k": 3, **filters})
                dt = (time.perf_counter() - t1) * 1000
                assert not res.is_error, res.content[0].text
                out = res.structured_content
                print(f"\n{query!r} {filters or ''}  -> {out['count']} hits, {dt:.0f} ms ({out['mode']})")
                for hit in out["results"]:
                    print(brief(hit))
                first_hit = first_hit or (out["results"][0] if out["results"] else None)

            if first_hit:
                ctx = (await session.call_tool("get_context", {"chunk_id": first_hit["chunk_id"], "neighbors": 1, "section": True})).structured_content
                print(f"\nget_context({first_hit['chunk_id']}): {len(ctx['pieces'])} pieces, truncated={ctx['truncated']}")
                doc = (await session.call_tool("get_document", {"source_path": first_hit["source"]})).structured_content
                print(f"get_document: {doc['source']} pages={doc['page_count']} outline entries={len(doc['outline'])}")

            bad = await session.call_tool("create_note", {"path": "../../escape.md", "content": "x"})
            print(f"\nguard check: create_note('../../escape.md') -> is_error={bad.is_error}: {bad.content[0].text[:110]}")
    print(f"\ntotal wall time {time.perf_counter() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--vault")
    p.add_argument("--no-watch", action="store_true")
    p.add_argument("--delay", type=float, default=0, help="seconds to wait before the first query (shows the warm-up working)")
    raise SystemExit(asyncio.run(main(p.parse_args())))
