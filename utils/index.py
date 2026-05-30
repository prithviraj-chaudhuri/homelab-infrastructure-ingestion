import ast
import hashlib
import uuid
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

def make_id(text: str) -> str:
    """Generate a stable UUID string for a file chunk.

    Qdrant point IDs must be either unsigned integers or UUIDs.
    We use UUIDv5 to create a deterministic UUID from the input text.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_URL, text))


def build_embedding_text(file_path: str, chunk: Dict) -> str:
    return f"""
File: {file_path}

Symbol: {chunk['symbol']}
Type: {chunk['type']}
Lines: {chunk['start_line']}-{chunk['end_line']}

Code:
{chunk['code']}
"""


def read_text_file(file_path: str) -> str:
    """Read a file as text with fallbacks for different encodings.

    Opens the file in binary mode and attempts common decodings so the
    caller can handle YAML, shell scripts, and other text file types.
    Falls back to latin-1 with replacement if all attempts fail.
    """
    with open(file_path, "rb") as f:
        raw = f.read()

    for enc in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
        try:
            return raw.decode(enc)
        except Exception:
            continue

    logger.warning(
        "Unable to decode %s with common encodings; using latin-1 with replacement",
        file_path,
    )
    return raw.decode("latin-1", errors="replace")


def extract_python_chunks(file_path: str) -> List[Dict]:
    source = read_text_file(file_path)
    logger.debug("Extracting chunks from file %s", file_path)

    tree = ast.parse(source)
    lines = source.splitlines()

    chunks = []

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            start = node.lineno
            end = getattr(node, "end_lineno", start)

            chunks.append({
                "type": "function",
                "symbol": node.name,
                "start_line": start,
                "end_line": end,
                "code": "\n".join(lines[start - 1:end]),
            })

        elif isinstance(node, ast.ClassDef):
            start = node.lineno
            end = getattr(node, "end_lineno", start)

            chunks.append({
                "type": "class",
                "symbol": node.name,
                "start_line": start,
                "end_line": end,
                "code": "\n".join(lines[start - 1:end]),
            })

    return chunks


def extract_chunks(file_path: str) -> List[Dict]:
    """Extract chunks from arbitrary text files.

    - Python files: uses AST-based splitting (functions/classes).
    - YAML: splits on document separator `---`.
    - Dockerfile: splits on `FROM` (multi-stage builds).
    - Shell scripts: attempts to split by function definitions.
    - Fallback: returns a single chunk containing the whole file.
    """
    from pathlib import Path
    import re

    suffix = Path(file_path).suffix.lower()
    name = Path(file_path).name

    if suffix == ".py":
        return extract_python_chunks(file_path)

    source = read_text_file(file_path)

    # YAML documents
    if suffix in (".yml", ".yaml"):
        parts = [p for p in source.split("\n---\n") if p.strip()]
        if not parts:
            parts = [source]

        chunks = []
        line_offset = 1
        for i, part in enumerate(parts, start=1):
            lines = part.splitlines()
            chunks.append({
                "type": "yaml",
                "symbol": f"{name}:doc{i}",
                "start_line": line_offset,
                "end_line": line_offset + len(lines) - 1,
                "code": part,
            })
            line_offset += len(lines) + 1
        return chunks

    # Dockerfile (split by FROM for multi-stage)
    if name.lower().startswith("dockerfile") or "dockerfile" in name.lower() or suffix == "":
        parts = []
        cur: List[str] = []
        for line in source.splitlines():
            if line.strip().upper().startswith("FROM "):
                if cur:
                    parts.append("\n".join(cur))
                cur = [line]
            else:
                cur.append(line)
        if cur:
            parts.append("\n".join(cur))

        if not parts:
            parts = [source]

        chunks = []
        line_offset = 1
        for i, part in enumerate(parts, start=1):
            lines = part.splitlines()
            chunks.append({
                "type": "dockerfile",
                "symbol": f"{name}:stage{i}",
                "start_line": line_offset,
                "end_line": line_offset + len(lines) - 1,
                "code": part,
            })
            line_offset += len(lines)
        return chunks

    # Shell scripts: attempt to split by function definitions
    if suffix in (".sh", ".bash"):
        func_re = re.compile(r"^\s*(?:function\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*(?:\(\))?\s*{")
        lines = source.splitlines()
        chunks: List[Dict] = []
        cur: List[str] = []
        cur_name = None
        start = 1
        for i, line in enumerate(lines, start=1):
            m = func_re.match(line)
            if m:
                if cur:
                    chunks.append({
                        "type": "shell",
                        "symbol": cur_name or f"{name}:part{len(chunks)+1}",
                        "start_line": start,
                        "end_line": i - 1,
                        "code": "\n".join(cur),
                    })
                cur = [line]
                cur_name = m.group(1)
                start = i
            else:
                cur.append(line)

        if cur:
            chunks.append({
                "type": "shell",
                "symbol": cur_name or f"{name}:part{len(chunks)+1}",
                "start_line": start,
                "end_line": start + len(cur) - 1,
                "code": "\n".join(cur),
            })

        return chunks

    # Fallback: single chunk with the full file contents
    lines = source.splitlines()
    return [
        {
            "type": "file",
            "symbol": name,
            "start_line": 1,
            "end_line": len(lines),
            "code": source,
        }
    ]
