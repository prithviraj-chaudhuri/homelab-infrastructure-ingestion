import ast
import hashlib
import re
import uuid
from io import StringIO
from pathlib import Path
from typing import Dict, List
import logging

from docutils import nodes
from docutils.core import publish_doctree

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


def extract_yaml_chunks(source: str, name: str) -> List[Dict]:
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


def extract_dockerfile_chunks(source: str, name: str) -> List[Dict]:
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


def extract_shell_chunks(source: str, name: str) -> List[Dict]:
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


def extract_file_chunk(source: str, name: str) -> List[Dict]:
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


def split_chunk_for_embedding(chunk: Dict, max_chars: int = 28000) -> List[Dict]:
    lines = chunk["code"].splitlines()
    pieces: List[Dict] = []
    current_lines: List[str] = []
    current_length = 0
    current_start = 0

    def flush_piece(stop_index: int) -> None:
        nonlocal current_lines, current_length, current_start
        if not current_lines:
            return

        start_line = chunk["start_line"] + current_start
        end_line = chunk["start_line"] + stop_index - 1
        pieces.append({
            "type": chunk["type"],
            "symbol": chunk["symbol"],
            "start_line": start_line,
            "end_line": end_line,
            "code": "\n".join(current_lines),
        })
        current_lines = []
        current_length = 0
        current_start = stop_index

    for index, line in enumerate(lines, start=0):
        line_text = line + "\n"
        line_length = len(line_text)
        if line_length > max_chars:
            if current_lines:
                flush_piece(index)
            for offset in range(0, len(line_text), max_chars):
                segment = line_text[offset : offset + max_chars]
                pieces.append({
                    "type": chunk["type"],
                    "symbol": chunk["symbol"],
                    "start_line": chunk["start_line"] + index,
                    "end_line": chunk["start_line"] + index,
                    "code": segment.rstrip("\n"),
                })
            current_start = index + 1
            continue

        if current_length + line_length > max_chars and current_lines:
            flush_piece(index)

        current_lines.append(line)
        current_length += line_length

    flush_piece(len(lines))

    if not pieces:
        return [chunk]

    return pieces


def _publish_doctree_safely(source: str, name: str):
    warning_stream = StringIO()
    settings_overrides = {
        "report_level": 5,
        "halt_level": 6,
        "warning_stream": warning_stream,
        "syntax_highlight": "none",
        "strip_comments": True,
        "doctitle_xform": False,
    }

    try:
        return publish_doctree(source, settings_overrides=settings_overrides)
    except Exception as exc:
        logger.warning("Failed to parse RST file %s: %s", name, exc)
        return None


def _max_node_line(node: nodes.Node) -> int | None:
    line_numbers = [getattr(node, "line", None)]
    for child in getattr(node, "children", []):
        child_line = _max_node_line(child)
        if child_line is not None:
            line_numbers.append(child_line)
    return max([num for num in line_numbers if num is not None], default=None)


def extract_rst_chunks(source: str, name: str) -> List[Dict]:
    doctree = _publish_doctree_safely(source, name)
    if doctree is None:
        return extract_file_chunk(source, name)

    sections = [
        node for node in doctree.traverse(nodes.section)
        if isinstance(node.parent, nodes.document)
    ]

    if not sections:
        return extract_file_chunk(source, name)

    chunks: List[Dict] = []
    for idx, section in enumerate(sections, start=1):
        title_node = section.next_node(nodes.title)
        symbol = title_node.astext() if title_node else f"{name}:section{idx}"
        start_line = section.line or 1
        end_line = _max_node_line(section) or start_line

        chunks.append(
            {
                "type": "rst",
                "symbol": symbol,
                "start_line": start_line,
                "end_line": end_line,
                "code": section.astext(),
            }
        )

    return chunks


def is_dockerfile(name: str, suffix: str) -> bool:
    normalized = name.lower()
    return normalized.startswith("dockerfile") or "dockerfile" in normalized or suffix == ""


def extract_chunks(file_path: str) -> List[Dict]:
    """Extract chunks from arbitrary text files.

    - Python files: uses AST-based splitting (functions/classes).
    - YAML: splits on document separator `---`.
    - Dockerfile: splits on `FROM` (multi-stage builds).
    - Shell scripts: attempts to split by function definitions.
    - Fallback: returns a single chunk containing the whole file.
    """
    suffix = Path(file_path).suffix.lower()
    name = Path(file_path).name

    if suffix == ".py":
        return extract_python_chunks(file_path)

    if suffix == ".rst":
        source = read_text_file(file_path)
        return extract_rst_chunks(source, name)

    source = read_text_file(file_path)

    if suffix in (".yml", ".yaml"):
        return extract_yaml_chunks(source, name)

    if is_dockerfile(name, suffix):
        return extract_dockerfile_chunks(source, name)

    if suffix in (".sh", ".bash"):
        return extract_shell_chunks(source, name)

    return extract_file_chunk(source, name)
