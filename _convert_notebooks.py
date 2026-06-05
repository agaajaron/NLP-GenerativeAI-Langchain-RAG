"""
Notebook → .py refactoring script.
Converts all .ipynb files in the repo into folders of modular .py files.
"""

import json
import re
import os
import py_compile
import pathlib
import shutil
import tempfile
from textwrap import dedent

ROOT = pathlib.Path("/home/user/NLP-GenerativeAI-Langchain-RAG")
ARCHIVE_DIR = ROOT / "notebooks" / "originals"


# ── helpers ──────────────────────────────────────────────────────────────────

def slugify(text: str) -> str:
    text = re.sub(r"[*_`#\[\]()]", "", text)          # strip markdown markup
    text = re.sub(r"[^\w\s-]", "", text, flags=re.ASCII)
    text = re.sub(r"[\s\-]+", "_", text.strip())
    text = text.strip("_").lower()
    return text[:60] or "section"


def convert_magic(line: str) -> str:
    """Convert Jupyter magic/shell commands to plain Python."""
    stripped = line.rstrip()

    # %matplotlib inline
    if re.match(r"^\s*%matplotlib\s+inline", stripped):
        return re.sub(r"%matplotlib\s+inline", "import matplotlib; matplotlib.use('Agg')", stripped)

    # %load_ext autoreload / %autoreload
    if re.match(r"^\s*%load_ext\s+autoreload", stripped):
        return re.sub(r"%load_ext\s+autoreload", "# autoreload enabled", stripped)
    if re.match(r"^\s*%autoreload", stripped):
        return re.sub(r"%autoreload.*", "# autoreload enabled", stripped)

    # !pip install xyz  →  # requires: xyz
    m = re.match(r"^(\s*)!pip(?:3)?\s+install\s+(.+)", stripped)
    if m:
        pkgs = m.group(2).strip()
        return f"{m.group(1)}# requires: {pkgs}"

    # !pip3 show
    if re.match(r"^\s*!pip3?\s+show", stripped):
        return re.sub(r"!pip3?\s+show\s+(\S+)", r"# requires: \1", stripped)

    # %timeit <expr>
    m = re.match(r"^(\s*)%timeit\s+(.+)", stripped)
    if m:
        indent = m.group(1)
        expr = m.group(2).strip()
        return (
            f"{indent}import time as _time; _t0 = _time.time()\n"
            f"{indent}{expr}\n"
            f"{indent}print(f'elapsed: {{_time.time()-_t0:.4f}}s')"
        )

    # any other ! shell command  →  comment
    if re.match(r"^\s*!", stripped):
        return re.sub(r"^\s*!", "# shell: ", stripped)

    # any other % magic  →  comment
    if re.match(r"^\s*%", stripped):
        return re.sub(r"^\s*%", "# magic: %", stripped)

    return line


def convert_code_cell(source_lines: list[str]) -> list[str]:
    """Convert a list of source lines from a code cell."""
    out = []
    in_pip_continuation = False
    for line in source_lines:
        if in_pip_continuation:
            # continuation line of a multi-line !pip install — comment it
            stripped = line.rstrip()
            out.append(f"# requires (cont): {stripped.strip().rstrip(chr(92))}")
            in_pip_continuation = stripped.endswith("\\")
            continue
        converted = convert_magic(line)
        # if this converted line is a # requires: ... and ends with \, flag continuation
        if re.match(r"^\s*# requires:", converted) and converted.rstrip().endswith("\\"):
            converted = converted.rstrip().rstrip("\\").rstrip()
            in_pip_continuation = True
        out.append(converted)
    return out


def markdown_to_comment_block(source: str) -> str:
    """Turn a markdown cell into a Python block comment."""
    lines = source.split("\n")
    result = ["# ---"]
    for line in lines:
        result.append(f"# {line}" if line else "#")
    result.append("# ---")
    return "\n".join(result)


def infer_description(cells) -> str:
    """Return a short description from the first markdown cell body."""
    for c in cells:
        if c["cell_type"] == "markdown":
            src = "".join(c.get("source", []))
            # skip pure heading lines
            lines = [l for l in src.split("\n")
                     if l.strip() and not re.match(r"^#+\s", l.strip())]
            if lines:
                desc = lines[0].strip()
                # strip markdown bold/italic
                desc = re.sub(r"\*+([^*]+)\*+", r"\1", desc)
                return desc[:120]
    return "Converted from Jupyter notebook."


def collect_pip_requirements(cells) -> list[str]:
    """Extract package names from !pip install and import lines."""
    pkgs = set()
    for c in cells:
        src = "".join(c.get("source", []))
        # !pip install ...
        for m in re.finditer(r"!pip(?:3)?\s+install\s+([^\n;]+)", src):
            for pkg in m.group(1).split():
                if not pkg.startswith("-"):
                    pkgs.add(pkg.split("==")[0].split(">=")[0].split("<=")[0])
        # import xyz / from xyz
        for m in re.finditer(r"^(?:import|from)\s+([a-zA-Z_][a-zA-Z0-9_]*)", src, re.M):
            pkgs.add(m.group(1))
    # keep only non-stdlib (rough filter)
    stdlib = {
        "os", "sys", "re", "json", "pathlib", "math", "time", "datetime",
        "collections", "itertools", "functools", "typing", "textwrap", "io",
        "abc", "copy", "random", "string", "struct", "hashlib", "base64",
        "urllib", "http", "csv", "sqlite3", "logging", "warnings", "traceback",
        "inspect", "types", "enum", "dataclasses", "contextlib", "threading",
        "multiprocessing", "subprocess", "shutil", "tempfile", "glob",
        "__future__", "builtins",
    }
    return sorted(pkgs - stdlib)


# ── splitting logic ───────────────────────────────────────────────────────────

def is_section_heading(cell) -> bool:
    if cell["cell_type"] != "markdown":
        return False
    src = "".join(cell.get("source", []))
    return bool(re.match(r"^\s*#{1,3}\s+\S", src.strip()))


def get_heading_text(cell) -> str:
    src = "".join(cell.get("source", []))
    first = src.strip().split("\n")[0]
    return re.sub(r"^#+\s*", "", first).strip()


def split_into_sections(cells) -> list[dict]:
    """
    Split cells into sections using # / ## / ### headings as boundaries.
    Returns list of {heading, heading_level, cells}.
    Falls back to 15-code-cell chunks if no headings found.
    """
    has_headings = any(is_section_heading(c) for c in cells)

    sections = []

    if has_headings:
        current = None
        for cell in cells:
            if is_section_heading(cell):
                if current is not None:
                    sections.append(current)
                src = "".join(cell.get("source", []))
                first_line = src.strip().split("\n")[0]
                level = len(re.match(r"^(#+)", first_line).group(1))
                current = {
                    "heading": get_heading_text(cell),
                    "heading_level": level,
                    "cells": [cell],
                }
            else:
                if current is None:
                    current = {"heading": "preamble", "heading_level": 0, "cells": []}
                current["cells"].append(cell)
        if current:
            sections.append(current)
    else:
        # chunk every 15 code cells
        chunk_size = 15
        code_cells = [c for c in cells if c["cell_type"] == "code"]
        for i in range(0, max(1, len(code_cells)), chunk_size):
            sections.append({
                "heading": f"section_{i // chunk_size + 1}",
                "heading_level": 0,
                "cells": code_cells[i : i + chunk_size],
            })

    return sections


# ── file generation ───────────────────────────────────────────────────────────

def build_py_content(section: dict, notebook_name: str, index: int) -> str | None:
    """
    Build the content of a .py file for one section.
    Returns None if the section produces no executable code.
    """
    heading = section["heading"]
    cells = section["cells"]

    lines = []

    # module docstring
    lines.append(f'"""')
    lines.append(f"Notebook : {notebook_name}")
    lines.append(f"Section  : {heading}")
    lines.append(f"Index    : {index:02d}")
    lines.append(f'"""')
    lines.append("")

    has_code = False
    for cell in cells:
        src = "".join(cell.get("source", []))
        if not src.strip():
            continue

        if cell["cell_type"] == "markdown":
            lines.append(markdown_to_comment_block(src))
            lines.append("")
        elif cell["cell_type"] == "code":
            code_lines = src.split("\n")
            converted = convert_code_cell(code_lines)
            # check if there's any real code (not just comments/empty)
            real_code = [
                l for l in converted
                if l.strip() and not l.strip().startswith("#")
            ]
            if real_code:
                has_code = True
            lines.extend(converted)
            lines.append("")

    if not has_code:
        return None

    return "\n".join(lines)


def make_unique_filename(name: str, used: set[str]) -> str:
    if name not in used:
        used.add(name)
        return name
    suffix = ord("a")
    candidate = f"{name}_{chr(suffix)}"
    while candidate in used:
        suffix += 1
        candidate = f"{name}_{chr(suffix)}"
    used.add(candidate)
    return candidate


# ── per-notebook processing ───────────────────────────────────────────────────

def process_notebook(nb_path: pathlib.Path) -> dict:
    """Process a single notebook. Returns a summary dict."""
    summary = {
        "notebook": nb_path.name,
        "py_files": [],
        "warnings": [],
        "skipped_sections": [],
    }

    with open(nb_path) as f:
        nb = json.load(f)

    cells = nb.get("cells", [])
    nb_stem = nb_path.stem
    folder = nb_path.parent / nb_stem

    # ── archive original ──────────────────────────────────────────────────────
    archive_dest = ARCHIVE_DIR / nb_path.relative_to(ROOT)
    archive_dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(nb_path, archive_dest)

    # ── create output folder ──────────────────────────────────────────────────
    folder.mkdir(parents=True, exist_ok=True)

    # ── split into sections ───────────────────────────────────────────────────
    sections = split_into_sections(cells)

    used_names: set[str] = set()
    py_file_records = []  # [(path, section_heading)]

    for idx, section in enumerate(sections, start=1):
        heading = section["heading"]
        slug = slugify(heading)
        base_name = f"{idx:02d}_{slug}"
        unique_name = make_unique_filename(base_name, used_names)
        py_filename = unique_name + ".py"
        py_path = folder / py_filename

        content = build_py_content(section, nb_stem, idx)

        if content is None:
            summary["skipped_sections"].append(f"  section {idx:02d} '{heading}' — no code, skipped")
            continue

        py_path.write_text(content, encoding="utf-8")

        # ── py_compile check ──────────────────────────────────────────────────
        try:
            py_compile.compile(str(py_path), doraise=True)
        except py_compile.PyCompileError as e:
            # Attempt simple fix: wrap offending lines as comments
            msg = str(e)
            summary["warnings"].append(
                f"  {py_filename}: syntax error — {msg[:120]}"
            )
            # Mark the file as having an error (don't delete it)
            header = f"# SYNTAX ERROR during conversion — review manually\n# {msg[:200]}\n\n"
            py_path.write_text(header + content, encoding="utf-8")

        line_count = len(content.split("\n"))
        py_file_records.append((py_filename, heading, line_count))
        summary["py_files"].append(py_filename)

    # ── add __main__ guard to last .py ────────────────────────────────────────
    if py_file_records:
        last_py = folder / py_file_records[-1][0]
        existing = last_py.read_text(encoding="utf-8")
        guard = '\n\nif __name__ == "__main__":\n    pass  # entry point\n'
        if '__name__ == "__main__"' not in existing:
            last_py.write_text(existing + guard, encoding="utf-8")

    # ── generate folder README.md ─────────────────────────────────────────────
    description = infer_description(cells)
    pip_reqs = collect_pip_requirements(cells)

    readme_lines = [
        f"# {nb_stem}",
        "",
        description,
        "",
        "## Files",
        "",
        "| File | Section | Lines |",
        "|------|---------|-------|",
    ]
    for fname, heading, lcount in py_file_records:
        readme_lines.append(f"| `{fname}` | {heading} | {lcount} |")

    if pip_reqs:
        readme_lines += [
            "",
            "## Requirements",
            "",
            "```",
        ]
        readme_lines += pip_reqs
        readme_lines.append("```")

    (folder / "README.md").write_text("\n".join(readme_lines) + "\n", encoding="utf-8")

    return summary


# ── root README update ────────────────────────────────────────────────────────

def update_root_readme(summaries: list[dict]) -> None:
    root_readme = ROOT / "README.md"

    section_lines = [
        "",
        "## Notebook → Module Map",
        "",
        "| Notebook | Folder | .py Files |",
        "|----------|--------|-----------|",
    ]
    for s in summaries:
        nb = s["notebook"]
        stem = pathlib.Path(nb).stem
        folder_link = f"[{stem}/]({stem}/README.md)"
        count = len(s["py_files"])
        section_lines.append(f"| `{nb}` | {folder_link} | {count} |")

    new_block = "\n".join(section_lines) + "\n"

    if root_readme.exists():
        existing = root_readme.read_text(encoding="utf-8")
        marker = "\n## Notebook → Module Map"
        if marker in existing:
            existing = existing[: existing.index(marker)]
        root_readme.write_text(existing + new_block, encoding="utf-8")
    else:
        header = "# NLP-GenerativeAI-Langchain-RAG\n\nRefactored notebook repository.\n"
        root_readme.write_text(header + new_block, encoding="utf-8")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    notebooks = sorted(
        p for p in ROOT.glob("*.ipynb")
        if ".ipynb_checkpoints" not in str(p)
    )

    print(f"Found {len(notebooks)} notebooks:\n")
    for nb in notebooks:
        print(f"  {nb.name}")
    print()

    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    all_summaries = []

    for nb_path in notebooks:
        print(f"Processing: {nb_path.name} ...")
        summary = process_notebook(nb_path)
        all_summaries.append(summary)

        print(f"  → {len(summary['py_files'])} .py files created")
        for warn in summary["warnings"]:
            print(f"  WARNING: {warn}")
        for skip in summary["skipped_sections"]:
            print(f"  SKIP: {skip}")
        print()

    update_root_readme(all_summaries)

    # ── final file tree ───────────────────────────────────────────────────────
    print("\n=== NEW FILE TREE ===\n")
    for p in sorted(ROOT.rglob("*")):
        if ".git" in p.parts or ".ipynb_checkpoints" in p.parts:
            continue
        # skip the script itself
        if p.name == "_convert_notebooks.py":
            continue
        rel = p.relative_to(ROOT)
        depth = len(rel.parts) - 1
        prefix = "  " * depth + ("├── " if depth else "")
        print(f"{prefix}{p.name}{'/' if p.is_dir() else ''}")

    # ── totals ────────────────────────────────────────────────────────────────
    total_py = sum(len(s["py_files"]) for s in all_summaries)
    total_warn = sum(len(s["warnings"]) for s in all_summaries)
    print(f"\n=== SUMMARY ===")
    print(f"  Notebooks processed : {len(notebooks)}")
    print(f"  Folders created     : {len(notebooks)}")
    print(f"  .py files created   : {total_py}")
    print(f"  Warnings            : {total_warn}")

    if total_warn:
        print("\nNotebooks with warnings:")
        for s in all_summaries:
            if s["warnings"]:
                print(f"  {s['notebook']}")
                for w in s["warnings"]:
                    print(f"    {w}")


if __name__ == "__main__":
    main()
