#!/usr/bin/python3
# coding=utf-8

# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Post-build processing: fix .md links and generate download zips.

1. Fix .md → .html links in built HTML files.
   Raw HTML <a href="...md"> tags inside .md source files are preserved
   as-is by myst-parser, causing broken navigation in the browser.
2. Generate filtered download zips:
     _build/html/download_all.zip   — all source .md/.rst files (original, unfiltered)
     _build/html/download_950.zip   — only 950-supported .md/.rst files,
                                      with product support tables stripped to 950-only rows
"""

import json
import re
import urllib.parse
import zipfile
from pathlib import Path


# ── .md → .html link fix ──────────────────────────────────────────────

HREF_MD_RE = re.compile(r'href="([^"]+)\.md(#(?:.*?))?"', re.IGNORECASE)


def fix_md_links(html_dir: Path) -> tuple[int, int]:
    """Replace href="*.md" with href="*.html" in all HTML files.

    Returns (files_changed, total_replacements).
    """
    if not html_dir.is_dir():
        return 0, 0

    total_fixed = 0
    total_files = 0
    for html_file in html_dir.rglob('*.html'):
        content = html_file.read_text(encoding='utf-8')
        new_content, count = HREF_MD_RE.subn(r'href="\1.html\2"', content)
        if count:
            html_file.write_text(new_content, encoding='utf-8')
            total_files += 1
            total_fixed += count

    return total_files, total_fixed


EXCLUDE_DIRS = {'context', 'figures', '_build', '_static', '_templates', '.git', '__pycache__'}


# ── 950 content filter ────────────────────────────────────────────────

def filter_md_for_950(content: str) -> str:
    """Remove non-950 product rows from all '产品支持情况' tables in *content*.

    For each ``## 产品支持情况`` section, locate the ``<tbody>`` and keep
    only the ``<tr>`` whose first ``<td>`` contains *both* "950PR" and "950DT".
    The ``<thead>`` header row is always preserved.

    If no 950 row is found, the section is left untouched.
    """

    def _replace_section(match: re.Match) -> str:
        section = match.group(0)

        tbody_m = re.search(
            r"<tbody[^>]*>(.*?)</tbody>", section, re.DOTALL | re.IGNORECASE
        )
        if not tbody_m:
            return section

        tbody_content = tbody_m.group(1)
        rows = re.findall(
            r"<tr\b.*?</tr>", tbody_content, re.DOTALL | re.IGNORECASE
        )

        kept: list[str] = []
        for row in rows:
            td_m = re.search(
                r"<td[^>]*>(.*?)</td>", row, re.DOTALL | re.IGNORECASE
            )
            if td_m:
                td_text = re.sub(r"<[^>]+>", "", td_m.group(1)).strip()
                if "950PR" in td_text and "950DT" in td_text:
                    kept.append(row)
                    break

        if not kept:
            return section

        prefix = section[: tbody_m.start()]
        suffix = section[tbody_m.end():]
        return prefix + "<tbody>" + "".join(kept) + "</tbody>" + suffix

    return re.sub(
        r"## 产品支持情况.*?(?=\n## |\Z)",
        _replace_section,
        content,
        flags=re.DOTALL,
    )


# ── source-file mapping ───────────────────────────────────────────────

def _find_source_file(rel_path: str, src_root: Path) -> Path | None:
    for ext in (".md", ".rst"):
        candidate = src_root / (rel_path + ext)
        if candidate.exists():
            return candidate
    return None


def load_950_supported_sources(json_path: Path, src_root: Path) -> set[Path]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    supported: set[Path] = set()
    for key, is_supported in data.items():
        if not is_supported:
            continue
        html_name = urllib.parse.unquote(key)
        rel_path = html_name.rsplit(".", 1)[0]
        src_file = _find_source_file(rel_path, src_root)
        if src_file:
            supported.add(src_file)
    return supported


def collect_all_sources(src_root: Path) -> list[Path]:
    """Collect all .md/.rst source files, excluding build/config directories."""
    sources = []
    for ext in ('.md', '.rst'):
        for f in sorted(src_root.rglob(f'*{ext}')):
            if any(p in EXCLUDE_DIRS for p in f.parts):
                continue
            sources.append(f)
    return sources


# ── zip builder ───────────────────────────────────────────────────────

def build_src_zip(
    out_path: Path,
    source_files: list[Path],
    src_root: Path,
    filter_fn=None,
) -> None:
    """Create a zip archive of source .md/.rst files.

    Args:
        out_path: Destination zip file path.
        source_files: List of source file paths to include.
        src_root: Root directory for computing relative arcnames.
        filter_fn: Optional callable(str) -> str to transform .md content before zipping.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for src_file in sorted(source_files):
            arcname = str(src_file.relative_to(src_root))
            if filter_fn and src_file.suffix == ".md":
                content = src_file.read_text(encoding="utf-8")
                filtered = filter_fn(content)
                zf.writestr(arcname, filtered)
            else:
                zf.write(src_file, arcname)


# ── main ──────────────────────────────────────────────────────────────

def main():
    src_root = Path(__file__).resolve().parent.parent

    html_dir = src_root / "_build" / "dirhtml"
    if not html_dir.is_dir():
        html_dir = src_root / "_build" / "html"

    json_path = src_root / "_static" / "version_filter.json"

    if not html_dir.is_dir():
        return

    files, fixed = fix_md_links(html_dir)

    all_sources = collect_all_sources(src_root)

    # ── 全量 zip: original files, no filtering ──
    all_zip = html_dir / "download_all.zip"
    build_src_zip(all_zip, all_sources, src_root)

    # ── 950 zip: 950-only files + stripped product tables ──
    supported = load_950_supported_sources(json_path, src_root)

    zip_950 = html_dir / "download_950.zip"
    supported_list = sorted(supported)
    build_src_zip(zip_950, supported_list, src_root, filter_fn=filter_md_for_950)


if __name__ == "__main__":
    main()
