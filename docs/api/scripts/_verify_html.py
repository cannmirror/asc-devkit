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

"""Post-build HTML validation: check for common rendering issues after Sphinx build.

Checks:
  1. Residual .md links (verify generate_zips.py post-processing)
  2. Broken image references (<img src="..."> pointing to missing files)
  3. Empty <section> elements (may indicate MyST parsing failures)
  4. Backslash escape chars in visible text
"""

import re
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parent.parent

IMG_RE = re.compile(r'<img[^>]+src="([^"]+)"')
RESIDUAL_MD_RE = re.compile(r'href="[^"]+\.md(?:#[^"]*)?"')
SECTION_RE = re.compile(r"<section[^>]*>\s*</section>")
BAD_ESCAPE_RE = re.compile(r">[^<]*\\[_()[\]{}][^<]*<")


def find_build_output() -> Path | None:
    for d in ("_build/html", "_build/dirhtml"):
        p = SRC_ROOT / d
        if p.is_dir():
            return p
    return None


def check_residual_md_links(html_dir: Path) -> int:
    """Return count of files still containing .md links."""
    files_with_md = []
    for f in html_dir.rglob("*.html"):
        text = f.read_text(encoding="utf-8", errors="replace")
        if RESIDUAL_MD_RE.search(text):
            files_with_md.append(str(f.relative_to(html_dir)))
    return len(files_with_md)


def check_broken_images(html_dir: Path) -> int:
    """Check <img src="..."> refer to existing files under html_dir."""
    broken = []
    for f in html_dir.rglob("*.html"):
        text = f.read_text(encoding="utf-8", errors="replace")
        for m in IMG_RE.finditer(text):
            src = m.group(1)
            if src.startswith("http:") or src.startswith("https:") or src.startswith("data:"):
                continue
            img_path = (f.parent / src).resolve()
            if not img_path.exists():
                broken.append((str(f.relative_to(html_dir)), src))
    return len(broken)


def check_empty_sections(html_dir: Path) -> int:
    """Count <section> elements with only whitespace."""
    empty = 0
    for f in html_dir.rglob("*.html"):
        text = f.read_text(encoding="utf-8", errors="replace")
        count = len(SECTION_RE.findall(text))
        if count:
            empty += count
    return empty


def check_backslash_escapes(html_dir: Path) -> int:
    """Check for visible backslash escapes that shouldn't appear."""
    hits = 0
    for f in html_dir.rglob("*.html"):
        text = f.read_text(encoding="utf-8", errors="replace")
        count = len(BAD_ESCAPE_RE.findall(text))
        if count:
            hits += count
    return hits


def main():
    html_dir = find_build_output()
    if not html_dir:
        sys.exit(1)

    issues = 0
    issues += check_residual_md_links(html_dir)
    issues += check_broken_images(html_dir)
    issues += check_empty_sections(html_dir)
    issues += check_backslash_escapes(html_dir)

    sys.exit(0 if issues == 0 else 1)


if __name__ == "__main__":
    main()
