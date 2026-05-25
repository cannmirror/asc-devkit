#!/usr/bin python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------


import pathlib
import re
import cmarkgfm

_CSS = None


def _load_css():
    global _CSS
    if _CSS is not None:
        return _CSS
    candidates = [
        pathlib.Path(__file__).parent.parent.parent / "node_modules" / "github-markdown-css" / "github-markdown.css",
        pathlib.Path(__file__).parent.parent / "node_modules" / "github-markdown-css" / "github-markdown.css",
        pathlib.Path("/home/zjj/md-parser/node_modules/github-markdown-css/github-markdown.css"),
    ]
    for p in candidates:
        if p.exists():
            _CSS = p.read_text(encoding="utf-8")
            return _CSS
    _CSS = ""
    return _CSS

_OPTIONS = cmarkgfm.Options.CMARK_OPT_UNSAFE

_MD_LINK_RE = re.compile(r'(href|src)="((?!https?:|//)[^"]+)\.md"')

_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
{style}
</style>
</head>
<body>
<article class="markdown-body">
{body}
</article>
</body>
</html>
"""


def markdown_to_html(text: str) -> str:
    return cmarkgfm.markdown_to_html(text, options=_OPTIONS)


def gfm_to_html(text: str) -> str:
    return cmarkgfm.github_flavored_markdown_to_html(text, options=_OPTIONS)


def _fix_links(html: str) -> str:
    return _MD_LINK_RE.sub(r'\1="\2.html"', html)


def parse_string(text: str, gfm: bool = True) -> str:
    if gfm:
        return _fix_links(gfm_to_html(text))
    return _fix_links(markdown_to_html(text))


def parse_file(filepath: str, gfm: bool = True) -> str:
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    return parse_string(content, gfm=gfm)


def render_html(body: str) -> str:
    return _TEMPLATE.format(style=_load_css(), body=body)
