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

import json
import subprocess
import sys
import hashlib
import time
from pathlib import Path

project = 'Ascend C API 参考'

extensions = [
    'myst_parser',
    'sphinx_external_toc',
]

myst_enable_extensions = [
    'colon_fence',
    'deflist',
    'dollarmath',
    'fieldlist',
    'html_image',
    'replacements',
    'smartquotes',
    'strikethrough',
    'substitution',
    'tasklist',
]

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

master_doc = 'index'
exclude_patterns = ['_build', 'README.md.orig', 'figures', '*.py', 'context', '.git', 'scripts']

_project_root = Path(__file__).resolve().parent.parent
templates_path = [str(_project_root / '_templates')]
html_theme = 'pydata_sphinx_theme'
html_static_path = [str(_project_root / '_static')]
external_toc_path = 'scripts/_toc.yml'

version = '950'
release = version

html_theme_options = {
    'navigation_depth': 6,
    'collapse_navigation': True,
    'show_nav_level': 2,
    'navbar_align': 'left',
    'navbar_start': ['navbar-logo'],
    'navbar_end': ['version-filter', 'navbar-icon-links'],
    'article_header_start': ['breadcrumbs'],
    'footer_start': ['copyright'],
    'footer_end': [],
    'show_prev_next': False,
    'back_to_top_button': False,
    'use_edit_page_button': False,
    'icon_links': [],
    'logo': {
        'text': 'Ascend C API参考',
    },
    'switcher': {
        'json_url': '_static/switcher.json',
        'version_match': '全量',
    },
    'check_switcher': False,
}

html_title = 'Ascend C API参考'

html_context = {
    'current_version': '全量',
    'versions': [
        ('全量', '全量 (当前版本)'),
        ('950', '950'),
    ],
    'build_hash': hashlib.md5(str(time.time()).encode()).hexdigest()[:8],
}

html_css_files = []
html_js_files = []

with open(_project_root / '_static' / 'version_filter.json', 'r', encoding='utf-8') as f:
    _filter_data = json.load(f)
_hidden_keys = [k for k, v in _filter_data.items() if not v]

html_context['version_filter_hidden'] = json.dumps(_hidden_keys, ensure_ascii=False)

external_toc_exclude_missing = True


def on_build_finished(app, exception):
    if exception is not None:
        return
    _scripts_dir = _project_root / 'scripts'
    _post_scripts = ['generate_zips.py', '_verify_html.py']
    for _script in _post_scripts:
        subprocess.run(
            [sys.executable, str(_scripts_dir / _script)],
            capture_output=True, cwd=str(_project_root),
        )


def setup(app):
    app.connect('build-finished', on_build_finished)
