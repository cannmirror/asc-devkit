# Ascend C API 参考 —— 静态网页自动生成指南

## 目录结构

```
docs/api/
  index.rst                        首页入口

  scripts/                         构建脚本 & 配置
    conf.py                         Sphinx 配置（主题、扩展、构建钩子）
    _toc.yml                        _gen_toc.py 直接生成（Sphinx 侧边栏数据源）
    _gen_toc.py                     从 README.md 生成 TOC 文件（×3 输出）
    _verify_toc.py                  验证 _toc.yml 引用的源文件是否存在
    _verify_html.py                 构建后校验（残留 .md 链接、空段、转义符、图片引用）
    generate_zips.py                构建后处理：修复 .md 链接 + 生成下载 zip
    build.sh                        Linux 一键构建脚本
    AscendC_Api_Auto_Gen_Html.md    本指南

  _static/
    css/custom.css                 自定义样式
    js/toc-data.js                 内嵌 TOC 数据（JS 变量，避免 fetch，支持 file://）
    js/toc-toggle.js               侧边栏 SPA 导航（分区拦截、AJAX 加载、动态重建目录）
    js/version-filter.js           版本筛选（全量 / 950 切换，侧边栏重建后重新筛选）
    toc_data.json                  TOC 数据 JSON（HTTP 下 fetch 备用，file:// 不依赖）
    switcher.json                  版本切换器菜单
    version_filter.json            每页面是否支持 950 版本的映射表

  _templates/
    layout.html                    基础布局（注入变量 + cache-busting + 下载按钮）
    download-button.html           浮动下载按钮
    version-filter.html            版本筛选下拉组件

  _build/html/                     构建输出（每个页面一个 .html 文件）
```

## 前置依赖

| 包 | 用途 |
|---|---|
| `sphinx` | 文档构建引擎 |
| `myst-parser` | 解析 Markdown → Sphinx AST |
| `sphinx-external-toc` | 读取 `_toc.yml` 生成侧边栏目录 |
| `pydata-sphinx-theme` | 页面主题（导航栏、侧边栏、搜索、深浅色切换） |
| `pyyaml` | `_gen_toc.py` 写 YAML 时使用 |

安装：

```bash
pip install sphinx myst-parser sphinx-external-toc pydata-sphinx-theme pyyaml
```

---

## 一键部署脚本

### Linux / macOS（Bash）

```bash
# 已内置在 scripts/build.sh 中，直接运行
bash scripts/build.sh
```

> 一键脚本已简化为 3 步：`generate_zips.py` 和 `_verify_html.py` 在 `conf.py` 的 `on_build_finished` 钩子中自动运行。

---


## 构建产物

构建完成后，入口文件为：

```
_build/html/index.html
```

整个 `_build/html/` 目录为完整静态网站，可直接部署到任意静态服务器（Nginx、Apache、GitHub Pages 等）。
