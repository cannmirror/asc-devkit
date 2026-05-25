import { defineConfig } from 'vitepress'
import { existsSync, readFileSync } from 'node:fs'
import { join, resolve } from 'node:path'
import { load as cheerioLoad } from 'cheerio'
import sidebars from './sidebar.mjs'

function extractBodyContent(html) {
  const bodyMatch = html.match(/<body[^>]*>([\s\S]*)<\/body>/i)
  if (bodyMatch) return bodyMatch[1]
  return html
}

function normalizeHtml(html) {
  const $ = cheerioLoad(html, {
    xml: {
      xmlMode: false,
      decodeEntities: false,
      withStartIndices: false,
      withEndIndices: false,
    },
  }, false)
  return $('body').length ? $('body').html() : $.html()
}

function htmlAsMdPlugin() {
  const root = resolve(import.meta.dirname, '..')

  return {
    name: 'html-as-md',
    enforce: 'pre',
    resolveId(id) {
      if (!id.endsWith('.md') || id.includes('?')) return null

      const absId = id.startsWith(root)
        ? id
        : id.startsWith('/')
          ? resolve(root, '.' + id)
          : resolve(root, id)

      const htmlPath = absId.replace(/\.md$/, '.html')
      if (!existsSync(htmlPath)) return null

      if (existsSync(absId)) return null
      return htmlPath + '?rawhtml=1'
    },
    load(id) {
      if (id.includes('?rawhtml=1')) {
        const htmlPath = id.split('?')[0]
        this.addWatchFile(htmlPath)
        const content = readFileSync(htmlPath, 'utf-8')
        return '\n<!-- RAW_HTML -->\n' + normalizeHtml(extractBodyContent(content)) + '\n'
      }

      if (!id.endsWith('.md') || id.includes('?')) return null
      const htmlPath = id.replace(/\.md$/, '.html')
      if (!existsSync(htmlPath)) return null

      this.addWatchFile(htmlPath)
      const content = readFileSync(htmlPath, 'utf-8')
      return '\n<!-- RAW_HTML -->\n' + normalizeHtml(extractBodyContent(content)) + '\n'
    },
  }
}

function placeholderPlugin() {
  const root = resolve(import.meta.dirname, '..')
  const placeholder = resolve(root, 'api', 'figures', 'placeholder.png')
  return {
    name: 'placeholder-images',
    resolveId(id, importer) {
      if (!importer || !id.match(/\.(png|jpg|jpeg|gif|svg|webp)$/i)) return
      let importerDir = resolve(importer, '..')
      if (importerDir.includes('?')) {
        importerDir = importerDir.split('?')[0]
      }
      const resolved = resolve(importerDir, id)
      if (!existsSync(resolved)) {
        return placeholder
      }
      return resolved
    },
  }
}

function fixCannFilterTags(src) {
  if (!/<(cann-filter|term|ph|__gm__|__ubuf__)\b/i.test(src)) return src
  return src
    .replace(/<\/?cann-filter\b[^>]*>/gi, '')
    .replace(/<(term|ph|__gm__|__ubuf__)\b[^>]*>([\s\S]*?)<\/\1>/gis, '$2')
    .replace(/<(term|ph|__gm__|__ubuf__)\b[^>]*>/gi, '')
    .replace(/<\/(term|ph|__gm__|__ubuf__)>/gi, '')
}

function stripInnerTags(text) {
  return text
    .replace(/<a\s+name\s*=\s*"[^"]*"\s*><\/a>/gi, '')
    .replace(/<\/?span\b[^>]*>/gi, '')
    .replace(/<\/?p\b[^>]*>/gi, '')
    .replace(/<a\s+href\s*=\s*"([^"]*)"\s*>/gi, '<a href="$1">')
    .trim()
}

function convertHtmlTablesToMarkdown(src) {
  let result = src
  result = result.replace(
    /<table\b[\s\S]*?<\/table>/gi,
    (table) => {
      const theadMatch = table.match(/<thead\b[\s\S]*?<\/thead>/i)
      const tbodyMatches = []
      const tbodyRegex = /<tbody\b[\s\S]*?<\/tbody>/gi
      let m
      while ((m = tbodyRegex.exec(table)) !== null) {
        tbodyMatches.push(m[0])
      }

      const rows = []
      if (theadMatch) {
        const thsMatch = theadMatch[0].match(/<th\b[\s\S]*?<\/th>/gi)
        if (thsMatch) {
          rows.push(thsMatch.map(th => stripInnerTags(th.replace(/<th\b[^>]*>/i, '').replace(/<\/th>/i, ''))))
        }
      }

      for (const tbody of tbodyMatches) {
        const trsMatch = tbody.match(/<tr\b[\s\S]*?<\/tr>/gi)
        if (trsMatch) {
          for (const tr of trsMatch) {
            const tdsMatch = tr.match(/<td\b[\s\S]*?<\/td>/gi)
            if (tdsMatch) {
              const cells = tdsMatch.map(td => {
                let text = td.replace(/<td\b[^>]*>/i, '').replace(/<\/td>/i, '')
                text = stripInnerTags(text)
                const linkMatch = text.match(/<a href="([^"]*)">([^<]*)<\/a>/i)
                if (linkMatch) {
                  text = text.replace(linkMatch[0], '[' + linkMatch[2] + '](' + linkMatch[1] + ')')
                }
                return text
              })
              rows.push(cells)
            }
          }
        }
      }

      if (rows.length === 0) return table
      const colCount = Math.max(...rows.map(r => r.length))
      const aligned = rows.map(r => {
        while (r.length < colCount) r.push('')
        return r
      })

      const mdRows = aligned.map((r, i) => '| ' + r.join(' | ') + ' |')
      const sep = '| ' + Array(colCount).fill('---').join(' | ') + ' |'
      return '\n' + mdRows[0] + '\n' + sep + '\n' + mdRows.slice(1).join('\n') + '\n'
    }
  )

  result = result.replace(/<a\s+name\s*=\s*"[^"]*"\s*><\/a>/gi, '')
  result = result.replace(/<br\s*\/?>/gi, '\n')
  return result
}

function fixIndentedBlockquotes(src) {
  return src.replace(/^[ \t]{4,}(?=>)/gm, '')
}

function fixSpanCrossingTableCells(src) {
  let prev
  do {
    prev = src
    src = src.replace(
      /(<span[^>]*>)([\s\S]*?)\|([\s\S]*?)<\/span>/g,
      (match, open, beforePipe, afterPipe) =>
        open + beforePipe + '</span>|' + afterPipe
    )
  } while (src !== prev)
  return src
}

function escapeVueInterpolations(html) {
  return html.replace(
    /(<code\b[\s\S]*?<\/code>|<pre\b[\s\S]*?<\/pre>)|(\{\{[\s\S]*?\}\})/g,
    (match, codeBlock, templateExpr) => {
      if (codeBlock) return codeBlock
      return '<span v-pre>' + templateExpr + '</span>'
    }
  )
}

function loadHeaderCache(filePath) {
  const cachePath = filePath + '.header'
  if (!existsSync(cachePath)) return null
  try {
    return JSON.parse(readFileSync(cachePath, 'utf-8'))
  } catch {
    return null
  }
}

export default defineConfig({
  mpa: false,
  title: 'Ascend C',
  description: 'Ascend C 算子开发文档',
  lang: 'zh-CN',
  ignoreDeadLinks: true,
  lastUpdated: false,

  markdown: {
    html: false,
    config(md) {
      const originalRender = md.render.bind(md)
      md.render = function (src, env) {
        if (src.includes('<!-- RAW_HTML -->')) {
          const htmlContent = src.replace(/^.*<!-- RAW_HTML -->\s*/s, '').replace(/\s*$/, '')
          let html = escapeVueInterpolations(htmlContent)
          return `<div v-pre>\n${html}\n</div>`
        }

        src = fixCannFilterTags(src)

        if (src.startsWith('    >') || src.includes('\n    >')) {
          src = fixIndentedBlockquotes(src)
        }

        if (src.includes('<span') && src.includes('|')) {
          src = fixSpanCrossingTableCells(src)
        }

        if (src.includes('<table')) {
          src = convertHtmlTablesToMarkdown(src)
        }

        let html = originalRender(src, env)

        if (html.includes('CANNFILTER_DIV_')) {
          html = html.replace(/<p>\s*CANNFILTER_DIV_(\w+)_OPEN\s*<\/p>/g, '<div data-filter="$1">')
          html = html.replace(/<p>\s*CANNFILTER_DIV_(\w+)_CLOSE\s*<\/p>/g, '</div>')
        }

        if (html.includes('{{')) {
          html = escapeVueInterpolations(html)
        }

        return `<div v-pre>${html}</div>`
      }
    },
  },

  transformPageData(pageData) {
    const cache = loadHeaderCache(pageData.filePath)
    if (cache && cache.length > 0) {
      pageData.headers = cache
    }
  },

  vite: {
    resolve: {
      preserveSymlinks: true,
    },
    plugins: [htmlAsMdPlugin(), placeholderPlugin()],
  },

  vue: {
    template: {
      compilerOptions: {
        onError: (error) => {
          if (error.code === 2) return
          throw error
        },
      },
    },
  },

  themeConfig: {
    nav: [
      { text: '首页', link: '/' },
      { text: 'AscendC算子开发指南', link: '/guide/入门教程/Ascend-C概述与学习路径' },
      { text: 'Ascend C API', link: '/api/Ascend-C-API列表' },
    ],

    sidebar: sidebars,

    search: {
      provider: 'local',
      options: {
        _render(src, env, md) {
          if (!src || !src.trim()) {
            const htmlPath = env.path.replace(/\.md$/, '.html')
            if (existsSync(htmlPath)) {
              const raw = readFileSync(htmlPath, 'utf-8')
              const seen = {}
              const headingRegex = /<h([23456])([^>]*)>(.*?)<\/h\1>/gi
              return raw.replace(headingRegex, (_, level, attrs, text) => {
                const clean = text.replace(/<[^>]+>/g, '').trim()
                let id = clean.replace(/\s+/g, '-').toLowerCase()
                if (seen[id]) id = `${id}-${seen[id]++}`
                else seen[id] = 1
                const inner = `<a class="header-anchor" href="#${id}">\u00b6</a>`
                const existing = text.match(/<a[^>]*>.*?<\/a>/)
                if (existing) return `<h${level}${attrs}>${text}${inner}</h${level}>`
                return `<h${level}${attrs}>${clean}${inner}</h${level}>`
              })
            }
            return ''
          }
          const html = md.render(src, env)
          if (!src.includes('<!-- RAW_HTML -->')) return html
          const seen = {}
          const headingRegex = /<h([23456])([^>]*)>(.*?)<\/h\1>/gi
          return html.replace(headingRegex, (_, level, attrs, text) => {
            if (text.includes('header-anchor')) return `<h${level}${attrs}>${text}</h${level}>`
            const clean = text.replace(/<[^>]+>/g, '').trim()
            let id = clean.replace(/\s+/g, '-').toLowerCase()
            if (seen[id]) id = `${id}-${seen[id]++}`
            else seen[id] = 1
            const existing = text.match(/<a[^>]*>.*?<\/a>/)
            if (existing) return `<h${level}${attrs}>${text}<a class="header-anchor" href="#${id}">\u00b6</a></h${level}>`
            return `<h${level}${attrs}>${clean}<a class="header-anchor" href="#${id}">\u00b6</a></h${level}>`
          })
        },
        translations: {
          button: { buttonText: '搜索文档' },
          modal: { noResultsText: '未找到结果', footer: { selectText: '选择', navigateText: '切换' } },
        },
      },
    },

    outline: {
      level: [2, 3],
      label: '本页内容',
    },

    docFooter: {
      prev: '上一页',
      next: '下一页',
    },

    footer: {
      copyright: 'Copyright Huawei Technologies Co., Ltd.',
    },
  },
})
