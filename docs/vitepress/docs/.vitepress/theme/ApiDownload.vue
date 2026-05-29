<script setup>
import { ref, computed } from 'vue'
import { useRoute } from 'vitepress'
import JSZip from 'jszip'

const route = useRoute()
const loading = ref(false)
const isApiPage = computed(() => route.path.startsWith('/api/'))

function getFilterVersion() {
  try {
    return localStorage.getItem('cann-filter-selection') || 'all'
  } catch {
    return 'all'
  }
}

function filterContentByVersion(content, version) {
  const htmlBlockRegex = /<p>\s*CANNFILTER_DIV_(\w+)_OPEN\s*<\/p>([\s\S]*?)<p>\s*CANNFILTER_DIV_\1_CLOSE\s*<\/p>/g
  const plainBlockRegex = /CANNFILTER_DIV_(\w+)_OPEN[ \t]*\n?([\s\S]*?)CANNFILTER_DIV_\1_CLOSE/g

  if (version === 'all') {
    content = content.replace(htmlBlockRegex, '$2')
    content = content.replace(plainBlockRegex, '$2')
    return content
  }

  content = content.replace(htmlBlockRegex, (match, ver, inner) => ver === version ? inner : '')
  content = content.replace(plainBlockRegex, (match, ver, inner) => ver === version ? inner : '')
  return content
}

function getSidebarLinks() {
  if (typeof document === 'undefined') return []
  const links = document.querySelectorAll('.VPSidebar a[href^="/api/"]')
  const hrefs = [...links].map(a => a.getAttribute('href'))
  return [...new Set(hrefs)]
}

function hrefToSourcePath(href) {
  const relative = href.replace(/^\/api\//, '').replace(/\/$/, '')
  return '/api-source/' + relative + '.md'
}

async function checkExists(url) {
  try {
    const resp = await fetch(url, { method: 'HEAD' })
    return resp.ok
  } catch {
    return false
  }
}

function timestamp() {
  const now = new Date()
  const y = now.getFullYear()
  const m = String(now.getMonth() + 1).padStart(2, '0')
  const d = String(now.getDate()).padStart(2, '0')
  const h = String(now.getHours()).padStart(2, '0')
  const min = String(now.getMinutes()).padStart(2, '0')
  const s = String(now.getSeconds()).padStart(2, '0')
  return y + m + d + '-' + h + min + s
}

function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  document.body.appendChild(a)
  a.click()
  document.body.removeChild(a)
  URL.revokeObjectURL(url)
}

async function handleDownload() {
  const version = getFilterVersion()
  const versionLabel = version === 'all' ? 'all' : ('v' + version)
  const hrefs = getSidebarLinks()

  if (hrefs.length === 0) {
    return
  }

  loading.value = true

  try {
    const sourcePaths = []
    const batchSize = 30

    for (let i = 0; i < hrefs.length; i += batchSize) {
      const batch = hrefs.slice(i, i + batchSize)
      const results = await Promise.all(batch.map(async (href) => {
        const path = hrefToSourcePath(href)
        const ok = await checkExists(path)
        return ok ? path : null
      }))
      sourcePaths.push(...results.filter(Boolean))
    }

    const zip = new JSZip()

    for (let i = 0; i < sourcePaths.length; i += batchSize) {
      const batch = sourcePaths.slice(i, i + batchSize)
      const results = await Promise.all(batch.map(async (sp) => {
        try {
          const resp = await fetch(sp)
          if (!resp.ok) return null
          let content = await resp.text()
          content = filterContentByVersion(content, version)
          const filename = sp.replace(/^\/api-source\//, '')
          return { filename, content }
        } catch {
          return null
        }
      }))
      for (const r of results) {
        if (r) zip.file(r.filename, r.content)
      }
    }

    const blob = await zip.generateAsync({ type: 'blob' })
    const ts = timestamp()
    downloadBlob(blob, 'AscendC-API_' + versionLabel + '_' + ts + '.zip')
  } catch (err) {
    console.error('Download failed:', err)
  } finally {
    loading.value = false
  }
}
</script>

<template>
  <button
    v-if="isApiPage"
    class="api-download-btn"
    :class="{ loading }"
    :disabled="loading"
    :title="loading ? '打包中...' : '下载全部 API Markdown'"
    @click="handleDownload"
  >
    <svg
      v-if="!loading"
      width="20"
      height="20"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      stroke-width="2"
      stroke-linecap="round"
      stroke-linejoin="round"
    >
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
      <polyline points="7 10 12 15 17 10" />
      <line x1="12" y1="15" x2="12" y2="3" />
    </svg>
    <span v-else class="spinner" />
  </button>
</template>

<style scoped>
.api-download-btn {
  position: fixed;
  bottom: 40px;
  right: 96px;
  width: 44px;
  height: 44px;
  border: 1px solid var(--vp-c-divider);
  border-radius: 50%;
  background: var(--vp-c-bg-soft);
  color: var(--vp-c-text-2);
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 99;
  transition: all 0.3s;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}

.api-download-btn:hover:not(:disabled) {
  border-color: var(--vp-c-brand-1);
  color: var(--vp-c-brand-1);
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
}

.api-download-btn:disabled {
  cursor: wait;
  opacity: 0.8;
}

.api-download-btn.loading {
  border-color: var(--vp-c-brand-1);
  color: var(--vp-c-brand-1);
}

.spinner {
  width: 18px;
  height: 18px;
  border: 2px solid var(--vp-c-divider);
  border-top-color: var(--vp-c-brand-1);
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}
</style>
