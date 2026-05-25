<script setup>
import { ref, onMounted, onUnmounted } from 'vue'

const searchInput = ref(null)
const query = ref('')
const results = ref([])
const loading = ref(false)
const showResults = ref(false)
const selectedIndex = ref(-1)

function initPagefind() {
  try {
    if (window.__pagefind__) return
    const script = document.createElement('script')
    script.src = '/pagefind/pagefind.js'
    script.onload = () => {
      if (window.Pagefind) {
        window.Pagefind.init()
      }
    }
    document.head.appendChild(script)
    window.__pagefind__ = true
  } catch {}
}

function debounce(fn, delay) {
  let timer
  return (...args) => {
    clearTimeout(timer)
    timer = setTimeout(() => fn(...args), delay)
  }
}

async function doSearch(q) {
  if (!window.Pagefind || !q.trim()) {
    results.value = []
    return
  }
  loading.value = true
  try {
    const p = await window.Pagefind.init()
    const search = await p.search(q.trim())
    results.value = search.results.slice(0, 10)
    selectedIndex.value = -1
  } catch {
    results.value = []
  } finally {
    loading.value = false
  }
}

const debouncedSearch = debounce(doSearch, 200)

function onInput() {
  showResults.value = !!query.value.trim()
  debouncedSearch(query.value)
}

function onKeyDown(e) {
  if (!showResults.value || results.value.length === 0) return
  if (e.key === 'ArrowDown') {
    e.preventDefault()
    selectedIndex.value = (selectedIndex.value + 1) % results.value.length
  } else if (e.key === 'ArrowUp') {
    e.preventDefault()
    selectedIndex.value = selectedIndex.value <= 0 ? results.value.length - 1 : selectedIndex.value - 1
  } else if (e.key === 'Enter' && selectedIndex.value >= 0) {
    const r = results.value[selectedIndex.value]
    if (r) window.location.href = r.data.url
    showResults.value = false
    query.value = ''
  } else if (e.key === 'Escape') {
    showResults.value = false
    searchInput.value?.blur()
  }
}

function onClickOutside(e) {
  if (!e.target.closest('.pagefind-search')) {
    showResults.value = false
  }
}

onMounted(() => {
  document.addEventListener('click', onClickOutside)
  initPagefind()
})

onUnmounted(() => {
  document.removeEventListener('click', onClickOutside)
})
</script>

<template>
  <div class="pagefind-search">
    <input
      ref="searchInput"
      v-model="query"
      class="search-input"
      type="text"
      placeholder="搜索文档..."
      @input="onInput"
      @focus="onInput"
      @keydown="onKeyDown"
    />
    <div v-if="loading" class="search-loading">搜索中...</div>
    <ul v-if="showResults && results.length" class="search-results">
      <li
        v-for="(r, i) in results"
        :key="r.id"
        :class="{ selected: i === selectedIndex }"
      >
        <a :href="r.data.url" @click="showResults = false; query = ''">
          <span class="result-title">{{ r.data.meta?.title || r.data.url }}</span>
          <span class="result-excerpt" v-html="r.data.excerpt"></span>
        </a>
      </li>
    </ul>
    <div v-if="showResults && !loading && query.trim() && results.length === 0" class="search-empty">
      未找到结果
    </div>
  </div>
</template>

<style scoped>
.pagefind-search {
  position: relative;
  margin-right: 8px;
}

.search-input {
  width: 200px;
  padding: 6px 12px;
  border: 1px solid var(--vp-c-divider);
  border-radius: 8px;
  background: var(--vp-c-bg-alt);
  color: var(--vp-c-text-1);
  font-size: 14px;
  outline: none;
  transition: border-color 0.2s;
}

.search-input:focus {
  border-color: var(--vp-c-brand);
}

.search-input::placeholder {
  color: var(--vp-c-text-3);
}

.search-loading {
  position: absolute;
  top: 100%;
  left: 0;
  right: 0;
  padding: 12px;
  background: var(--vp-c-bg-soft);
  border: 1px solid var(--vp-c-divider);
  border-radius: 8px;
  margin-top: 4px;
  font-size: 13px;
  color: var(--vp-c-text-2);
  z-index: 100;
}

.search-results {
  position: absolute;
  top: 100%;
  left: 0;
  min-width: 400px;
  max-height: 400px;
  overflow-y: auto;
  background: var(--vp-c-bg-soft);
  border: 1px solid var(--vp-c-divider);
  border-radius: 8px;
  margin-top: 4px;
  padding: 8px 0;
  list-style: none;
  z-index: 100;
  box-shadow: 0 4px 16px rgba(0, 0, 0, 0.12);
}

.search-results li {
  padding: 0;
}

.search-results a {
  display: block;
  padding: 8px 16px;
  text-decoration: none;
  color: var(--vp-c-text-1);
  transition: background-color 0.15s;
}

.search-results a:hover,
.search-results li.selected a {
  background: var(--vp-c-bg-alt);
}

.result-title {
  display: block;
  font-size: 14px;
  font-weight: 500;
  margin-bottom: 2px;
  color: var(--vp-c-brand);
}

.result-excerpt {
  display: block;
  font-size: 12px;
  color: var(--vp-c-text-2);
  line-height: 1.5;
}

.result-excerpt :deep(mark) {
  background: var(--vp-c-brand-light);
  color: var(--vp-c-brand-dark);
  border-radius: 2px;
  padding: 0 2px;
}

.search-empty {
  position: absolute;
  top: 100%;
  left: 0;
  right: 0;
  padding: 12px;
  background: var(--vp-c-bg-soft);
  border: 1px solid var(--vp-c-divider);
  border-radius: 8px;
  margin-top: 4px;
  font-size: 13px;
  color: var(--vp-c-text-2);
  z-index: 100;
}
</style>
