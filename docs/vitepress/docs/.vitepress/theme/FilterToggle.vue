<script setup>
import { ref, watch, onMounted, nextTick } from 'vue'
import { useRoute } from 'vitepress'

const route = useRoute()
const isApiPage = ref(false)

const filterOptions = [
  { value: 'all', label: '全量' },
  { value: '950', label: '950' },
]

const selectedFilter = ref('all')

function loadSavedFilter() {
  try {
    const saved = localStorage.getItem('cann-filter-selection')
    if (saved && filterOptions.some(o => o.value === saved)) {
      selectedFilter.value = saved
    }
  } catch {}
}

function applyFilter() {
  if (typeof document === 'undefined') return

  const content = document.querySelector('.vp-doc')
  if (!content) return

  const allFilterDivs = content.querySelectorAll('[data-filter]')

  if (selectedFilter.value === 'all') {
    allFilterDivs.forEach(el => { el.style.display = '' })
  } else {
    allFilterDivs.forEach(el => {
      el.style.display = el.getAttribute('data-filter') === selectedFilter.value ? '' : 'none'
    })
  }
}

function onFilterChange() {
  localStorage.setItem('cann-filter-selection', selectedFilter.value)
  applyFilter()
}

onMounted(async () => {
  loadSavedFilter()
  if (route.path.startsWith('/api/')) {
    isApiPage.value = true
  }
  await nextTick()
  setTimeout(applyFilter, 100)
})

watch(() => route.path, async (path) => {
  isApiPage.value = path.startsWith('/api/')
  if (isApiPage.value) {
    await nextTick()
    setTimeout(applyFilter, 200)
  }
}, { immediate: true })
</script>

<template>
  <div class="cann-filter-bar" :class="{ hidden: !isApiPage }">
    <select v-model="selectedFilter" class="filter-select" :disabled="!isApiPage" @change="onFilterChange">
      <option v-for="opt in filterOptions" :key="opt.value" :value="opt.value">
        {{ opt.label }}
      </option>
    </select>
  </div>
</template>
