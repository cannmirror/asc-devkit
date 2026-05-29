<script setup>
import DefaultTheme from 'vitepress/theme'
import FilterToggle from './FilterToggle.vue'
import BackToTop from './BackToTop.vue'
import CustomOutline from './CustomOutline.vue'
</script>

<template>
  <DefaultTheme.Layout>
    <template #nav-bar-content-after>
      <FilterToggle />
    </template>
    <template #aside-outline-before>
      <CustomOutline />
    </template>
    <template #layout-bottom>
      <BackToTop />
    </template>
  </DefaultTheme.Layout>
</template>
