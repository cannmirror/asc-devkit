/**
* Copyright (c) 2026 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

const { readdirSync, writeFileSync, existsSync, mkdirSync } = require('node:fs')
const { join, resolve } = require('node:path')

const docsDir = resolve(__dirname, '..', 'docs')
const manifestFile = resolve(__dirname, '..', '.stubs-manifest.json')
const figuresDir = join(docsDir, 'api', 'figures')

const PLACEHOLDER_PNG = Buffer.from(
  'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==',
  'base64'
)

function findAllHtml(dir) {
  const results = []
  const entries = readdirSync(dir, { withFileTypes: true })
  for (const entry of entries) {
    const fullPath = join(dir, entry.name)
    if (entry.isDirectory()) {
      results.push(...findAllHtml(fullPath))
    } else if (entry.name.endsWith('.html')) {
      results.push(fullPath)
    }
  }
  return results
}

function genStubs(subdir) {
  const dir = join(docsDir, subdir)
  if (!existsSync(dir)) {
    console.log(`Directory ${subdir}/ not found, skipping`)
    return
  }
  const htmlFiles = findAllHtml(dir)
  const stubs = []
  for (const htmlPath of htmlFiles) {
    const mdPath = htmlPath.replace(/\.html$/, '.md')
    if (!existsSync(mdPath)) {
      writeFileSync(mdPath, '', 'utf-8')
      stubs.push(mdPath)
    }
  }
  allStubs.push(...stubs)
  console.log(`  ${subdir}/: generated ${stubs.length} .md stubs (${htmlFiles.length} .html files)`)
}

const allStubs = []
genStubs('api')
genStubs('guide')

writeFileSync(manifestFile, JSON.stringify(allStubs, null, 2), 'utf-8')
console.log(`Total: ${allStubs.length} .md stubs generated`)

if (!existsSync(figuresDir)) {
  mkdirSync(figuresDir, { recursive: true })
}
if (!existsSync(join(figuresDir, 'placeholder.png'))) {
  writeFileSync(join(figuresDir, 'placeholder.png'), PLACEHOLDER_PNG)
  console.log('Created figures/placeholder.png')
}
