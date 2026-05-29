/**
* Copyright (c) 2026 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

const fs = require('fs')
const path = require('path')

const apiDir = path.resolve(__dirname, '..', 'docs', 'api')
const outDir = path.resolve(__dirname, '..', 'docs', 'public', 'api-source')

function walkDir(dir, entries) {
  const list = fs.readdirSync(dir)
  for (const name of list) {
    const full = path.join(dir, name)
    const stat = fs.statSync(full)
    if (stat.isDirectory()) {
      walkDir(full, entries)
    } else if (stat.isFile() && name.endsWith('.md')) {
      entries.push(full)
    }
  }
}

if (!fs.existsSync(apiDir)) {
  console.error('api dir not found:', apiDir)
  process.exit(1)
}

fs.mkdirSync(outDir, { recursive: true })

const files = []
walkDir(apiDir, files)

for (const src of files) {
  const rel = path.relative(apiDir, src)
  const dest = path.join(outDir, rel)
  fs.mkdirSync(path.dirname(dest), { recursive: true })
  fs.copyFileSync(src, dest)
}

console.log(`copied ${files.length} .md files to docs/public/api-source/`)
