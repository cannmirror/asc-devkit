export interface PagefindResult {
  url: string
  content: string
  word_count: number
  filters: Filters
  meta: Meta
  anchors: Anchor[]
  weighted_locations: WeightedLocation[]
  locations: number[]
  raw_content: string
  raw_url: string
  excerpt: string
  sub_results: SubResult[]
}

interface SubResult {
  title: string
  url: string
  anchor: Anchor
  weighted_locations: WeightedLocation[]
  locations: number[]
  excerpt: string
}

interface WeightedLocation {
  weight: number
  balanced_score: number
  location: number
}

interface Anchor {
  element: string
  id: string
  text: string
  location: number
}

interface Meta {
  image_alt: string
  title: string
  image: string
  base64: string
}

interface Filters {
}

export interface PagefindOption {
  excludeSelector?: string[]
  forceLanguage?: string
  indexingCommand?: string
}

export interface SearchItem {
  route: string
  meta: Record<string, any>
  result: PagefindResult
}
export interface SearchConfig {
  btnPlaceholder?: string
  placeholder?: string
  emptyText?: string
  heading?: string
  toSelect?: string
  toNavigate?: string
  toClose?: string
  searchBy?: string
  langReload?: boolean
  customSearchQuery?: (input: string) => string
  resultOptimization?: boolean
  filter?: (searchItem: SearchItem, idx: number, array: SearchItem[]) => boolean
  sort?: (a: SearchItem, b: SearchItem) => number
  showDate?: boolean | ((date: number, lang: string) => string)
  timeZone?: number
  locales?: Record<string, Omit<SearchConfig, 'locales'>>
  ignorePublish?: boolean
  manual?: boolean
  delay?: number
  pageResultCount?: number
  mpaDefaultUI?: boolean
}

export type PagefindConfig = PagefindOption & SearchConfig
