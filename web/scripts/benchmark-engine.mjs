import { spawnSync } from 'node:child_process'
import { dirname, resolve } from 'node:path'
import { fileURLToPath, pathToFileURL } from 'node:url'

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..')
const wasmDirectory = resolve(root, 'public/wasm')
const nativeBinary = resolve(root, 'engine/analyze_cli')
const engineFactory = (await import(
  pathToFileURL(resolve(wasmDirectory, 'gravity4-engine.js')).href
)).default
const wasm = await engineFactory({
  locateFile: (file) => resolve(wasmDirectory, file),
})

const runs = 3
const depth = 9

function compact(result) {
  return {
    bestMove: result.bestMove,
    score: result.score,
    candidates: result.candidates.length,
    elapsedMs: result.stats.elapsedMs,
    nodes: result.stats.nodes,
  }
}

function nativeRun() {
  const child = spawnSync(nativeBinary, ['', String(depth), '0', '32'], {
    encoding: 'utf8',
  })
  if (child.status !== 0) {
    throw new Error(child.stderr || `native benchmark exited ${child.status}`)
  }
  return compact(JSON.parse(child.stdout))
}

function wasmRun() {
  return compact(JSON.parse(wasm.analyzePosition('', depth, 0, 32)))
}

function benchmark(name, run) {
  const samples = Array.from({ length: runs }, run)
  const averageMs = samples.reduce((sum, sample) => sum + sample.elapsedMs, 0) / runs
  const reference = samples[0]
  for (const sample of samples) {
    if (
      sample.bestMove !== reference.bestMove ||
      sample.score !== reference.score ||
      sample.candidates !== reference.candidates
    ) {
      throw new Error(`${name} returned inconsistent results across runs`)
    }
  }
  return { name, averageMs, samples, ...reference }
}

const native = benchmark('native', nativeRun)
const browser = benchmark('wasm', wasmRun)
const ratio = browser.averageMs / native.averageMs

console.log('Depth 9 full-candidate benchmark (empty board, 3 runs)')
for (const result of [native, browser]) {
  const samples = result.samples.map((sample) => sample.elapsedMs.toFixed(3)).join(', ')
  console.log(
    `${result.name.padEnd(6)} avg=${result.averageMs.toFixed(3)}ms ` +
    `samples=[${samples}] nodes=${result.nodes} best=${result.bestMove} ` +
    `score=${result.score} candidates=${result.candidates}`,
  )
}
console.log(`wasm/native=${ratio.toFixed(3)}x`)

if (ratio > 2) {
  throw new Error(`WASM regression gate failed: ${ratio.toFixed(3)}x > 2.000x`)
}
