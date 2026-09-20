# Gravity Four Web UI

围棋式 3D 人机对战和复盘界面。前端使用 React、React Three Fiber 和 ECharts；Minimax 引擎通过 Emscripten 编译到 WebAssembly，并运行在独立 Web Worker 中。

## 环境

- Node.js 22+
- pnpm 10+
- Emscripten SDK（`em++` 在 `PATH` 中）

## 开发

```bash
cd web
pnpm install
pnpm build:wasm
pnpm dev
```

如果 WASM 尚未构建，界面会自动使用仅供视觉和交互调试的轻量预览引擎，并在页面右下角明确提示。

## 构建

```bash
pnpm build
```

完整构建会先生成 `public/wasm/gravity4-engine.js` 和 `.wasm`，再进行 TypeScript 检查和 Vite 生产构建。

## 测试与 Benchmark

```bash
make native-test
pnpm test
pnpm benchmark:engine
```

深度 9 的完整候选分析在本机 Native 平均 2.305 秒、WASM 平均 2.397 秒；测试口径与逐次数据见 [Web 引擎 Benchmark](./BENCHMARK.md)。

## 交互

- 拖动棋盘旋转，滚轮缩放；
- 对局、分析、分层三种相机视角；
- 落点胜率开关和最佳点标记；
- 点击胜率曲线或底部手数，回到该手落子前查看全部候选；
- 切换执黑/执白会开始新对局；
- 深思模式固定搜索深度 9。
