import { LineChart } from 'echarts/charts'
import {
  GridComponent,
  MarkLineComponent,
  TooltipComponent,
} from 'echarts/components'
import * as echarts from 'echarts/core'
import { SVGRenderer } from 'echarts/renderers'
import { useEffect, useRef } from 'react'
import type { HistoryEntry } from '../types'

echarts.use([LineChart, GridComponent, MarkLineComponent, TooltipComponent, SVGRenderer])

interface WinRateChartProps {
  history: HistoryEntry[]
  initialBlackWinRate: number
  selectedPly: number | null
  onSelectPly: (ply: number) => void
}

export function WinRateChart({
  history,
  initialBlackWinRate,
  selectedPly,
  onSelectPly,
}: WinRateChartProps) {
  const container = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!container.current) return
    const chart = echarts.init(container.current, undefined, { renderer: 'svg' })
    const black = [initialBlackWinRate, ...history.map((entry) => entry.blackWinRate)]
    const white = black.map((rate) => 1 - rate)
    const labels = ['开局', ...history.map((entry) => `${entry.ply}`)]

    chart.setOption({
      animationDuration: 480,
      animationEasing: 'cubicOut',
      grid: { left: 34, right: 18, top: 18, bottom: 30 },
      tooltip: {
        trigger: 'axis',
        backgroundColor: 'rgba(28, 27, 23, .96)',
        borderColor: '#76634c',
        textStyle: { color: '#f2e8d4', fontFamily: 'Songti SC, serif' },
        formatter: (params: unknown) => {
          const items = params as Array<{ dataIndex: number }>
          const index = items[0]?.dataIndex ?? 0
          if (index === 0) {
            return `开局<br/>黑方 ${(black[0] * 100).toFixed(1)}%<br/>白方 ${(white[0] * 100).toFixed(1)}%`
          }
          const entry = history[index - 1]
          return `第 ${entry.ply} 手 · ${entry.side === 'B' ? '黑' : '白'}方<br/>黑方 ${(black[index] * 100).toFixed(1)}%<br/>白方 ${(white[index] * 100).toFixed(1)}%`
        },
      },
      xAxis: {
        type: 'category',
        boundaryGap: false,
        data: labels,
        axisLine: { lineStyle: { color: '#625847' } },
        axisTick: { show: false },
        axisLabel: { color: '#a99c87', interval: Math.max(0, Math.floor(labels.length / 7)) },
      },
      yAxis: {
        type: 'value',
        min: 0,
        max: 1,
        interval: 0.25,
        axisLabel: {
          color: '#a99c87',
          formatter: (value: number) => `${Math.round(value * 100)}%`,
        },
        splitLine: { lineStyle: { color: 'rgba(226, 213, 187, .09)' } },
      },
      series: [
        {
          name: '黑方',
          type: 'line',
          data: black,
          symbol: 'circle',
          symbolSize: 6,
          showSymbol: history.length < 26,
          lineStyle: { width: 2.5, color: '#10110f' },
          itemStyle: { color: '#10110f', borderColor: '#bda887', borderWidth: 1 },
          areaStyle: {
            color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
              { offset: 0, color: 'rgba(9, 10, 9, .36)' },
              { offset: 1, color: 'rgba(9, 10, 9, 0)' },
            ]),
          },
          markLine: {
            silent: true,
            symbol: 'none',
            label: { show: false },
            data: [
              { yAxis: 0.5, lineStyle: { color: '#8c7c63', type: 'dashed', width: 1 } },
              ...(selectedPly === null
                ? []
                : [{ xAxis: selectedPly, lineStyle: { color: '#b94632', width: 1.5 } }]),
            ],
          },
        },
        {
          name: '白方',
          type: 'line',
          data: white,
          symbol: 'circle',
          symbolSize: 6,
          showSymbol: history.length < 26,
          lineStyle: { width: 2.2, color: '#f0e5cf' },
          itemStyle: { color: '#f0e5cf', borderColor: '#5b5143', borderWidth: 1 },
        },
      ],
    })

    chart.on('click', (params) => {
      if (typeof params.dataIndex === 'number' && params.dataIndex > 0) {
        onSelectPly(params.dataIndex)
      }
    })
    const observer = new ResizeObserver(() => chart.resize())
    observer.observe(container.current)
    return () => {
      observer.disconnect()
      chart.dispose()
    }
  }, [history, initialBlackWinRate, onSelectPly, selectedPly])

  return <div className="win-rate-chart" ref={container} aria-label="双方胜率曲线" />
}
