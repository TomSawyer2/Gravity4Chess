import { Html, OrbitControls, RoundedBox } from '@react-three/drei'
import { Canvas, useFrame, useThree } from '@react-three/fiber'
import { useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'
import * as THREE from 'three'
import type { AnalysisResult, Side, ViewMode } from '../types'

interface BoardSceneProps {
  moves: number[]
  analysis: AnalysisResult | null
  heatmapEnabled: boolean
  viewMode: ViewMode
  interactive: boolean
  selectedMove: number | null
  animateLastMove: boolean
  onColumnClick: (move: number) => void
}

interface Piece {
  side: Side
  move: number
  layer: number
}

function buildStacks(moves: number[]): Side[][] {
  const stacks = Array.from({ length: 25 }, () => [] as Side[])
  moves.forEach((move, ply) => stacks[move]?.push(ply % 2 === 0 ? 'B' : 'W'))
  return stacks
}

function movePosition(move: number): [number, number] {
  const row = Math.floor(move / 5)
  const col = move % 5
  return [col - 2, row - 2]
}

function CameraRig({ mode }: { mode: ViewMode }) {
  const { camera, invalidate } = useThree()

  useEffect(() => {
    const orthographic = camera as THREE.OrthographicCamera
    if (mode === 'analysis') {
      camera.position.set(0.2, 10.8, 4.2)
      orthographic.zoom = 66
    } else if (mode === 'layers') {
      camera.position.set(8.4, 6.7, 9.2)
      orthographic.zoom = 57
    } else {
      camera.position.set(7.6, 7.4, 9.4)
      orthographic.zoom = 62
    }
    camera.lookAt(0, 0.9, 0)
    orthographic.updateProjectionMatrix()
    invalidate()
  }, [camera, invalidate, mode])

  return null
}

function StoneInstances({ pieces, side, spacing }: {
  pieces: Piece[]
  side: Side
  spacing: number
}) {
  const filtered = pieces.filter((piece) => piece.side === side)
  const ref = useRef<THREE.InstancedMesh>(null)
  const transform = useMemo(() => new THREE.Object3D(), [])

  useLayoutEffect(() => {
    if (!ref.current) return
    filtered.forEach((piece, index) => {
      const [x, z] = movePosition(piece.move)
      transform.position.set(x, 0.31 + piece.layer * spacing, z)
      transform.scale.set(1, 0.56, 1)
      transform.updateMatrix()
      ref.current?.setMatrixAt(index, transform.matrix)
    })
    ref.current.instanceMatrix.needsUpdate = true
    ref.current.computeBoundingSphere()
  }, [filtered, spacing, transform])

  if (filtered.length === 0) return null
  return (
    <instancedMesh ref={ref} args={[undefined, undefined, filtered.length]} castShadow receiveShadow>
      <sphereGeometry args={[0.365, 36, 20]} />
      <meshPhysicalMaterial
        color={side === 'B' ? '#161714' : '#eee5d2'}
        roughness={side === 'B' ? 0.24 : 0.38}
        metalness={side === 'B' ? 0.18 : 0.02}
        clearcoat={side === 'B' ? 0.72 : 0.42}
        clearcoatRoughness={0.24}
      />
    </instancedMesh>
  )
}

function AnimatedStone({ piece, spacing }: { piece: Piece; spacing: number }) {
  const mesh = useRef<THREE.Mesh>(null)
  const [x, z] = movePosition(piece.move)
  const targetY = 0.31 + piece.layer * spacing
  const startedAt = useRef<number | null>(null)

  useFrame(({ clock }, delta) => {
    if (!mesh.current) return
    if (startedAt.current === null) startedAt.current = clock.elapsedTime
    mesh.current.position.y = THREE.MathUtils.damp(
      mesh.current.position.y,
      targetY,
      12,
      delta,
    )
    const elapsed = clock.elapsedTime - startedAt.current
    const settle = elapsed > 0.22 ? Math.exp(-(elapsed - 0.22) * 8) * Math.sin((elapsed - 0.22) * 26) : 0
    mesh.current.scale.set(1 + settle * 0.035, 0.56 - settle * 0.04, 1 + settle * 0.035)
  })

  return (
    <mesh ref={mesh} position={[x, targetY + 3.2, z]} castShadow receiveShadow>
      <sphereGeometry args={[0.365, 36, 20]} />
      <meshPhysicalMaterial
        color={piece.side === 'B' ? '#161714' : '#eee5d2'}
        roughness={piece.side === 'B' ? 0.24 : 0.38}
        metalness={piece.side === 'B' ? 0.18 : 0.02}
        clearcoat={0.65}
      />
    </mesh>
  )
}

function BoardModel({
  moves,
  analysis,
  heatmapEnabled,
  viewMode,
  interactive,
  selectedMove,
  animateLastMove,
  onColumnClick,
}: BoardSceneProps) {
  const [hoveredMove, setHoveredMove] = useState<number | null>(null)
  const stacks = useMemo(() => buildStacks(moves), [moves])
  const spacing = viewMode === 'layers' ? 0.78 : 0.49
  const allPieces = useMemo(() => stacks.flatMap((stack, move) =>
    stack.map((side, layer) => ({ side, move, layer }))), [stacks])
  const animatedPiece = animateLastMove && moves.length > 0
    ? allPieces.find((piece) =>
        piece.move === moves[moves.length - 1] &&
        piece.layer === stacks[piece.move].length - 1)
    : undefined
  const pieces = animatedPiece
    ? allPieces.filter((piece) => piece !== animatedPiece)
    : allPieces
  const bestScore = analysis?.candidates.reduce(
    (best, candidate) => Math.max(best, candidate.score),
    Number.NEGATIVE_INFINITY,
  )
  const side = analysis?.sideToMove ?? (moves.length % 2 === 0 ? 'B' : 'W')

  useEffect(() => {
    document.body.style.cursor = hoveredMove !== null && interactive ? 'pointer' : ''
    return () => { document.body.style.cursor = '' }
  }, [hoveredMove, interactive])

  return (
    <group>
      <RoundedBox args={[6.15, 0.38, 6.15]} radius={0.18} smoothness={5} position={[0, -0.25, 0]} receiveShadow>
        <meshStandardMaterial color="#b9834d" roughness={0.72} />
      </RoundedBox>
      <RoundedBox args={[5.38, 0.12, 5.38]} radius={0.08} smoothness={4} position={[0, -0.01, 0]} receiveShadow>
        <meshStandardMaterial color="#d6a66a" roughness={0.62} />
      </RoundedBox>

      {Array.from({ length: 5 }, (_, index) => {
        const offset = index - 2
        return (
          <group key={index}>
            <mesh position={[offset, 0.065, 0]} receiveShadow>
              <boxGeometry args={[0.028, 0.018, 4.05]} />
              <meshStandardMaterial color="#4d3825" roughness={0.8} />
            </mesh>
            <mesh position={[0, 0.066, offset]} receiveShadow>
              <boxGeometry args={[4.05, 0.018, 0.028]} />
              <meshStandardMaterial color="#4d3825" roughness={0.8} />
            </mesh>
          </group>
        )
      })}

      <StoneInstances pieces={pieces} side="B" spacing={spacing} />
      <StoneInstances pieces={pieces} side="W" spacing={spacing} />
      {animatedPiece && <AnimatedStone piece={animatedPiece} spacing={spacing} />}

      {moves.length > 0 && (() => {
        const lastMove = moves[moves.length - 1]
        const [x, z] = movePosition(lastMove)
        const layer = stacks[lastMove].length - 1
        return (
          <mesh position={[x, 0.31 + layer * spacing + 0.215, z]} rotation={[-Math.PI / 2, 0, 0]}>
            <ringGeometry args={[0.065, 0.102, 28]} />
            <meshBasicMaterial color="#b94632" transparent opacity={0.95} />
          </mesh>
        )
      })()}

      {Array.from({ length: 25 }, (_, move) => {
        const [x, z] = movePosition(move)
        const height = stacks[move].length
        const candidate = analysis?.candidates.find((item) => item.move === move)
        const canPlace = interactive && height < 5
        const isHovered = hoveredMove === move && canPlace
        return (
          <group key={move}>
            <mesh
              position={[x, 1.55, z]}
              onPointerOver={(event) => {
                event.stopPropagation()
                if (canPlace) setHoveredMove(move)
              }}
              onPointerOut={() => setHoveredMove((current) => current === move ? null : current)}
              onClick={(event) => {
                event.stopPropagation()
                if (canPlace) onColumnClick(move)
              }}
            >
              <cylinderGeometry args={[0.47, 0.47, 3.1, 20]} />
              <meshBasicMaterial transparent opacity={0} depthWrite={false} />
            </mesh>

            {isHovered && (
              <mesh position={[x, 0.31 + height * spacing, z]} scale={[1, 0.56, 1]}>
                <sphereGeometry args={[0.37, 32, 18]} />
                <meshStandardMaterial
                  color={side === 'B' ? '#282a25' : '#fff8e9'}
                  transparent
                  opacity={0.48}
                  roughness={0.35}
                />
              </mesh>
            )}

            {heatmapEnabled && candidate && (
              <Html
                center
                position={[x, 0.79 + height * spacing, z]}
                zIndexRange={[30, 0]}
                style={{ pointerEvents: 'none' }}
              >
                <div
                  className={[
                    'move-probability',
                    candidate.score === bestScore ? 'is-best' : '',
                    candidate.move === selectedMove ? 'is-played' : '',
                    candidate.tag ? `is-${candidate.tag}` : '',
                  ].filter(Boolean).join(' ')}
                  style={{ '--rate': candidate.sideWinRate } as React.CSSProperties}
                >
                  {candidate.tag === 'win'
                    ? '胜'
                    : candidate.tag === 'block'
                      ? '防'
                      : `${Math.round(candidate.sideWinRate * 100)}%`}
                </div>
              </Html>
            )}
          </group>
        )
      })}
    </group>
  )
}

export function BoardScene(props: BoardSceneProps) {
  return (
    <div className="board-canvas" aria-label="三维重力四子棋棋盘">
      <Canvas
        orthographic
        shadows
        dpr={[1, 1.7]}
        camera={{ position: [7.6, 7.4, 9.4], zoom: 62, near: 0.1, far: 100 }}
        gl={{ antialias: true, alpha: true, powerPreference: 'high-performance' }}
      >
        <CameraRig mode={props.viewMode} />
        <ambientLight intensity={1.35} />
        <hemisphereLight args={['#fff4d8', '#473523', 1.25]} />
        <directionalLight
          castShadow
          position={[5, 10, 6]}
          intensity={2.1}
          shadow-mapSize={[1024, 1024]}
          shadow-camera-left={-6}
          shadow-camera-right={6}
          shadow-camera-top={6}
          shadow-camera-bottom={-6}
        />
        <BoardModel {...props} />
        <OrbitControls
          makeDefault
          target={[0, 0.85, 0]}
          enablePan={false}
          enableDamping
          dampingFactor={0.08}
          minZoom={46}
          maxZoom={86}
          minPolarAngle={0.42}
          maxPolarAngle={1.22}
          minAzimuthAngle={-0.86}
          maxAzimuthAngle={0.92}
        />
      </Canvas>
      <div className="sr-only" role="group" aria-label="键盘落子">
        {Array.from({ length: 25 }, (_, move) => {
          const candidate = props.analysis?.candidates.find((item) => item.move === move)
          const disabled = !props.interactive || !candidate
          const row = Math.floor(move / 5) + 1
          const col = String.fromCharCode(65 + (move % 5))
          return (
            <button
              type="button"
              key={move}
              disabled={disabled}
              onClick={() => props.onColumnClick(move)}
              aria-label={`${col}${row} 落子${candidate ? `，当前估算胜率 ${Math.round(candidate.sideWinRate * 100)}%` : ''}`}
            />
          )
        })}
      </div>
      <div className="board-help">拖动旋转 · 滚轮缩放 · 视角按钮快速定位</div>
    </div>
  )
}
