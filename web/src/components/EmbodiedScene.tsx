/** MuJoCo geometry viewer adapted from MIT-licensed EmbodiedJev; see robot-scene.ts. */
import { memo, useEffect, useRef, useState } from 'react'
import { RobotScene, type EmbodiedCameraMode, type EmbodiedFrame, type EmbodiedSceneData } from './robot-scene'
import './embodiedScene.css'

export type { EmbodiedCameraMode, EmbodiedFrame, EmbodiedSceneData } from './robot-scene'

export type EmbodiedSceneProps = {
  scene: EmbodiedSceneData | null
  frame: EmbodiedFrame | null
  label?: string
  cameraMode?: EmbodiedCameraMode
  onCameraModeChange?: (mode: EmbodiedCameraMode) => void
  className?: string
}

export const EmbodiedScene = memo(function EmbodiedScene({
  scene, frame, label = 'MuJoCo 机械臂场景', cameraMode, onCameraModeChange, className = '',
}: EmbodiedSceneProps) {
  const host = useRef<HTMLDivElement>(null)
  const renderer = useRef<RobotScene | null>(null)
  const loadedScene = useRef<EmbodiedSceneData | null>(null)
  const [ready, setReady] = useState(false)
  const [frameReady, setFrameReady] = useState(false)
  const [rendererError, setRendererError] = useState<string | null>(null)
  const [dataError, setDataError] = useState<string | null>(null)
  const [reload, setReload] = useState(0)
  const [localMode, setLocalMode] = useState<EmbodiedCameraMode>('orbit')
  const mode = cameraMode ?? localMode
  const initial = useRef({ label, mode })
  initial.current = { label, mode }

  useEffect(() => {
    if (!host.current) return
    setReady(false)
    setFrameReady(false)
    setRendererError(null)
    setDataError(null)
    loadedScene.current = null
    let instance: RobotScene | null = null
    try {
      instance = new RobotScene(host.current, initial.current.label, setRendererError)
      instance.setCamera(initial.current.mode)
      renderer.current = instance
      setReady(true)
    } catch {
      instance?.dispose()
      renderer.current = null
      setRendererError('当前浏览器无法创建 WebGL 场景。请启用图形加速，或查看该回合的实验视频。')
    }
    return () => {
      instance?.dispose()
      renderer.current = null
      loadedScene.current = null
    }
  }, [reload])

  useEffect(() => {
    renderer.current?.setLabel(label)
  }, [label, ready])

  useEffect(() => {
    renderer.current?.setCamera(mode)
  }, [mode, ready])

  useEffect(() => {
    const viewer = renderer.current
    if (!viewer || !ready || rendererError) return
    try {
      if (!scene) {
        viewer.clear()
        loadedScene.current = null
        setFrameReady(false)
        setDataError(null)
        return
      }
      if (loadedScene.current !== scene) {
        viewer.load(scene)
        loadedScene.current = scene
      }
      if (!frame) {
        viewer.hideFrame()
        setFrameReady(false)
      } else {
        viewer.renderFrame(frame)
        setFrameReady(true)
      }
      setDataError(null)
    } catch (error) {
      viewer.hideFrame()
      setFrameReady(false)
      setDataError(error instanceof Error ? error.message : '场景或轨迹数据无法读取。')
    }
  }, [scene, frame, ready, reload, rendererError])

  function selectCamera(next: EmbodiedCameraMode) {
    setLocalMode(next)
    onCameraModeChange?.(next)
    if (!cameraMode) renderer.current?.setCamera(next)
  }

  const error = rendererError || dataError
  const enabled = ready && !error && frameReady
  const timestamp = typeof frame?.time === 'number' && Number.isFinite(frame.time) ? frame.time.toFixed(2) : null

  return <section className={`embodied-scene ${className}`} aria-label={label}>
    <div className="embodied-scene__heading">
      <div className="embodied-scene__title"><span className="embodied-scene__marker" aria-hidden="true" />
        <strong title={label}>{label}</strong><span className="embodied-scene__tag">3D</span>
      </div>
      <div className="embodied-scene__controls" aria-label="场景视角">
        <button type="button" aria-pressed={mode === 'orbit'} disabled={!enabled}
          title="自由视角，可拖动旋转" onClick={() => selectCamera('orbit')}>自由</button>
        <button type="button" aria-pressed={mode === 'top'} disabled={!enabled}
          title="固定俯视，可平移与缩放" onClick={() => selectCamera('top')}>俯视</button>
        <button type="button" disabled={!enabled} title="恢复当前视角的初始位置"
          onClick={() => renderer.current?.setCamera(mode)}>复位</button>
      </div>
    </div>
    <div className="embodied-scene__viewport">
      <div className="embodied-scene__canvas" ref={host} />
      {(!enabled || error) && <div className={`embodied-scene__message${error ? ' embodied-scene__message--error' : ''}`}
        role={error ? 'alert' : 'status'}>
        <span className="embodied-scene__message-icon" aria-hidden="true">{error ? '!' : '◇'}</span>
        <strong>{error ? '暂时无法显示三维场景' : !scene ? '等待场景数据' : !frame ? '等待轨迹帧' : '正在加载三维场景'}</strong>
        <p>{error || '载入已保存的仿真几何与姿态后，即可旋转、俯视和逐帧查看。'}</p>
        {rendererError && <button type="button" onClick={() => setReload(value => value + 1)}>重新加载场景</button>}
      </div>}
    </div>
    <div className="embodied-scene__footer">
      <span>{mode === 'top' ? '俯视 · 右键平移 · 滚轮缩放' : '拖动旋转 · 右键平移 · 滚轮缩放'}</span>
      <span className="embodied-scene__time">{enabled && timestamp !== null ? `仿真 ${timestamp} s` : '原始仿真姿态'}</span>
    </div>
  </section>
})

export default EmbodiedScene