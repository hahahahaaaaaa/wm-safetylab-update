/**
 * Adapted from FBddcz/embodied-jev frontend/robot-scene.js.
 * Copyright (c) 2026 EmbodiedJev contributors. MIT licensed.
 * Full upstream license: native/embodied_jev/UPSTREAM_LICENSE.txt.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
import * as THREE from 'three'
import { OrbitControls } from 'three/addons/controls/OrbitControls.js'

export type EmbodiedSceneData = {
  geometries: Array<{ id: number; name: string; type: number; size: number[]; color: number[]; mesh: number }>
  meshes: Record<string, { vertices: number[][]; faces: number[][] }>
  scene_hash?: string
  task?: string
  seed?: number
}

export type EmbodiedFrame = {
  positions: number[][]
  rotations: number[][]
  time?: number
  qpos?: number[]
  scene_hash?: string
}

export type EmbodiedCameraMode = 'orbit' | 'top'

type RobotMesh = THREE.Mesh<THREE.BufferGeometry, THREE.MeshStandardMaterial>

function vector(value: unknown, count: number, label: string): number[] {
  if (!Array.isArray(value) || value.length !== count || !value.every(v => typeof v === 'number' && Number.isFinite(v))) {
    throw new Error(`${label}缺失或包含无效数值`)
  }
  return value as number[]
}

/** Renders saved geometry poses only. No trajectory estimation or interpolation. */
export class RobotScene {
  private readonly container: HTMLDivElement
  private readonly scene = new THREE.Scene()
  private readonly camera = new THREE.PerspectiveCamera(36, 1, 0.01, 20)
  private readonly renderer: THREE.WebGLRenderer
  private readonly controls!: OrbitControls
  private readonly observer!: ResizeObserver
  private intersection: IntersectionObserver | null = null
  private visible = true
  private readonly robot = new THREE.Group()
  private readonly floor!: RobotMesh
  private readonly keyLight!: THREE.DirectionalLight
  private readonly matrix = new THREE.Matrix4()
  private readonly onError: (message: string) => void
  private objects = new Map<number, RobotMesh>()
  private geometries = new Set<THREE.BufferGeometry>()
  private materials = new Set<THREE.MeshStandardMaterial>()
  private sceneHash: string | undefined
  private animation: number | null = null
  private disposed = false
  private contextLost = false

  constructor(container: HTMLDivElement, label: string, onError: (message: string) => void) {
    this.container = container
    this.onError = onError
    this.scene.background = new THREE.Color('#eef3f4')
    this.camera.up.set(0, 0, 1)
    if (typeof ResizeObserver === 'undefined') throw new Error('当前浏览器不支持场景尺寸观察。')
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false })
    try {
      this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.5))
      this.renderer.shadowMap.enabled = true
      this.renderer.shadowMap.type = THREE.PCFSoftShadowMap
      this.renderer.outputColorSpace = THREE.SRGBColorSpace
      this.renderer.toneMapping = THREE.ACESFilmicToneMapping
      this.renderer.toneMappingExposure = 1
      this.renderer.domElement.setAttribute('role', 'img')
      this.renderer.domElement.setAttribute('aria-label', label)
      this.renderer.domElement.tabIndex = 0
      container.append(this.renderer.domElement)
      this.controls = new OrbitControls(this.camera, this.renderer.domElement)
      this.controls.enableDamping = true
      this.controls.dampingFactor = 0.12
      this.controls.minDistance = 0.45
      this.controls.maxDistance = 3.2
      this.controls.maxPolarAngle = Math.PI * 0.49
      this.controls.addEventListener('change', this.requestRender)
      this.renderer.domElement.addEventListener('webglcontextlost', this.handleContextLost)
      const hemisphere = new THREE.HemisphereLight(0xffffff, 0xa6b9ad, 1.3)
      hemisphere.position.set(0, 0, 3)
      this.scene.add(hemisphere)
      this.keyLight = new THREE.DirectionalLight(0xffffff, 2.2)
      this.keyLight.position.set(-0.8, -1.1, 2.5)
      this.keyLight.castShadow = true
      this.keyLight.shadow.mapSize.set(1024, 1024)
      Object.assign(this.keyLight.shadow.camera, { left: -1.2, right: 1.2, top: 1.2, bottom: -1.2 })
      this.keyLight.shadow.normalBias = 0.001
      this.keyLight.shadow.bias = -0.0001
      this.scene.add(this.keyLight)
      const fill = new THREE.DirectionalLight(0xdce9ff, 0.6)
      fill.position.set(1, 1, 1.4)
      this.scene.add(fill)
      this.floor = new THREE.Mesh(new THREE.PlaneGeometry(20, 20),
        new THREE.MeshStandardMaterial({ color: '#eef3f4', roughness: 0.93 }))
      this.floor.position.z = -0.075
      this.floor.receiveShadow = true
      this.scene.add(this.floor)
      this.robot.visible = false
      this.scene.add(this.robot)
      this.observer = new ResizeObserver(this.resize)
      this.observer.observe(container)
      if (typeof IntersectionObserver !== 'undefined') {
        this.intersection = new IntersectionObserver(entries => {
          if (this.disposed) return
          this.visible = entries.some(entry => entry.isIntersecting)
          if (this.visible) this.requestRender()
          else if (this.animation !== null) {
            cancelAnimationFrame(this.animation)
            this.animation = null
          }
        })
        this.intersection.observe(container)
      }
      this.setCamera('orbit')
      this.resize()
    } catch (error) {
      this.dispose()
      throw error
    }
  }

  private handleContextLost = (event: Event) => {
    event.preventDefault()
    if (this.disposed) return
    this.contextLost = true
    if (this.animation !== null) cancelAnimationFrame(this.animation)
    this.animation = null
    this.onError('浏览器的 3D 渲染上下文已丢失，请重新加载场景。')
  }

  private resize = () => {
    if (this.disposed || this.contextLost) return
    const width = this.container.clientWidth
    const height = this.container.clientHeight
    if (!width || !height) return
    this.renderer.setSize(width, height, false)
    this.camera.aspect = width / height
    this.camera.updateProjectionMatrix()
    this.requestRender()
  }

  setLabel(label: string) {
    this.renderer.domElement.setAttribute('aria-label', label)
  }

  setCamera(mode: EmbodiedCameraMode) {
    const damping = this.controls.enableDamping
    this.controls.enableDamping = false
    this.controls.update()
    this.controls.enableRotate = mode !== 'top'
    if (mode === 'top') {
      this.camera.position.set(0.42, -0.001, 1.85)
      this.controls.target.set(0.4, 0, 0.05)
    } else {
      this.camera.position.set(1.4, -1.65, 1.27)
      this.controls.target.set(0.32, 0, 0.24)
    }
    this.controls.update()
    this.controls.enableDamping = damping
    this.requestRender()
  }

  clear() {
    this.robot.visible = false
    this.robot.clear()
    this.objects.clear()
    this.geometries.forEach(geometry => geometry.dispose())
    this.materials.forEach(material => material.dispose())
    this.geometries.clear()
    this.materials.clear()
    this.sceneHash = undefined
    this.requestRender()
  }

  load(data: EmbodiedSceneData) {
    if (!Array.isArray(data.geometries) || !data.geometries.length || !data.meshes || typeof data.meshes !== 'object') {
      throw new Error('场景几何数据缺失，无法重建机器人。')
    }
    const objects = new Map<number, RobotMesh>()
    const geometries = new Set<THREE.BufferGeometry>()
    const materials = new Set<THREE.MeshStandardMaterial>()
    const meshCache = new Map<number, THREE.BufferGeometry>()
    try {
      for (const item of data.geometries) {
        if (!Number.isInteger(item.id) || item.id < 0 || objects.has(item.id)) throw new Error('场景几何编号无效或重复。')
        const size = vector(item.size, 3, '几何尺寸')
        const color = vector(item.color, 4, '几何颜色')
        if (size.some(value => value < 0) || color.some(value => value < 0 || value > 1)) throw new Error('场景尺寸或颜色超出有效范围。')
        let geometry: THREE.BufferGeometry
        if (item.type === 7) {
          const cached = meshCache.get(item.mesh)
          if (cached) geometry = cached
          else {
            const source = data.meshes[String(item.mesh)]
            if (!source || !Array.isArray(source.vertices) || !source.vertices.length || !Array.isArray(source.faces) || !source.faces.length) {
              throw new Error('机器人网格数据缺失，无法重建真实形状。')
            }
            source.vertices.forEach(vertex => vector(vertex, 3, '网格顶点'))
            source.faces.forEach(face => {
              if (vector(face, 3, '网格面').some(index => !Number.isInteger(index) || index < 0 || index >= source.vertices.length)) {
                throw new Error('机器人网格面索引无效。')
              }
            })
            geometry = new THREE.BufferGeometry()
            geometries.add(geometry)
            geometry.setAttribute('position', new THREE.Float32BufferAttribute(source.vertices.flat(), 3))
            geometry.setIndex(source.faces.flat())
            geometry.computeVertexNormals()
            meshCache.set(item.mesh, geometry)
          }
        } else if (item.type === 6) geometry = new THREE.BoxGeometry(size[0] * 2, size[1] * 2, size[2] * 2)
        else if (item.type === 2) geometry = new THREE.SphereGeometry(size[0], 24, 16)
        else if (item.type === 5) {
          geometry = new THREE.CylinderGeometry(size[0], size[0], size[1] * 2, 32)
          geometry.rotateX(Math.PI / 2)
        } else if (item.type === 3) {
          geometry = new THREE.CapsuleGeometry(size[0], size[1] * 2, 8, 24)
          geometry.rotateX(Math.PI / 2)
        } else if (item.type === 4) {
          geometry = new THREE.SphereGeometry(1, 24, 16)
          geometry.scale(size[0], size[1], size[2])
        } else if (item.type === 0) geometry = new THREE.PlaneGeometry((size[0] || 2) * 2, (size[1] || 2) * 2)
        else throw new Error(`当前回放不支持 MuJoCo 几何类型 ${item.type}。`)
        geometries.add(geometry)
        const material = new THREE.MeshStandardMaterial({
          color: new THREE.Color(color[0], color[1], color[2]).convertSRGBToLinear(),
          roughness: item.name === 'cube_geom' ? 0.35 : 0.56,
          metalness: 0.08, transparent: color[3] < 1, opacity: color[3],
        })
        materials.add(material)
        const mesh = new THREE.Mesh(geometry, material)
        mesh.castShadow = mesh.receiveShadow = true
        objects.set(item.id, mesh)
      }
    } catch (error) {
      geometries.forEach(geometry => geometry.dispose())
      materials.forEach(material => material.dispose())
      throw error
    }
    this.clear()
    this.objects = objects
    this.geometries = geometries
    this.materials = materials
    this.sceneHash = data.scene_hash
    for (const mesh of objects.values()) this.robot.add(mesh)
  }

  hideFrame() {
    this.robot.visible = false
    this.requestRender()
  }

  renderFrame(frame: EmbodiedFrame) {
    if (frame.scene_hash && this.sceneHash && frame.scene_hash !== this.sceneHash) throw new Error('轨迹帧与场景版本不匹配。')
    if (!Array.isArray(frame.positions) || !Array.isArray(frame.rotations)) throw new Error('此轨迹没有保存三维姿态数据。')
    // Validate the whole frame before changing any mesh, so a partial/corrupt
    // frame cannot mix new and stale poses in the displayed robot.
    for (const id of this.objects.keys()) {
      vector(frame.positions[id], 3, `几何 ${id} 的位置`)
      vector(frame.rotations[id], 9, `几何 ${id} 的旋转`)
    }
    for (const [id, mesh] of this.objects) {
      const position = frame.positions[id]
      const r = frame.rotations[id]
      mesh.position.set(position[0], position[1], position[2])
      this.matrix.set(r[0], r[1], r[2], 0, r[3], r[4], r[5], 0, r[6], r[7], r[8], 0, 0, 0, 0, 1)
      mesh.quaternion.setFromRotationMatrix(this.matrix)
    }
    this.robot.visible = true
    this.requestRender()
  }

  private requestRender = () => {
    if (this.animation !== null || this.disposed || this.contextLost || !this.visible) return
    this.animation = requestAnimationFrame(() => {
      this.animation = null
      if (this.disposed || this.contextLost || !this.visible || !this.container.clientWidth || !this.container.clientHeight) return
      try {
        // OrbitControls schedules another frame only while damping changes the
        // camera. An idle comparison with three scenes has no perpetual loop.
        this.controls.update()
        this.renderer.render(this.scene, this.camera)
      } catch {
        this.contextLost = true
        this.onError('3D 场景绘制失败，可重新加载或查看原始实验视频。')
      }
    })
  }

  dispose() {
    if (this.disposed) return
    this.disposed = true
    if (this.animation !== null) cancelAnimationFrame(this.animation)
    this.animation = null
    this.observer?.disconnect()
    this.intersection?.disconnect()
    this.controls?.removeEventListener('change', this.requestRender)
    this.controls?.dispose()
    this.renderer.domElement.removeEventListener('webglcontextlost', this.handleContextLost)
    this.clear()
    this.floor?.geometry.dispose()
    this.floor?.material.dispose()
    this.keyLight?.shadow.map?.dispose()
    this.scene.clear()
    this.renderer.renderLists.dispose()
    this.renderer.dispose()
    this.renderer.forceContextLoss()
    this.renderer.domElement.remove()
  }
}