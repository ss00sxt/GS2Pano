[AGENTS.md](https://github.com/user-attachments/files/29449293/AGENTS.md)
# GS2Pano

> 3DGS 场景（.ply）→ 360° equirectangular 全景图 的数据准备工具。给下游「单张全景图 → 3DGS」前馈网络做训练数据。

---

## 1. 项目概述

**做什么**（输入 → 输出，一句话）：

- 输入：现成的 3DGS `.ply` 文件 + 相机位姿（支持 `camera_params.json` 和 COLMAP `images.bin` 两种格式）
- 输出 1：单张 360° equirectangular 全景 RGB 图（默认 W=1024, H=512）
- 输出 2：**射线-GS 配对数据**（专利第一阶段监督用，每像素对应穿过的高斯列表 + 距离）

**不做什么**（边界，避免 scope creep）：

- ❌ 不做深度图、mask、置信度等额外通道
- ❌ 不训 / 不评估下游网络
- ❌ 不解码 `.splat` / `.ksplat`（第一版只支持 `.ply`）
- ❌ 不做 6 cube 拼接（cube map 有几何接缝，已弃用）
- ❌ 不自研 CUDA / 自研光栅化（用 gsplat）

**主方法**：**球面投影 + gsplat 3DGUT 逐像素射线渲染**——

**第一阶段 — 球面投影 → Tile 分配**（Python, ~0.1s）：
- 每个 3D 高斯球映射到 equirectangular 像素坐标 `(u, v)`
- 用保守的 3σ 角张量计算像素半径 `(r_u, r_v)`
- 调用 gsplat 的 `isect_tiles()` 完成 tile 分配（每个 tile 只包含其球面投影区域覆盖的高斯）
- 公式见 §7「球面投影算法」

**第二阶段 — 逐像素射线渲染**（CUDA kernel, ~0.10s）：
- 按专利 equirectangular 公式为每个像素生成独立射线
- `rasterize_to_pixels_eval3d_extra()` 直接做 per-ray 3DGUT 评估
- 每条射线对 tile 内高斯做 `hit_t ≥ 0` + `d² < 9` + α-blending
- 无接缝、无 pinhole 畸变、纯 360° 全景

---

## 2. 工作目录

仓库根：`/home/user/Projects_Sun/GS2Pano/`

```
GS2Pano/
├── AGENTS.md                      # 本文件
├── README.md                      # 项目说明 + 快速上手
├── scripts/
│   └── render_panorama.py         # CLI 入口 (轻量, 仅参数解析+流程编排)
├── gs2pano/                        # 核心库
│   ├── load/                      # 数据加载
│   │   ├── ply.py                 # .ply → numpy (sigmoid 自动检测)
│   │   └── poses.py               # COLMAP binary + JSON 位姿提取
│   ├── render/                    # 渲染管线
│   │   ├── projection.py          # 球面投影 (equirect/Mercator)
│   │   ├── rays.py                # 逐像素射线生成
│   │   └── engine.py              # GPU 上传 + tile 分配 + CUDA kernel
│   ├── output/                    # 数据输出
│   │   └── paired_data.py         # GS-射线配对数据集生成
│   └── outputs/                   # 演示素材
├── data/                          # 测试数据
├── datasets/                      # 外部数据集
├── submodules/gsplat              # gsplat 源码 (4 个 gs2pano patch)
└── patent/                        # 专利文档 (只读)
```

---

## 3. 工作风格

### 核心原则

1. **先轻量，再上量**——新想法先用 `test/` 里几条射线跑通，再做完整 1024×512 渲染。
2. **三分钟能跑出图**——从克隆仓库到能渲染一张图，三分钟内。
3. **专利是契约**——专利决定数据要支持什么；但**第一版只覆盖 §1 列的范围**。
4. **每段代码都详细注释**——不写复杂实现，写得能让接手的人 5 分钟看懂。
5. **不懂得多和开发者沟通**——技术细节不确定时，停下来问用户，不要自己拍脑袋。

### 不要做的事

- ❌ 不要在根目录堆 .py 脚本——用 `test/` 隔离
- ❌ 不要修改 `sphere_viewer.html`、专利 docx
- ❌ 不要在专利 docx 上加批注 / 改格式——只读
- ❌ 不要发明专利里没有的监督信号 / 评估指标
- ❌ 不要在没有测试 .ply 的情况下硬编码文件路径
- ❌ 不要用 `diff-gaussian-rasterization`（Inria 原版，编译痛苦）——gsplat 够用
- ❌ 不要自研 CUDA / 自研光栅化（除了 T6 配对数据需要的几何部分）
- ❌ 不要在 `submodules/gsplat/` 里改源码——要提 PR 到上游，或在 `src/` 里 patch

### 沟通约定

- 用户已经确认过的设计决策（§1, §4），不需再问
- 自由发挥的细节（具体 hyperparameter 调优）：按默认即可，不请示
- **真正需要请示的**：
  - 输出格式 / 通道有歧义
  - 需要引入新依赖（先问再装）
  - 改动会影响 §1 "不做什么" 列表
- **新规则**：对 gsplat / 3DGUT / 3DGS 内部行为不确定时，**先看 gsplat 源码 + 跑最小测试**，再问用户，不要靠猜。

---

## 4. 详细 todolist

按优先级排序。**T0-T2 是公共准备（已完成），T3 起是 3DGUT 主线（完全重做）**——旧版「T3-1 修 GLM / T3-2 HiGS / T4 手动光栅化」全部作废。

### 公共准备（已完成）

- [x] **T0：旧环境**（`gsplat` conda env, Python 3.10 + torch 2.4 + gsplat 1.5.3）— **被 T3 取代，仅作历史**
- [x] **T1：加载 .ply** — `src/load/load_ply.py`
  - HY_desk/HY_face 的 ply 里 `scale_*` 是 `log(s)`，需 `exp()`
  - `f_dc_*` 是 SH DC 项，颜色 = `0.5 + SH_C0 · f_dc`
- [x] **T2：相机位姿接口** — `src/load/camera.py`
  - **必须传入** `camera_params.json` 路径（不传则报错）
  - 提取 `c2w[:3, 3]` 作为位置，`c2w[2, :3]` 作为 forward
  - forward = c2w 第 3 行 = +Z 方向在世界系

### 主线：球面投影 + 逐像素射线渲染

- [x] **T3：环境配置** ✅ 2026-06-05

  编译命令（完整版，含 CUDA fix 后所需的所有参数）：
  ```bash
  cd submodules/gsplat
  CUDA_HOME=/usr/local/cuda-12.4 \
    BUILD_3DGUT=1 \
    MAX_JOBS=$(nproc) \
    NVCC_FLAGS="-allow-unsupported-compiler" \
    CC=/usr/bin/gcc CXX=/usr/bin/g++ \
    pip install --no-build-isolation -e .
  ```

- [x] **T4：gsplat 源码修改 + 球面投影渲染管线** ✅ 2026-06-08

  对 gsplat 做了 4 处修改（详见 §6 源码修改记录），构建了完整的两阶段渲染管线：

  **Step 1 — 球面投影 & Tile 分配**（Python, ~0.1s）
  - 每个 3D 高斯球 → equirectangular 像素坐标 `(u, v)`
  - 3σ 角张量 → 保守的像素半径 `(r_u, r_v)`
  - 调用 `isect_tiles(means2d, radii, depths)` → 每个 tile 仅包含其球面覆盖区域内的高斯
  - 结果：~2M tile 交叉（约 1.7/高斯），替代 brute-force 的 615M

  **Step 2 — 逐像素射线渲染**（CUDA kernel `rasterize_to_pixels_eval3d_extra`, ~0.10s）
  - 专利 equirectangular 公式逐像素生成射线
  - Per-ray 3DGUT 评估：`hit_t ≥ 0` + `d² < 9` + front-to-back α-blending
  - 每条射线最终渲染 ~1-50 个高斯

  **验证结果**（HY_room, 256×512 → 1024×512）：
  | 场景 | 分辨率 | 时间 | 显存 | 覆盖 |
  |------|--------|------|------|------|
  | HY_room | 1024×512 | 0.10s | 138 MB | 31.8% |
  | mkbgs (cam 1592) | 1024×512 | ~0.10s | ~400 MB | 99.3% |

  **支持的位姿格式**：
  - `camera_params.json`（HY_* 系列）：直接读取 c2w 矩阵
  - COLMAP `sparse/images.bin`（mkbgs 等）：标准 w2c 约定 `pos = -R^T·t`

- [x] **T5：支持墨卡托投影全景图格式** ✅ 2026-06-08

  已实现 `--projection mercator`。两种投影：

  | | Equirect | Mercator |
  |--|----------|----------|
  | φ → v | v = (φ+π/2)/π·H | ψ=ln(tan(π/4+φ/2)), v=(1+ψ/π)/2·H |
  | v → φ | φ = π(v+0.5)/H - π/2 | ψ=π(2(v+0.5)/H-1), φ=2atan(e^ψ)-π/2 |
  | 像素半径 | r_v = Δ·H/π | r_v = Δ·H/π·(1/cos φ) |

- [ ] **T6：GS-射线配对数据集生成** ⏳ 进行中 (2026-06-11, numba加速完成, 待批量优化)

  **渲染质量修复**（2026-06-09 已完成）：

  发现问题：mkbgs 渲染模糊、颜色不对。排查过程：

  | 尝试 | opacity | 颜色 | 结果 |
  |------|---------|------|------|
  | 初版 | raw（无 sigmoid） | DC only | 模糊，色块重叠 |
  | SH3 世界帧 | sigmoid | SH3（世界帧） | 清晰，颜色偏暗（SH 方向错误） |
  | **最终方案** | **sigmoid** | **DC only** | **清晰，颜色正确** |

  **根因**：mkbgs PLY 存的是 raw logit opacity（范围 [-7, 18]），不是 sigmoid 后的值。直接传入 kernel 导致所有 GS 的 `alpha≈0.99`，失去遮挡关系。

  **SH 评估教训**：手动实现的 SH basis 与 gsplat `spherical_harmonics()` 不一致（差 ~0.47）；gsplat SH 用世界帧方向评估，不需转到局部帧；全景 360° 下逐像素 SH 才是正确的（当前用 camera→GS 方向的单方向近似，T6 待优化）。

  已实现基础功能（`--paired-data`），支持两种模式：

  | 模式 | `--pair-mode` | 过滤条件 | 64×32 配对量 | 用途 |
  |------|---------------|----------|-------------|------|
  | 视锥体 | `frustum` | GS 像素 bbox 覆盖该像素 | 49M 对 | 全量配对（数据量大） |
  | 射线过滤 | `ray` (默认) | `hit_t>0`, `d²<9`, `T>1e-4` | 8K 对 | 和渲染一致（紧凑） |

  **已验证**：
  - 像素覆盖 99.8% 匹配渲染结果（配对数据正确识别了哪些像素有可见 GS）
  - 每条射线平均 4 个 GS（3-16 范围）
  - α-blending 重建的颜色与 CUDA 渲染有偏差（Python 实现的 3DGUT 评估精度不足）

  **性能瓶颈**：
  - Python 逐像素嵌套循环计算 `hit_t`/`d²`，64×32 需 ~30s
  - 1024×512 预估 ~2 小时（524K 像素 × ~2400 GS/tile 的 numpy 运算）
  
  **待优化**：
  - 方案 A：numpy 向量化（batch 整个 tile 的像素×GS 计算）
  - 方案 B：修改 CUDA kernel 直接输出 per-Gaussian 数据（最彻底，颜色值精确匹配）
  - 可调参数：`--pair-d2-thresh`（默认 9.0）、`--pair-T-thresh`（默认 1e-4）

**当前进度**：T0 ✅，T1 ✅，T2 ✅，T3 ✅，T4 ✅，T5 ✅，T6 进行中。

---

## 5. 配对数据 schema（JSON 格式）

```json
{
  "camera": {
    "pos": [x, y, z],
    "forward": [dx, dy, dz]
  },
  "pano_shape": [W, H],
  "pixels": [
    {
      "pixel": [u, v],          // 二维坐标
      "theta": -1.5,
      "phi": 0.2,
      "gaussians": [
        {
          "idx": 42,            // .ply 里的 index
          "t": 0.85,            // 沿射线的位置（高斯中心最接近射线的点）
          "sigma": 0.5,         // 物理密度
          "alpha": 0.3,         // 该高斯对此像素的局部不透明度
          "T": 0.7              // 累积透过率（到达该高斯时）
        }
      ]
    }
  ]
}
```

**约定**：
- **不存** `ray_origin`（可从 `camera.pos` 算出）
- **不存**像素颜色（可从全景图反查）
- **存所有没被剔除的 GS**（当前 `OCCLUSION_THRESHOLD=∞`，实际无剔除）

---

## 6. 工作注意事项

### 坐标系（避免方向错乱）

- OpenCV / gsplat 惯例：**X 右、Y 下、Z 前**
- **v=0 是天顶**（方向 = 世界 -Y，OpenCV 系"上"是 -Y）
- **全景图中心 (u=W/2, v=H/2) = 相机前向** = c2w 第 3 行
- 用户**必须**传 `camera_params.json`（无默认）

### gsplat 3DGUT 使用陷阱

- **必须 `packed=False`**——3DGUT 跟 packed 模式不兼容（实测会报 `Packed mode is not supported with UT`）
- **`rays` 形状**：`[C, H, W, 6]`，最后一维是 `(ox, oy, oz, dx, dy, dz)`——见 `gsplat/rendering.py:289`
- **⚠️ 上游 bug（2026-06-04）**：`rendering.py` 接受 `[C, H, W, 6]` 但传给 `_wrapper.py` 时期望 `[C, P, 6]`（P=H*W）。已在 rendering.py 加 reshape patch，标记 `FIXME(gs2pano)`。等上游修复后可 revert。
- **`with_ut=True` 和 `with_eval3d=True` 必须同时开**才能用 `rays`；3DGUT 不是默认路径
- **main 分支 API 变化**：参数名从 `sh0`+`sh_rest` 变成了 `colors`（预计算好的颜色 `[N, 3]` 或 `[C, N, 3]`），与 PyPI v1.5.3 不同
- **`info` 返回值**：3DGUT+rays 模式下返回的是一个 `[C, H, W, 1]` 的 Tensor（alpha），meta dict 里有 `depths`、`radii`、`isect_offsets` 等字段
- **PyPI 版 (v1.5.3) 没有 `rays` 参数**——必须从 main 分支源码装
- **HiGS 路径**（`from experimental import render_scene`）**不支持 rays / 3DGUT / 非 pinhole**——不要走那条路
- **`camera_model="pinhole"` 仍要传**（入口要求），但 K 矩阵在 rays 模式下**几乎不起作用**——`viewmat` 给出姿态，**实际投影方向完全由 `rays` 决定**
- **SH 系数维度**因 .ply 而异，HY_* 数据用 f_dc_* (0 阶)
- **标准 pinhole 路径**（非 3DGUT）在从本地源码 build 时 `projection_ewa_3dgs_fused_fwd` 报错——不影响 3DGUT 路径，但也说明 main 分支 API 在迭代中

### 脚本入口维护规则

- README 中推荐的入口必须走统一的新渲染路径：
  - `scripts/render_panorama.py`、`scripts/render_dl3dv_panos.py`：直接调用 `gs2pano.render.engine.render()`
  - `scripts/render_pano.py`：也必须直接调用 `engine.render()`，不要再手写 projection/tile/rays/rasterize 流程
  - `scripts/render_and_pair.py`、`scripts/batch_mipnerf360.py`：因为需要设置 pair buffer，可以保留底层 `rasterize_to_pixels_eval3d_extra()`，但 tile 输入必须复用 `gs2pano.render.engine._build_tile_inputs()`
- 不要在脚本里重新实现旧版 `spherical_project() -> isect_tiles()` 直连路径。旧路径缺少高纬度横向半径补偿和 `u=0/W` 接缝 wrap，容易在全景图上下高纬区域产生明显的 16×16 tile 色块。
- `gs2pano.load.poses.extract_json_poses()` 同时支持两种 JSON：
  - list schema：MipNerf360 / DL3DV `cameras.json`，字段为 `position`、`rotation`、`img_name`
  - dict schema：旧 `camera_params.json`，字段为 `extrinsics[].matrix`

### .ply 格式不统一

- inria 标准 / gsplat 内部 / 其他变体字段名可能不同
- 加载时先 `print(ply.keys())` 看看

**⚠️ opacity 存储方式不统一**（2026-06-09 发现）：

| 数据集 | opacity 格式 | 加载方式 |
|--------|-------------|----------|
| HY_room/face/desk | 已 sigmoid（[0,1]） | 直接使用 |
| mkbgs | **raw logit**（[-7,18]） | 需 `sigmoid(x)` |

如果 raw opacity 直接传入 CUDA kernel，几乎所有 GS 的 `alpha=0.99`（截断），导致：
- 前景 GS 无法遮挡背景 → **模糊、色块重叠**
- 颜色被大量叠加的 GS 稀释

**判断方法**：检查 opacity 范围是否在 [0,1]。如果 > 1（如 mkbgs 的 18.3），就是 raw logit，需要 sigmoid。

### 源码修改记录 (gs2pano patches)

以下是对 `submodules/gsplat` 源码的所有修改。**迁移时必须重新应用这些 patch**。

#### Patch 1: `gsplat/rendering.py` — rays shape 修复

**位置**: 第 581-585 行附近的 `if rays is not None:` 块
**变更**: 在 shape 断言后加 reshape，把 `[C, H, W, 6]` 转为 `[C, H*W, 6]`（上游 `rendering.py` 和 `_wrapper.py` 对 rays shape 约定不一致）
**标记**: `FIXME(gs2pano)`

```python
if rays is not None:
    assert_shape("rays", rays, batch_dims + (C, H, W, 6))
    rays = rays.reshape(*batch_dims, C, H * W, 6)
```

#### Patch 2: `gsplat/rendering.py` — near_plane 调整

**位置**: 第 714-719 行 `fully_fused_projection_with_ut` 调用前
**变更**: 当 `rays is not None` 时用 `_near_plane = -1e10` 代替默认的 `near_plane=0.01`，避免 360° 全景中 pinhole 相机把高斯标记为"背面"
**标记**: `FIXME(gs2pano)`

```python
_near_plane = -1e10 if rays is not None else near_plane
# ... 调用 fully_fused_projection_with_ut(..., near_plane=_near_plane, ...)
```

#### Patch 3: `gsplat/rendering.py` — radii 扩展

**位置**: 第 800 行 `valid_gaussians = ...` 之后
**变更**: 当 `rays is not None` 时，把所有高斯的 radii 扩展到覆盖全图，强制 `isect_tiles` 分配所有高斯到所有 tile
**标记**: `FIXME(gs2pano)`

```python
if rays is not None:
    radii = radii.clone()
    radii[..., 0] = max(width, height)
    radii[..., 1] = max(width, height)
```

#### Patch 4: `gsplat/cuda/csrc/RasterizeToPixelsFromWorld3DGSFwd.cu` — hit_t 剔除

**位置**: 第 374-380 行 `gro`/`grd` 计算之后，`gcrod` 之前
**变更**: 在 3DGUT per-ray kernel 中，跳过射线原点背后的高斯（`hit_t < 1e-6`）。原 kernel 只计算到无限射线的垂直距离，不区分前方/后方。
**标记**: `FIXME(gs2pano)`

```cuda
const float hit_t = glm::dot(grd, -gro);
if (hit_t < 1e-6f) {
    continue;
}
```

#### Patch 5 (✅ 成功): CUDA kernel `__device__` 全局指针输出配对数据

**日期**: 2026-06-11
**方法**: `__device__` 全局指针 + `extern "C"` setter，**零函数签名修改**。ctypes 直接调用 setter，不经过 TorchScript。
**目标**: 在 `RasterizeToPixelsFromWorld3DGSFwd.cu` kernel 中, 渲染同时用 `atomicAdd` 把 `(gid, hit_t, opac, alpha, T)` 写入 output buffer, 实现零额外开销的配对数据生成。

**修改的文件** (共 2 个):

| 文件 | 行号 | 修改 |
|------|------|------|
| `RasterizeToPixelsFromWorld3DGSFwd.cu:44-46` | +3 | `__device__ float *g_pair_buf` / `int *g_pair_cnt` / `int g_pair_cap` |
| `RasterizeToPixelsFromWorld3DGSFwd.cu:315-318` | +4 | `bool collecting = (g_pair_buf != nullptr)` |
| `RasterizeToPixelsFromWorld3DGSFwd.cu:320-326` | 改 | `__syncthreads_count` 始终调用（屏障），break 仅 `!collecting` |
| `RasterizeToPixelsFromWorld3DGSFwd.cu:380-386` | 改 | 内层循环移除 `!done` 条件，改为循环体内检查 |
| `RasterizeToPixelsFromWorld3DGSFwd.cu:416-422` | ~12 | 配对数据写入：通过几何筛选即写入，不受 `done` 控制 |
| `RasterizeToPixelsFromWorld3DGSFwd.cu:425` | +1 | `if (done) continue` — 跳过渲染但不跳过配对收集 |
| `RasterizeToPixelsFromWorld3DGSFwd.cu:433` | +1 | `done ? 0.0f : T` — 被遮挡 GS 的 T=0 |
| `RasterizeToPixelsFromWorld3DGSFwd.cu:440-443` | 改 | done 后 `if (!collecting) break` + `continue` |
| `RasterizeToPixelsFromWorld3DGSFwd.cu:731-735` | +5 | `extern "C" void gs2pano_set_pair_buffers(...)` — cudaMemcpyToSymbol |
| `Rasterization.h:312` | +1 | `extern "C"` 声明 |

**为何不像 Patch 1-4 修改 `_wrapper.py` / `ext.cpp`**: Patch 5 使用 `__device__` 全局指针而非修改函数签名，因此不需要改 TorchScript 绑定。Python 侧通过 ctypes 直接调用 setter。

**Python 调用**:
```python
import ctypes
lib = ctypes.CDLL("gsplat/gsplat/csrc.so")
lib.gs2pano_set_pair_buffers.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
lib.gs2pano_set_pair_buffers(buf.data_ptr(), cnt.data_ptr(), cap)
# ... 调用 rasterize_to_pixels_eval3d_extra() ...
lib.gs2pano_set_pair_buffers(0, 0, 0)  # 禁用
n = cnt.item(); pairs = buf[:n*6].reshape(n, 6)
# 格式: [pix_id, gid, hit_t, opac, alpha, T]
```

**重新编译命令**（修改任何 CUDA 源码后都需执行）:
```bash
cd submodules/gsplat
CUDA_HOME=/usr/local/cuda-12.4 \
  BUILD_3DGUT=1 \
  NVCC_FLAGS="-allow-unsupported-compiler" \
  CC=/usr/bin/gcc CXX=/usr/bin/g++ \
  pip install --no-build-isolation -e .
```

---
### GS 筛选算法全景

一条射线穿过场景，哪些 GS 参与 tile 分配？哪些被记录到配对数据？哪些被渲染？以下是完整流水线。

#### Tile 分配: 椭圆 bbox 相交

**GS 进入某个 tile 的唯一条件：其椭圆投影的轴对齐包围盒与该 tile 重叠。**

代码: `IntersectTile.cu:196-249`
```
① 逆协方差 conic = (A, B, C) = Sigma^-1 上三角
② 等值线阈值 t = min(3.33^2, 2 * ln(opacity / (1/255)))
③ SNUGBOX: 椭圆 AAB = [mean - sqrt(-t*C/disc), mean + sqrt(-t*A/disc)]
④ tile 坐标范围 = [AAB_min / tile_size, AAB_max / tile_size + 1)
⑤ 对每个覆盖的 tile, 写入 isect_id = (image_id | tile_id | depth_bitcast)
⑥ Radix sort 按 (tile_id, depth) -> tile 内 GS 前->后排列
```

**没有 d^2 / opacity / T 过滤**。GS 能碰到 tile 就进入。

#### Per-ray 评估: 双阶段筛选

代码: `RasterizeToPixelsFromWorld3DGSFwd.cu:374-478`

每个像素独立评估射线与 tile 内 GS：

```
① gro = S^{-1} R^T (ray_o - mu)       # GS 局部坐标系中相机位置
② grd = normalize(S^{-1} R^T ray_d)    # GS 局部坐标系中射线方向
③ hit_t = grd . (-gro)                # 沿射线到最近点的距离
   若 hit_t < 1e-6 -> skip             # Patch 4
④ d^2 = ||cross(grd, gro)||^2         # 马氏距离平方
⑤ max_response = exp(-d^2 / 2)
⑥ alpha = min(0.99, opac * max_response)
```

**第一关: 几何筛选 -> 控制「记录」**

| 条件 | 结果 |
|------|------|
| `hit_t < 1e-6` | skip |
| `alpha < 1/255` | skip |
| `max_response <= 0.0113` (即 `d^2 >= 8.97`) | skip |
| 以上全通过 | **写入配对数据** |

配对数据格式 (6-field float32):
```
[pix_id, gid, hit_t, opac, alpha, T]
```
- 可见 GS: `T = 实际 transmittance`
- 被遮挡 GS: `T = 0`

**第二关: 累积筛选 -> 控制「渲染」**

```
⑦ next_T = T * (1 - alpha)
   若 next_T <= 1e-4 (TRANSMITTANCE_THRESHOLD) -> done = true
   若 done = true -> 跳过颜色累加 + T 更新
   若 done = false:
       pixel_color += GS_color * alpha * T
       T = next_T
```

**关键: `done` 不解耦配对数据收集。** 被遮挡 GS 仍被评估、记录（T=0）。

#### 筛选决策表

| hit_t | d^2 | alpha | next_T | done? | 记录? | 渲染? |
|-------|------|-------|--------|:-----:|:-----:|:-----:|
| < 1e-6 | - | - | - | - | No | No |
| >= 1e-6 | >= 8.97 | - | - | - | No | No |
| >= 1e-6 | < 8.97 | < 1/255 | - | - | No | No |
| >= 1e-6 | < 8.97 | >= 1/255 | > 1e-4 | false | T=真实值 | Yes |
| >= 1e-6 | < 8.97 | >= 1/255 | <= 1e-4 | 刚变 true | T=真实值 | No |
| (后续) | < 8.97 | >= 1/255 | - | true | T=0 | No |

#### 常量速查

| 常量 | 值 | 含义 |
|------|-----|------|
| `ALPHA_THRESHOLD` | 1/255 ≈ 0.0039 | alpha 下限 |
| `MAX_ALPHA` | 0.99 | alpha 截断上限 |
| `MAX_KERNEL_DENSITY_CUTOFF` | 0.0113 | max_response 下限, 对应 d^2≈8.97 |
| `TRANSMITTANCE_THRESHOLD` | 1e-4 | T 下限, = (1-MAX_ALPHA)^2 |

### 已知风险

- **球面投影的像素半径**: 已从旧的 `atan2(3σ_max, D)` 改为更保守的 `asin(3σ_max / D)`；若相机落入 3σ 范围内，角半径设为 `π`。这样可以避免近场高斯在脚下/极区被 tile 分配漏掉，但可能增加 tile 交叉数。
- **高纬度 tile 分配**: equirectangular 图中同一球面角半径在高纬度会覆盖更宽的 `u` 范围，`r_u` 必须乘 `1/cos(φ)`。缺少该补偿会在上下高纬区域出现 16×16 tile 色块边界。
- **全景水平接缝**: 跨 `u=0/W` 的高斯必须通过 `_build_tile_inputs()` 生成临时 wrap 副本，并把 `flatten_ids` 映射回原始 GS id。不要直接把原始 `ug/vg/pr/D` 传给 `isect_tiles()`。
- **Python 射线生成慢**: 1024×512 需 ~10s 纯 Python 循环生成射线；后续应 vectorize 或移到 CUDA
- **COLMAP 位姿约定**: `images.bin` 中 `q, t` 为标准 w2c（`X_cam = R·X_world + t`），相机中心 `C = -R^T·t`；部分非标准 pipeline 可能存 c2w，加载时需验证
- **SH 系数**: 部分 .ply 含 `f_rest_*`（SH degree≥1），当前只用 `f_dc_*`（degree 0）计算颜色，忽略 view-dependent 效果
- **T6 配对数据风险**: `hit_t` 和 `alpha` 在 kernel 内部以局部空间坐标计算，输出到 Python 层需要额外通道

---

## 8. 方案演进历史

### 起点：6 面立方体拼接（已废弃）

最初的方案是用 pinhole 相机渲染 6 个面（前、后、左、右、上、下），拼成全景图。发现有几何接缝（边缘像素不对齐），且 GPU memory 需要同时存 6 次渲染结果。废弃，改用球面投影。

### 第一阶段：球面投影 + 逐像素射线渲染（2026-06-05 ~ 06-08）

**方法**：每个 3D GS 直接映射到 equirectangular 像素坐标 → `isect_tiles` 分配 tile → 逐像素生成世界空间射线 → gsplat 3DGUT kernel 评估。纯 CUDA 渲染，~0.1s 完成 1024×512。

| 里程碑 | 时间 | 说明 |
|--------|------|------|
| Patch 1-4 | 06-05~08 | gsplat 源码修改：rays reshape 修复、near_plane 调整、radii 扩展、hit_t 剔除 |
| 首次 1024×512 渲染 | 06-08 | HY_room 0.10s / 138MB；mkbgs 0.10s / 488MB |
| Mercator 支持 | 06-08 | `--projection mercator`，像素半径含 `1/cos(φ)` 校正 |

### 第二阶段：配对数据生成 — Python 方案（2026-06-09 ~ 06-11）

**目标**：输出每个像素对应的 GS 列表 + 距离 + α-blending 参数。

**初版**：CPU numpy 双重循环 + numba JIT。每个 tile 逐像素逐 GS 计算 hit_t / grayDist / mask。

| 分辨率 | 配对数 | 耗时 | 瓶颈 |
|--------|--------|------|------|
| 64×32 | ~8K | **103s** (numba) | Python 逐 tile 循环 + numba 排序 |
| 1024×512 | 预估 ~41M | **~6h 预估** | 像素半径大（max 509px），GS/像素多 |

**加速尝试（Python 侧）**：
- 方案 A (numba JIT)：103s → 基本版
- 方案 B (d² 阈值)：减少配对量但效果有限
- 方案 C (numba 完整重写)：~30s for 64×32，仍不够

### 第三阶段：CUDA kernel 内联配对数据输出（2026-06-11）

**核心思路**：在 gsplat 的 CUDA kernel 中，渲染的同时用 `atomicAdd` 把 `(gid, hit_t, opac, alpha, T)` 写入 output buffer。

#### 尝试一：修改函数签名 ❌

在 `RasterizeToPixelsFromWorld3DGSFwd.cu` 的 kernel 函数签名里加 `pair_data` / `pair_counter` 参数。牵涉 4 个文件（.cu、.h、.cpp、ext.cpp）。

**失败原因**：
1. TorchScript `m.def` / `m.impl` 参数数量不匹配（30 vs 28 args）
2. JIT 编译路径不受 `pip install` 参数控制，conda GCC 14 编译失败
3. 符号冲突 — bwd kernel、multiple template instantiations 仍引用旧签名

#### 尝试二：`__device__` 全局指针 + ctypes ✅

**不走 TorchScript**。在 kernel 中声明 `__device__ float *g_pair_buf` 全局指针，用 `extern "C"` setter 通过 `cudaMemcpyToSymbol` 设置。Python 侧用 `ctypes.CDLL` 直接调用 setter。

**修改文件数**：从 4 个降到 2 个（`.cu` + `.h`），完全不需要改 `_wrapper.py` / `ext.cpp`。

**64×32 验证** (mkbgs, cam 1592, equirect)：

| 版本 | 渲染 | 配对数 | 渲染一致性 | 重建误差 | 唯一 GS |
|------|------|--------|-----------|---------|---------|
| v6 (初版, 5-field, 无 pix_id) | 0.30s | 67K | ✗ max diff 0.966 | 98.5% mismatch | 78K |
| v7 (加 barrier 修复) | 0.17s | 116K | ✅ max diff 0.000 | 2.5% diff>0.01 | 78K |
| v8 (T=0 标记遮挡) | — | 116K | ✅ | — | 78K, 42% occlusion |

**关键 Bug 修复**：
1. **`__syncthreads_count` 屏障缺失**：当 `collecting=true` 时短路跳过了同步，导致 shared memory 竞争损坏渲染。修复：始终调用 `__syncthreads_count` 作为屏障。
2. **done 解耦**：`done` 标志只控制颜色累加，不控制配对数据收集。被遮挡 GS 被记录但 T=0。
3. **被遮挡 GS 的 T=0**：`g_pair_buf[slot*6+5] = done ? 0.0f : T`

**1024×512 结果** (mkbgs, cam 1592, equirect, ray 方法)：

| 指标 | 值 |
|------|-----|
| 渲染 | **0.1s** |
| 配对数 | 27.5M (visible 15.9M + occluded 11.7M) |
| 唯一 GS | 1.65M / 3.09M (53.3%) |
| 总耗时 | **55s** (含 NPZ 写出 + 验证) |
| 重建最大误差 | 0.049 (0.002% 像素 >0.01) |
| 对比 Python 方案 | **~400× 加速** (6h → 55s) |

### 第四阶段：Python 侧优化（2026-06-12）

用户手动进行的优化（在 CUDA kernel 基础上）：

| 优化 | 文件 | 效果 |
|------|------|------|
| 射线生成向量化 | `rays.py` | Python 双重循环 → numpy 广播，**20s → 0s** |
| 配对数据 GPU 加速 | `paired_data.py` | CPU `np.einsum` → GPU `torch.einsum` |
| 统一渲染+配对管线 | `engine.py` | `render_with_pairs()` — 球面投影+tile 分配只做一次 |
| 进度条 + ETA | `paired_data.py` | 每 5% tile 输出进度 |
| 图片上下翻转修复 | 多处 | 删除 `np.flipud()`、`v_saved = H-1-v` 翻转 |
| GPU 配对缓冲区集成 | `render_acc.py` 等 | ctypes 设置 pair buffers + frustum params |

### 第五阶段：锥体 (frustum) 方法替代射线方法（2026-06-12）

**动机**：原射线方法用像素中心的零宽度射线判定 GS 命中。锥体方法考虑像素的角宽度——只要像素锥体内的**任意一条**射线命中 GS，就记录配对。

**实现**：在 kernel 中计算 GS 在相机局部的球面坐标，找到像素 frustum `(θ_min, θ_max, φ_min, φ_max)` 内离 GS 最近的射线方向，对该方向跑完整 3DGUT。

**试错过程**：

| 版本 | 问题 | 表现 |
|------|------|------|
| v1 (偏移量近似) | `rel_world = ray_o - xyz` 方向反了 | 0 pairs |
| v2 (修正方向) | host/device 指针混淆，`cudaMemcpyToSymbol` 传了 GPU 指针 | 0 pairs |
| v3 (修正指针) | 偏移量近似 `d² = ‖iscl_rot·offset‖²` 不是真正的 3DGUT 距离 | **25.9M pairs** (比射线 41.4M 少！) |
| v4 (完整 3DGUT) | 用最近射线方向跑完整 `grd × gro` 公式 | **54.0M pairs** (+30%) |

**1024×512 Mercator 对比** (mkbgs, origin)：

| | 射线方法 | 锥体方法 (v4) |
|------|---------|-------------|
| 总配对数 | 41.4M | **54.0M** |
| 可见 GS | 20.9M | 17.5M |
| 被遮挡 GS | 20.5M (49.5%) | 36.5M (**67.6%**) |
| 唯一 GS | 1.50M (48.7%) | 2.32M (**75.1%**) |
| 对/像素 | 78.9 | 103.0 |

成功实现了目标：锥体方法捕获了更多被遮挡 GS 和唯一 GS。

### 第六阶段：npz 输出格式升级 + 多场景验证（2026-06-12 ~ 06-15）

**npz 格式升级**：
- 配对按 `(pix_id, hit_t)` 排序 — 同像素连续，像素内前→后
- 内联 GS 属性：`xyz[N,3]`, `rgb[N,3]`, `scale[N,3]`, `quat[N,4]`
- 每像素 CSR 偏移：`pixel_starts[H*W+1]`
- 每像素 frustum 边界：`pixel_bounds[H*W,4]`
- 训练时一行代码取 GS：`data[k][pixel_starts[pid]:pixel_starts[pid+1]]`

**多场景验证** (1024×512 Mercator, frustum 方法)：

| 场景 | GS 数 | 渲染 | 配对数 | 唯一 GS | 总耗时 |
|------|-------|------|--------|---------|--------|
| mkbgs @ origin | 3.09M | 0.16s | 54.0M | 2.32M (75.1%) | ~27s |
| bicycle | 1.55M | 0.13s | 15.5M | 1.55M (100%) | ~9s |
| garden DSC07957 | 2.64M | 0.15s | 14.1M | 2.64M (100%) | **19.9s** |

### 完整性能演进总结

| 阶段 | 1024×512 配对耗时 | 加速比 | 方法 |
|------|-------------------|--------|------|
| Python numba (初版) | ~6h (预估) | 1× | CPU numpy + numba JIT |
| Python GPU 加速 | ~20min (预估) | ~18× | GPU torch.einsum |
| CUDA 全局指针 (ray) | **55s** | **~400×** | `__device__` 指针 + ctypes |
| CUDA 全局指针 (frustum) + 优化 | **20s** | **~1000×** | frustum 3DGUT + rays 向量化 + sorted npz |

- **Q：为什么用 3DGUT 而不是经典 3DGS？**
  A：3DGUT 支持 per-ray 输入，能直接吃 equirectangular 射线；经典 3DGS 必须先投影到 2D 平面（cube map 拼接有接缝）。

- **Q：为什么不用 pinhole 相机模型做全景？**
  A：pinhole 有 FOV 限制，会把 360° 中非前方的 GS 剔除（"相机背面"误判）；全景图不存在"背面"，每个 GS 总落在某个像素上。球面投影方案用等距圆柱坐标做 tile 分配，完美匹配 360° 全景。

- **Q：Tile 分配中怎么判断某个 GS 属于某个 tile？**
  A：GS → 球面投影 `(u,v)` + 像素半径 `(r_u,r_v)` → bbox 与 tile 区域相交即分配。详见 §7「球面投影算法」。

- **Q：每条射线实际渲染多少个高斯？**
  A：~1-50 个。kernel 遍历 tile 内所有高斯，但 `d² > 9` 的被跳过（`response ≈ 0`），`hit_t < 0` 的被跳过（射线背后）。实际参与 α-blending 的只有 d² 最小的几个。

- **Q：为什么新建 `gs2pano` env 不复用 `hyworld2`？**
  A：`hyworld2` 是别的项目的 env，混装风险大；新建干净 env 调试对照清晰。

- **Q：COLMAP 位姿怎么加载？**
  A：`images.bin` 中 `q, t` 为标准 w2c：`X_cam = R(q)·X_world + t`；相机位置 `pos = -R^T·t`。

- **Q：为什么必须传相机位姿？**
  A：保证前向/位置明确无歧义；360° 全景没有 FOV 概念，位姿决定全景图的朝向（中心像素 = 相机前向）。

- **Q：配对数据 (`--paired-data`) 的 frustum 和 ray 模式有什么区别？**
  A：frustum 模式用 GS 像素 bbox 覆盖判断（每个像素的视锥体内所有 GS 都记录），数据量大（64×32 就有 49M 对）。ray 模式用 `hit_t > 0` + `d² < 9` 过滤（和渲染 kernel 一致），紧凑（同分辨率仅 8K 对），适合训练。

- **Q：为什么 64×32 的 ray 配对数据只有 3213 个独特 GS？**
  A：低分辨率射线很粗（5.6° 张角），大部分小 GS 碰不到任何射线。提高分辨率到 1024×512（0.35° 张角）可大幅增加命中数。配对数据的唯一 GS 数 = 能被至少一条射线"看到"的 GS 数。

---

## 7. 球面投影算法

### 7.1 GS → 球面坐标

```
D     = ‖μ - cam_pos‖                          距离
θ     = atan2(X - cam_x, Z - cam_z)            方位角 [-π, π]
φ     = arcsin((Y - cam_y) / D)                 仰角 [-π/2, π/2]
```

### 7.2 球面角 → 全景图像素

```
u = (θ + π) / (2π) · W           ∈ [0, W]
v = (φ + π/2) / π · H            ∈ [0, H]
```

### 7.3 角张量（像素半径）

```
σ_max = max(s_x, s_y, s_z)                      最大物理尺度
α     = asin(3·σ_max / D), 若 3·σ_max ≥ D 则 α=π  保守角半径
r_u   = ⌈α / (2π) · W · 1/cos(φ)⌉ + 1            高纬补偿后的横向像素半径
r_v   = ⌈α / π · H⌉ + 1
```

### 7.4 Tile 分配条件

GS 属于 tile `(t_u, t_v)` ⟺ 像素 bbox `[u-r_u, u+r_u] × [v-r_v, v+r_v]` 与 tile 区域 `[t_u·16, (t_u+1)·16) × [t_v·16, (t_v+1)·16)` 相交。

若 bbox 跨越 `u=0/W` 接缝，`_build_tile_inputs()` 会为 tile 分配生成 `u+W` 或 `u-W` 临时副本；`flatten_ids` 随后映射回原始 GS id，避免渲染阶段读取错误高斯。

### 7.5 Per-ray 评估 (CUDA kernel)

```
① gro = S⁻¹R^T(ray_o - μ), grd = normalize(S⁻¹R^T·ray_d)
② hit_t = grd·(-gro); 若 < 0 → skip
③ d² = ‖grd × gro‖²
④ alpha = min(0.99, opac·exp(-d²/2))
⑤ 若 alpha < 1/255 或 d² > 9 → skip
⑥ front-to-back α-blending
```

---

**配对数据性能状态**（2026-06-11）:

| 分辨率 | 像素 | 配对总数 | 可见 GS | 被遮挡 GS | 唯一 GS | 渲染耗时 | 总耗时 |
|--------|------|---------|---------|----------|---------|---------|--------|
| 64×32 | 2,048 | **116K** | 67K | 48K (42%) | 78K (2.5%) | 0.17s | ~1s |
| **1024×512** | 524,288 | **27.5M** | 15.9M | 11.7M (42%) | 1.65M (53%) | **0.1s** | **55s** |

**核心方案**: CUDA kernel 内联配对数据输出（最终成功）, 通过 `__device__` 全局指针 + ctypes setter, **零函数签名修改**。原 Python numba 方案预估 1024×512 ~6h → CUDA **55s** (~400× 加速)。
