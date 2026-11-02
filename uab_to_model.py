#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
uab_to_model.py —— 从 Unity .uab 重建带「骨骼 + 蒙皮权重 + BlendShape + 动画」的角色模型

输入：包含 Mesh / AnimationClip（可选 Avatar）的若干 .uab（文件或目录均可）
输出：<角色名>.glb（必出）、skeleton.json、names_template.json；加 --fbx 且有 Blender 时再转出 .fbx

为什么需要“重建”而不是“直接提取 fbx”
  Unity 打包后 .fbx 已被拆成 Mesh / AnimationClip 等运行时资源，包内不存在 fbx 文件
  （容器路径里的 .fbx 只是源资源路径）。本脚本把这些运行时数据重新组装成标准模型。

还原了什么
  * 网格：顶点 / 法线 / UV / 子网格(按材质拆分) / LOD 选择
  * 蒙皮：每顶点 4 骨骼权重 + 骨骼索引 + BindPose（来自 Mesh 自身）
  * BlendShape：形变目标及其动画权重
  * 动画：解码 Streamed / Dense / Constant / Quantized 四种曲线
  * 骨骼层级：包内若无 Prefab/Avatar，则用 BindPose 与动画局部位移交叉验证反推父子关系；
              有 Avatar(TOS) 时自动取得真实骨骼名

依赖：pip install UnityPy numpy        （转 FBX 需要 Blender，可用 --blender 指定路径）

示例
  python uab_to_model.py ./bundles --character Parmon_10583 -o ./out
  python uab_to_model.py ./bundles --character Parmon_10583 -o ./out --clips "Idle|Walk|Run" --fbx
  python uab_to_model.py ./bundles --character Parmon_10583 -o ./out --no-anim --lod 1
  python uab_to_model.py ./bundles --character Parmon_10583 -o ./out --names my_names.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import zlib
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

try:
    import UnityPy
    from UnityPy.helpers.MeshHelper import MeshHandler
except ImportError:  # pragma: no cover
    sys.exit("缺少依赖，请先执行: pip install UnityPy numpy")

MAGICS = (b"UnityFS", b"UnityWeb", b"UnityRaw", b"UnityArchive")
S_MIRROR = np.diag([-1.0, 1.0, 1.0, 1.0])          # Unity(左手) -> glTF(右手)：镜像 X


# =========================================================================== #
# 四元数 / 矩阵工具（四元数顺序 x,y,z,w）
# =========================================================================== #
def qnorm(q):
    return q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-12)


def qmul(a, b):
    ax, ay, az, aw = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bx, by, bz, bw = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([aw * bx + bw * ax + ay * bz - az * by,
                     aw * by + bw * ay + az * bx - ax * bz,
                     aw * bz + bw * az + ax * by - ay * bx,
                     aw * bw - ax * bx - ay * by - az * bz], -1)


def qangle(a, b):
    d = np.clip(np.abs(np.sum(a * b, axis=-1)), 0.0, 1.0)
    return np.degrees(2.0 * np.arccos(d))


def quat_to_mat(q):
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    R = np.empty(q.shape[:-1] + (3, 3), q.dtype)
    R[..., 0, 0] = 1 - 2 * (y * y + z * z); R[..., 0, 1] = 2 * (x * y - z * w); R[..., 0, 2] = 2 * (x * z + y * w)
    R[..., 1, 0] = 2 * (x * y + z * w); R[..., 1, 1] = 1 - 2 * (x * x + z * z); R[..., 1, 2] = 2 * (y * z - x * w)
    R[..., 2, 0] = 2 * (x * z - y * w); R[..., 2, 1] = 2 * (y * z + x * w); R[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def mat_to_quat(R):
    R = np.asarray(R, np.float64)
    t = R[0, 0] + R[1, 1] + R[2, 2]
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        q = [(R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s, 0.25 * s]
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = math.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k]) * 2
        q = [0.0, 0.0, 0.0, 0.0]
        q[i] = 0.25 * s
        q[j] = (R[j, i] + R[i, j]) / s
        q[k] = (R[k, i] + R[i, k]) / s
        q[3] = (R[k, j] - R[j, k]) / s
    q = np.array(q)
    return q / np.linalg.norm(q)


def fix_continuity(q):
    """沿时间轴让相邻四元数同半球，避免插值走远路。"""
    q = q.copy()
    for i in range(1, len(q)):
        if np.dot(q[i], q[i - 1]) < 0:
            q[i] = -q[i]
    return q


def m4(d):
    return np.array([[d[f"e{r}{c}"] for c in range(4)] for r in range(4)], np.float64)


def to_gl_vec(v):          # 位置/法线/位移
    v = np.asarray(v).copy()
    v[..., 0] *= -1
    return v


def to_gl_quat(q):         # 旋转：镜像 X => (x,-y,-z,w)
    q = np.asarray(q).copy()
    q[..., 1] *= -1
    q[..., 2] *= -1
    return q


# =========================================================================== #
# AnimationClip 曲线解码：Streamed / Dense / Constant / Quantized
# =========================================================================== #
CURVE_N = {1: 3, 2: 4, 3: 3, 4: 3}      # Transform：位置 / 旋转(四元数) / 缩放 / 欧拉


def _binding_curve_count(b):
    return CURVE_N.get(b["attribute"], 1) if b["typeID"] == 4 else 1


def _decode_streamed(sc, times):
    n = sc["curveCount"]
    out = np.zeros((n, len(times)), np.float32)
    if n == 0 or not sc["data"]:
        return out
    raw = struct.pack("<%dI" % len(sc["data"]), *sc["data"])
    keys = [[] for _ in range(n)]
    pos = 0
    while pos + 8 <= len(raw):
        tm, nk = struct.unpack_from("<fI", raw, pos)
        pos += 8
        for _ in range(nk):
            i, a, b, c, d = struct.unpack_from("<Iffff", raw, pos)
            pos += 20
            if i < n and (np.isfinite(tm) or tm < -1e30):
                keys[i].append((tm, a, b, c, d))
    for ci in range(n):
        ks = keys[ci]
        if not ks:
            continue
        kt = np.array([k[0] for k in ks], np.float64)
        co = np.array([k[1:] for k in ks], np.float64)
        idx = np.clip(np.searchsorted(kt, times, side="right") - 1, 0, len(ks) - 1)
        dt = times - kt[idx]
        dt[kt[idx] < -1e30] = 0.0
        dt = np.maximum(dt, 0.0)
        a, b, c, d = co[idx].T
        with np.errstate(all="ignore"):
            v = ((a * dt + b) * dt + c) * dt + d
        bad = ~np.isfinite(v)
        v[bad] = d[bad]                                   # 阶梯(斜率 inf)：取起始值
        out[ci] = v
    return out


def _decode_dense(dc, times):
    n = dc["m_CurveCount"]
    out = np.zeros((n, len(times)), np.float32)
    if n == 0:
        return out
    fc = dc["m_FrameCount"]
    arr = np.asarray(dc["m_SampleArray"], np.float32).reshape(fc, n)
    f = np.clip((times - dc["m_BeginTime"]) * dc["m_SampleRate"], 0, fc - 1)
    i0 = np.floor(f).astype(int)
    i1 = np.minimum(i0 + 1, fc - 1)
    w = (f - i0).astype(np.float32)
    for ci in range(n):
        out[ci] = arr[i0, ci] * (1 - w) + arr[i1, ci] * w
    return out


def _decode_quantized(qc, times):
    """布局：类型表(n×1B) | 索引表(n×4B) | 静态值(NumStatic×float) | 动态块(NumDynamic×FrameSize)；
    每个动态块 = float start + float step + FrameCount×uint16，值 = start + q*step"""
    n = qc["m_CurveCount"]
    out = np.zeros((n, len(times)), np.float32)
    if n == 0:
        return out
    buf = bytes(bytearray(qc["m_Data"]))
    types = np.frombuffer(buf, np.uint8, n, qc["m_TypeOffset"])
    idx = np.frombuffer(buf, "<u4", n, qc["m_IndicesOffset"])
    static = np.frombuffer(buf, "<f4", qc["m_NumStatic"], qc["m_StaticOffset"])
    fs, fc = qc["m_FrameSize"], qc["m_FrameCount"]
    f = np.clip((times - qc["m_BeginTime"]) * qc["m_SampleRate"], 0, fc - 1)
    i0 = np.floor(f).astype(int)
    i1 = np.minimum(i0 + 1, fc - 1)
    w = (f - i0).astype(np.float32)
    for ci in range(n):
        if types[ci] == 0:
            out[ci] = static[idx[ci]]
        else:
            base = qc["m_DynamicOffset"] + int(idx[ci]) * fs
            start, step = struct.unpack_from("<ff", buf, base)
            q = np.frombuffer(buf, "<u2", fc, base + 8).astype(np.float32)
            vals = start + q * step
            out[ci] = vals[i0] * (1 - w) + vals[i1] * w
    return out


def decode_clip(tree, times=None, fps=30.0):
    """
    返回 (times, tracks, blend, stop)
      tracks: {路径哈希: {'pos':(N,3)|None,'rot':(N,4)|None,'scl':(N,3)|None}}   (Unity 空间、局部坐标)
      blend : {属性哈希: (N,)}   SkinnedMeshRenderer 上的 BlendShape 权重曲线（0~100）
    """
    mc = tree["m_MuscleClip"]
    clip = mc["m_Clip"]["data"]
    stop, start = float(mc["m_StopTime"]), float(mc["m_StartTime"])
    if times is None:
        times = np.unique(np.append(start + np.arange(0, max(stop - start, 0) + 1e-9, 1.0 / fps), stop))
    times = np.asarray(times, np.float64)
    if clip.get("m_PredictClip", {}).get("m_CurveCount", 0):
        raise NotImplementedError("该剪辑使用 PredictClip 压缩，暂不支持")
    allv = np.concatenate([
        _decode_streamed(clip["m_StreamedClip"], times),
        _decode_dense(clip["m_DenseClip"], times),
        np.tile(np.asarray(clip["m_ConstantClip"]["data"], np.float32)[:, None], (1, len(times))),
        _decode_quantized(clip["m_QuantizedClip"], times),
    ], axis=0)
    bindings = tree["m_ClipBindingConstant"]["genericBindings"]
    need = sum(_binding_curve_count(b) for b in bindings)
    if need != allv.shape[0]:
        raise ValueError(f"曲线数不一致: 绑定需要 {need}, 实际 {allv.shape[0]}")
    tracks, blend, ci = {}, {}, 0
    for b in bindings:
        k = _binding_curve_count(b)
        if b["typeID"] == 4 and b["attribute"] in (1, 2, 3):
            ent = tracks.setdefault(b["path"], {"pos": None, "rot": None, "scl": None})
            ent[{1: "pos", 2: "rot", 3: "scl"}[b["attribute"]]] = allv[ci:ci + k].T.astype(np.float64)
        elif b["typeID"] == 137:
            blend[b["attribute"]] = allv[ci].astype(np.float64)
        ci += k
    return times, tracks, blend, stop


# =========================================================================== #
# 扫描 bundle、收集 Mesh / AnimationClip / Avatar
# =========================================================================== #
def sniff(path):
    try:
        with open(path, "rb") as f:
            return any(f.read(12).startswith(m) for m in MAGICS)
    except OSError:
        return False


def list_bundles(inputs):
    out = []
    for inp in inputs:
        p = Path(inp)
        if p.is_file():
            out.append(str(p))
        elif p.is_dir():
            for dp, _, fns in os.walk(p):
                for fn in fns:
                    full = os.path.join(dp, fn)
                    if fn.lower().endswith((".uab", ".ab", ".bundle", ".assetbundle", ".unity3d")) or sniff(full):
                        out.append(full)
    return sorted(set(out))


def container_paths(env):
    res = {}
    for obj in env.objects:
        if obj.type.name != "AssetBundle":
            continue
        try:
            ab = obj.read()
            fname = getattr(obj.assets_file, "name", "")
            for cp, info in ab.m_Container:
                if getattr(info.asset, "file_id", 0) == 0:
                    res.setdefault((fname, info.asset.path_id), cp)
        except Exception:
            pass
    return res


class Found:
    def __init__(self):
        self.meshes, self.clips, self.tos = [], [], {}


def scan(files, char_re, log):
    found = Found()
    for f in files:
        try:
            env = UnityPy.load(f)
            objs = list(env.objects)
        except Exception as e:
            log(f"  [跳过] {os.path.basename(f)}: {type(e).__name__}: {e}")
            continue
        cmap = container_paths(env)
        for obj in objs:
            tn = obj.type.name
            if tn not in ("Mesh", "AnimationClip", "Avatar"):
                continue
            cp = cmap.get((getattr(obj.assets_file, "name", ""), obj.path_id), "")
            if tn == "Avatar":
                try:                                    # TOS: 路径哈希 -> 路径字符串，可直接还原骨骼名
                    for h, path in obj.read_typetree().get("m_TOS", []):
                        found.tos[int(h)] = str(path)
                except Exception:
                    pass
                continue
            name = obj.peek_name() or ""
            if char_re.search(cp) or char_re.search(name):
                (found.meshes if tn == "Mesh" else found.clips).append(
                    {"obj": obj, "name": name, "container": cp, "bundle": os.path.basename(f)})
    return found


# =========================================================================== #
# Mesh 读取
# =========================================================================== #
MESH_NAME_RE = re.compile(r"^(?P<base>.+?)(?:_LOD(?P<lod>\d+))?_(?P<hash>-?\d+)$")


def split_mesh_name(name, char):
    """M_Parmon_10583_Body01_LOD1_623013025 -> (part='Body01', lod=1)"""
    m = MESH_NAME_RE.match(name)
    base, lod = (m.group("base"), int(m.group("lod") or 0)) if m else (name, 0)
    prefix = f"M_{char}_"
    part = base[len(prefix):] if base.startswith(prefix) else re.sub(r"^M_", "", base)
    return part, lod


def _pad_skin(bw, bi, nv):
    """每顶点影响骨骼数可能是 1~4（通道维度不固定），统一补齐为 4 列。"""
    if bi is None:
        return None, None
    if bw is None:                          # 刚性绑定：只有骨骼索引(如眼球)，每个顶点跟随一块骨骼，权重隐含为 1
        i = np.zeros((nv, 4), np.int64)
        i[:, 0] = np.asarray(bi, np.int64).reshape(nv, -1)[:, 0]
        w = np.zeros((nv, 4))
        w[:, 0] = 1.0
        return w, i
    bw = np.asarray(bw, np.float64).reshape(nv, -1)
    bi = np.asarray(bi, np.int64).reshape(nv, -1)
    k = min(bw.shape[1], bi.shape[1], 4)
    w = np.zeros((nv, 4))
    i = np.zeros((nv, 4), np.int64)
    w[:, :k], i[:, :k] = bw[:, :k], bi[:, :k]
    zero = w.sum(1) <= 1e-9                 # 没有权重的顶点：全部交给第一个骨骼，避免被丢弃
    w[zero, 0] = 1.0
    return w, i


def read_part(entry, part, lod):
    obj = entry["obj"]
    t = obj.read_typetree()
    h = MeshHandler(obj.read())
    h.process()
    P = {"name": part, "mesh_name": entry["name"], "lod": lod}
    P["pos"] = np.asarray(h.m_Vertices, np.float64).reshape(-1, 3)
    nv = len(P["pos"])
    P["nrm"] = np.asarray(h.m_Normals, np.float64).reshape(nv, -1)[:, :3] if h.m_Normals is not None else None
    P["uv"] = np.asarray(h.m_UV0, np.float64).reshape(nv, -1)[:, :2] if h.m_UV0 is not None else None
    P["tris"] = [np.asarray(s, np.uint32).reshape(-1, 3) for s in h.get_triangles()]
    P["bw"], P["bi"] = _pad_skin(h.m_BoneWeights, h.m_BoneIndices, nv)
    P["hashes"] = [int(x) for x in t.get("m_BoneNameHashes", [])]
    P["bindpose"] = [m4(m) for m in t.get("m_BindPose", [])]
    P["root_hash"] = int(t.get("m_RootBoneNameHash", 0))
    # ---- BlendShape ----
    P["morphs"] = []
    sh = t.get("m_Shapes") or {}
    if sh.get("vertices"):
        verts, shapes = sh["vertices"], sh["shapes"]
        for ch in sh["channels"]:
            si = ch["frameIndex"] + ch["frameCount"] - 1          # 取最后一帧(满权重)
            s = shapes[si]
            d = np.zeros((nv, 3), np.float64)
            for v in verts[s["firstVertex"]: s["firstVertex"] + s["vertexCount"]]:
                d[v["index"]] = (v["vertex"]["x"], v["vertex"]["y"], v["vertex"]["z"])
            nm = ch["name"].split(".", 1)[1] if ch["name"].startswith("blendShape.") else ch["name"]
            P["morphs"].append({"name": nm, "hash": int(ch["nameHash"]), "delta": d})
    return P


def load_platform_materials(paths):
    """从 PlatformData JSON 读取各 Mesh 部件对应的材质名（可选）。"""
    res = {}
    files = []
    for p in paths or []:
        p = Path(p)
        files += [str(x) for x in p.rglob("*PlatformData*.json")] if p.is_dir() else [str(p)]
    for f in files:
        try:
            d = json.load(open(f, encoding="utf-8"))
            r = d["m_DefaultConfig"]["m_Lods"][0]["m_Renderers"][0]
            mesh = Path(r["m_MeshPath"]["m_Path"]).stem
            mats = [Path(m["m_Path"]).stem for m in r["m_MaterialList"]]
            res[mesh] = mats
        except Exception:
            continue
    return res


# =========================================================================== #
# 骨骼层级推断
# =========================================================================== #
def infer_parents(H, W, samples, tol=1e-3):
    """
    H: 骨骼哈希列表；W: (n,4,4) 模型空间世界矩阵
    samples: [{hash:(pos(3)|None, rot(4)|None)}...] 每个 clip 的首帧局部位置/旋转
    原理：刚体骨骼的局部位移为常量，且等于 inv(W_父) @ W_子 的平移；
         对每块骨骼跨所有 clip 投票选出使两者吻合的候选父骨骼，并用旋转一致性消除并列。
    """
    n = len(H)
    Winv = np.linalg.inv(W)
    T = np.einsum("jab,ibc->jiac", Winv, W)[..., :3, 3]          # T[j,i] = (Winv[j] @ W[i]) 的平移
    votes = np.zeros((n, n), np.int64)
    seen = np.zeros(n, np.int64)
    for smp in samples:
        P = np.full((n, 3), np.nan)
        for i, h in enumerate(H):
            v = smp.get(h)
            if v is not None and v[0] is not None:
                P[i] = v[0]
        mask = ~np.isnan(P[:, 0])
        seen += mask
        d = np.linalg.norm(T - P[None, :, :], axis=2)            # (j, i)
        hit = (d < tol) & mask[None, :]
        hit[np.arange(n), np.arange(n)] = False
        votes += hit.T.astype(np.int64)
    parent = np.full(n, -1, np.int64)
    conf = np.zeros(n)
    ties = 0
    for i in range(n):
        mx = votes[i].max()
        if mx <= 0:
            continue
        cand = [j for j in range(n) if votes[i, j] >= 0.9 * mx]
        if len(cand) > 1:                                          # 位置并列 -> 用旋转一致性裁决
            ties += 1
            best, bj = 1e9, cand[0]
            for j in cand:
                qr = mat_to_quat((Winv[j] @ W[i])[:3, :3])
                errs = [qangle(s[H[i]][1], qr) for s in samples[::6]
                        if H[i] in s and s[H[i]][1] is not None]
                e = float(np.median(errs)) if errs else 1e9
                if e < best:
                    best, bj = e, j
            parent[i] = bj
        else:
            parent[i] = cand[0]
        conf[i] = votes[i, parent[i]] / max(seen[i], 1)
    # 环检测：出现环则断开置信度最低的边
    for i in range(n):
        seen_chain, k = [], i
        while k >= 0 and k not in seen_chain:
            seen_chain.append(k)
            k = int(parent[k])
        if k >= 0:
            cyc = seen_chain[seen_chain.index(k):]
            cut = min(cyc, key=lambda x: conf[x])
            parent[cut] = -1
    return parent, conf, ties


# =========================================================================== #
# glTF/GLB 写出
# =========================================================================== #
class Gltf:
    def __init__(self):
        self.bin = bytearray()
        self.views, self.accs = [], []

    def _pad(self):
        while len(self.bin) % 4:
            self.bin.append(0)

    def add(self, arr, target=None, minmax=False):
        arr = np.ascontiguousarray(arr)
        if arr.dtype == np.float64:
            arr = arr.astype(np.float32)
        ctype = {np.dtype("float32"): 5126, np.dtype("uint16"): 5123, np.dtype("uint32"): 5125,
                 np.dtype("uint8"): 5121}[arr.dtype]
        if arr.ndim == 1:
            typ, n = "SCALAR", len(arr)
        elif arr.ndim == 2:
            typ, n = {2: "VEC2", 3: "VEC3", 4: "VEC4"}[arr.shape[1]], len(arr)
        else:                                                   # (N,4,4) 矩阵，glTF 为列主序
            typ, n = "MAT4", len(arr)
            arr = np.ascontiguousarray(arr.transpose(0, 2, 1))
        self._pad()
        view = {"buffer": 0, "byteOffset": len(self.bin), "byteLength": arr.nbytes}
        if target:
            view["target"] = target
        self.bin += arr.tobytes()
        self.views.append(view)
        acc = {"bufferView": len(self.views) - 1, "componentType": ctype, "count": n, "type": typ}
        if minmax:
            flat = arr.reshape(n, -1)
            acc["min"] = [float(x) for x in flat.min(0)]
            acc["max"] = [float(x) for x in flat.max(0)]
        self.accs.append(acc)
        return len(self.accs) - 1


def write_glb(path, doc, gl):
    gl._pad()
    doc["bufferViews"], doc["accessors"] = gl.views, gl.accs
    doc["buffers"] = [{"byteLength": len(gl.bin)}]
    js = json.dumps(doc, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    js += b" " * ((4 - len(js) % 4) % 4)
    total = 12 + 8 + len(js) + 8 + len(gl.bin)
    with open(path, "wb") as f:
        f.write(struct.pack("<III", 0x46546C67, 2, total))
        f.write(struct.pack("<II", len(js), 0x4E4F534A) + js)
        f.write(struct.pack("<II", len(gl.bin), 0x004E4942) + bytes(gl.bin))


# =========================================================================== #
# 主流程：组装骨架 + 网格 + 动画
# =========================================================================== #
def safe_name(s):
    return re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", str(s)).strip() or "_"


def build(args, log):
    char_re = re.compile(re.escape(args.character) + r"(?!\d)", re.I)
    files = list_bundles(args.inputs)
    log(f"[1/6] 扫描 {len(files)} 个 bundle …")
    found = scan(files, char_re, log)
    log(f"      Mesh {len(found.meshes)} 个 | AnimationClip {len(found.clips)} 个 | Avatar(TOS) 骨骼名 {len(found.tos)} 条")
    if not found.meshes:
        sys.exit("[!] 没有找到该角色的 Mesh，请确认 --character 与包含 Mesh 的 .uab 是否在输入里")

    # ---- 选择 LOD、按部件读取 ----
    chosen = {}
    for e in found.meshes:
        part, lod = split_mesh_name(e["name"], args.character)
        if lod == args.lod:
            chosen[part] = (e, lod)
    if args.parts:
        keep = {p.strip() for p in args.parts.split(",")}
        chosen = {k: v for k, v in chosen.items() if k in keep}
    if not chosen:
        avail = sorted({split_mesh_name(e['name'], args.character) for e in found.meshes})
        sys.exit(f"[!] LOD{args.lod} 下没有可用部件。可用 (部件, LOD): {avail}")
    log(f"[2/6] 读取部件(LOD{args.lod}): {', '.join(sorted(chosen))}")
    parts = [read_part(e, p, lod) for p, (e, lod) in sorted(chosen.items(), key=lambda kv: -len(kv[1][0]['name']))]
    parts.sort(key=lambda P: -len(P["hashes"]))                 # 骨骼最多的部件(Body)作为骨架基准
    for P in parts:
        log(f"      {P['name']:8s} 顶点 {len(P['pos']):6d} 子网格 {len(P['tris'])} 骨骼 {len(P['hashes']):3d} "
            f"BlendShape {len(P['morphs'])} 权重 {'有' if P['bw'] is not None else '无'}")

    # ---- 骨骼并集 ----
    H, Wd = [], {}
    for P in parts:
        for h, bp in zip(P["hashes"], P["bindpose"]):
            if h not in Wd:
                Wd[h] = np.linalg.inv(bp)
                H.append(h)
    n = len(H)
    hidx = {h: i for i, h in enumerate(H)}
    W = np.stack([Wd[h] for h in H])
    # 不同部件的网格空间可能有微小平移，统计一下
    base = parts[0]
    for P in parts[1:]:
        diffs = [np.linalg.inv(bp)[:3, 3] - Wd[h][:3, 3] for h, bp in zip(P["hashes"], P["bindpose"]) if h in Wd]
        if diffs:
            d = np.array(diffs)
            log(f"      部件 {P['name']} 相对 {base['name']} 的骨骼平移差: 均值 {d.mean(0).round(5).tolist()} 最大 {np.abs(d).max():.5f}")

    # ---- 解码所有可用 clip 的首帧，用于推断层级 ----
    log(f"[3/6] 解码动画并推断骨骼层级（{n} 块骨骼）…")
    hashset = set(H)
    clip_trees, samples, skipped = [], [], 0
    for c in found.clips:
        try:
            t = c["obj"].read_typetree()
            binds = {b["path"] for b in t["m_ClipBindingConstant"]["genericBindings"] if b["typeID"] == 4}
            hit = len(binds & hashset)
            if hit < 10 or hit < 0.8 * len(binds):                   # 轨道大多不属于这副骨架 -> 其它角色的动画
                skipped += 1
                log(f"      [跳过 clip] {c['name']}: 仅 {hit}/{len(binds)} 条轨道属于该骨架")
                continue
            tm, tr, bl, stop = decode_clip(t, times=[float(t["m_MuscleClip"]["m_StartTime"])])
            samples.append({h: (v["pos"][0] if v["pos"] is not None else None,
                                v["rot"][0] if v["rot"] is not None else None) for h, v in tr.items()})
            clip_trees.append((c["name"], t))
        except Exception as e:
            log(f"      [跳过 clip] {c['name']}: {type(e).__name__}: {e}")
    log(f"      可用动画 {len(clip_trees)} 个（骨架不匹配而跳过 {skipped} 个）")
    if samples:
        parent, conf, ties = infer_parents(H, W, samples)
    else:
        parent, conf, ties = np.full(n, -1), np.zeros(n), 0
        log("      [警告] 没有可用动画，无法推断层级：所有骨骼将挂在根下（只能用于静态姿势）")
    roots = [i for i in range(n) if parent[i] < 0]
    log(f"      层级: 已确定 {n - len(roots)} 条父子关系 | 根骨骼 {len(roots)} 个 | 位置并列经旋转裁决 {ties} 处 | "
        f"低置信度(<50%) {int(((conf > 0) & (conf < 0.5)).sum())} 块")

    for r in roots:
        sub, st = 0, [r]
        while st:
            k = st.pop(); sub += 1
            st += [j for j in range(n) if parent[j] == k]
        log(f"      根骨骼 #{r:3d} hash={H[r]} 世界位置={np.round(W[r][:3, 3], 3).tolist()} 子树 {sub} 块")
    # ---- 骨骼名 ----
    hash2path = dict(found.tos)
    if args.names:
        nm = json.load(open(args.names, encoding="utf-8"))
        if isinstance(nm, list):
            for p in nm:
                parts_ = p.strip("/").split("/")
                for k in range(1, len(parts_) + 1):
                    pp = "/".join(parts_[:k])
                    hash2path[zlib.crc32(pp.encode("utf-8"))] = pp
        else:
            for k, v in nm.items():
                if v:
                    hash2path[int(k)] = str(v)
    names, used = [], set()
    for i, h in enumerate(H):
        base_n = hash2path[h].split("/")[-1] if h in hash2path else f"Bone_{h}"
        nm_ = safe_name(base_n)
        if nm_ in used:
            nm_ = f"{nm_}_{h}"
        used.add(nm_)
        names.append(nm_)
    named = sum(1 for h in H if h in hash2path)
    log(f"      骨骼名: 已还原 {named}/{n}" + ("" if named else "（包内无 Avatar/Prefab，使用 Bone_<哈希> 命名；可用 --names 提供）"))

    # ---- 根骨骼的“虚拟父节点”：包内没有其祖先链，用最贴近静止姿态的 clip 首帧反推 ----
    rest_loc = []         # 每块骨骼静止局部 (pos, quat)，Unity 空间
    for i in range(n):
        L = W[i] if parent[i] < 0 else np.linalg.inv(W[parent[i]]) @ W[i]
        rest_loc.append((L[:3, 3].copy(), mat_to_quat(L[:3, :3])))
    WP = {r: np.eye(4) for r in roots}
    if samples and roots:
        best, bk = 1e9, 0
        for k, smp in enumerate(samples):
            errs = [qangle(smp[H[i]][1], rest_loc[i][1]) for i in range(n)
                    if parent[i] >= 0 and H[i] in smp and smp[H[i]][1] is not None]
            if errs and np.median(errs) < best:
                best, bk = float(np.median(errs)), k
        log(f"      以动画「{clip_trees[bk][0]}」首帧作为静止姿态参考（与 BindPose 旋转中位偏差 {best:.2f}°）")
        for r in roots:
            v = samples[bk].get(H[r])
            if v is None or v[1] is None:
                continue
            p0 = v[0] if v[0] is not None else np.zeros(3)
            Lr = np.eye(4)
            Lr[:3, :3] = quat_to_mat(qnorm(np.asarray(v[1])))
            Lr[:3, 3] = p0
            WP[r] = W[r] @ np.linalg.inv(Lr)

    # ------------------------------------------------------------------ #
    # 构建 glTF
    # ------------------------------------------------------------------ #
    log("[4/6] 组装 glTF（骨骼 / 蒙皮 / BlendShape）…")
    gl = Gltf()
    nodes, children = [], defaultdict(list)
    for i in range(n):                                    # 节点 0..n-1 = 骨骼
        t_, q_ = rest_loc[i]
        nodes.append({"name": names[i], "translation": [float(x) for x in to_gl_vec(t_)],
                      "rotation": [float(x) for x in to_gl_quat(q_)]})
        if parent[i] >= 0:
            children[int(parent[i])].append(i)
    for i, ch in children.items():
        nodes[i]["children"] = ch
    top = {"name": "Armature", "children": roots}          # 名字尽量短：Blender/FBX 动作名上限 63 字符
    nodes.append(top)
    top_idx = len(nodes) - 1

    plat = load_platform_materials(args.platform)
    materials, meshes_out, skins, scene_nodes = [], [], [], [top_idx]
    mesh_node_of = {}
    for P in parts:
        nv = len(P["pos"])
        attrs = {"POSITION": gl.add(to_gl_vec(P["pos"]), 34962, minmax=True)}
        if P["nrm"] is not None:
            attrs["NORMAL"] = gl.add(to_gl_vec(P["nrm"]), 34962)
        if P["uv"] is not None:
            uv = P["uv"].copy()
            uv[:, 1] = 1.0 - uv[:, 1]                       # Unity V 向上 -> glTF V 向下
            attrs["TEXCOORD_0"] = gl.add(uv, 34962)
        skinned = P["bw"] is not None and P["bi"] is not None and len(P["hashes"]) > 0
        if skinned:
            attrs["JOINTS_0"] = gl.add(P["bi"].astype(np.uint16), 34962)
            attrs["WEIGHTS_0"] = gl.add(P["bw"], 34962)
        targets = []
        for mo in P["morphs"]:
            targets.append({"POSITION": gl.add(to_gl_vec(mo["delta"]), 34962, minmax=True)})
        mat_names = plat.get(P["mesh_name"]) or plat.get(re.sub(r"_LOD\d+", "", P["mesh_name"])) or []
        prims = []
        for k, tri in enumerate(P["tris"]):
            if len(tri) == 0:
                continue
            tri = tri[:, [0, 2, 1]]                         # 镜像后翻转绕序
            mname = mat_names[k] if k < len(mat_names) else f"{P['name']}_mat{k:02d}"
            materials.append({"name": mname, "pbrMetallicRoughness": {"baseColorFactor": [0.8, 0.8, 0.8, 1.0],
                                                                      "metallicFactor": 0.0, "roughnessFactor": 0.8},
                              "doubleSided": True})
            prim = {"attributes": attrs, "indices": gl.add(tri.reshape(-1).astype(np.uint32), 34963),
                    "material": len(materials) - 1}
            if targets:
                prim["targets"] = targets
            prims.append(prim)
        mesh = {"name": P["name"], "primitives": prims}
        if targets:
            mesh["weights"] = [0.0] * len(targets)
            mesh["extras"] = {"targetNames": [mo["name"] for mo in P["morphs"]]}
        meshes_out.append(mesh)
        node = {"name": P["name"], "mesh": len(meshes_out) - 1}
        if skinned:
            ibm = np.stack([S_MIRROR @ bp @ S_MIRROR for bp in P["bindpose"]])
            skins.append({"name": f"{P['name']}_Skin", "joints": [hidx[h] for h in P["hashes"]],
                          "inverseBindMatrices": gl.add(ibm), "skeleton": top_idx})
            node["skin"] = len(skins) - 1
        nodes.append(node)
        scene_nodes.append(len(nodes) - 1)
        mesh_node_of[P["name"]] = len(nodes) - 1

    doc = {"asset": {"version": "2.0", "generator": "uab_to_model.py"}, "scene": 0,
           "scenes": [{"name": args.character, "nodes": scene_nodes}], "nodes": nodes,
           "meshes": meshes_out, "materials": materials}
    if skins:
        doc["skins"] = skins

    # ------------------------------------------------------------------ #
    # 静止姿态自检：glTF 公式蒙皮后应与原网格一致（证明骨骼+BindPose+权重自洽）
    # ------------------------------------------------------------------ #
    Jw = np.zeros((n, 4, 4))
    for i in range(n):
        t_, q_ = to_gl_vec(rest_loc[i][0]), to_gl_quat(rest_loc[i][1])
        L = np.eye(4)
        L[:3, :3] = quat_to_mat(q_)
        L[:3, 3] = t_
        Jw[i] = L
    order, seen_ = [], set()

    def visit(i):
        if i in seen_:
            return
        if parent[i] >= 0:
            visit(int(parent[i]))
        seen_.add(i)
        order.append(i)
    for i in range(n):
        visit(i)
    for i in order:
        if parent[i] >= 0:
            Jw[i] = Jw[int(parent[i])] @ Jw[i]
    for P in parts:
        if P["bw"] is None:
            continue
        ibm = np.stack([S_MIRROR @ bp @ S_MIRROR for bp in P["bindpose"]])
        jm = np.stack([Jw[hidx[h]] @ ibm[k] for k, h in enumerate(P["hashes"])])        # (nb,4,4)
        v = np.c_[to_gl_vec(P["pos"]), np.ones(len(P["pos"]))]
        out = np.zeros((len(v), 3))
        for s in range(4):
            out += P["bw"][:, s:s + 1] * np.einsum("vab,vb->va", jm[P["bi"][:, s]], v)[:, :3]
        diff = out - to_gl_vec(P["pos"])
        off = np.median(diff, axis=0)                       # 部件网格空间相对骨架的整体平移（Unity 里同样存在）
        err = np.abs(diff - off).max()
        P["rest_err"] = float(err)
        note = f"，含部件整体平移 {np.abs(off).max() * 1000:.2f} mm" if np.abs(off).max() > 1e-4 else ""
        log(f"      自检 {P['name']:8s}: 静止姿态蒙皮重投影最大误差 {err:.2e} m{note}" + ("  ✓" if err < 1e-3 else "  ✗ 不一致!"))

    # ------------------------------------------------------------------ #
    # 动画
    # ------------------------------------------------------------------ #
    anims, anim_names = [], Counter()
    if not args.no_anim and clip_trees:
        pat = re.compile(args.clips) if args.clips else None
        sel = [(nm_, t) for nm_, t in clip_trees if pat is None or pat.search(nm_)]
        log(f"[5/6] 写出动画 {len(sel)}/{len(clip_trees)} 个（{args.fps:g} fps）…")
        blend_by_part = {P["name"]: {mo["hash"]: k for k, mo in enumerate(P["morphs"])} for P in parts if P["morphs"]}
        for ci, (cname, t) in enumerate(sel):
            try:
                tm, tr, bl, stop = decode_clip(t, fps=args.fps)
            except Exception as e:
                log(f"      [跳过 clip] {cname}: {type(e).__name__}: {e}")
                continue
            N = len(tm)
            tin = gl.add((tm - tm[0]).astype(np.float32), minmax=True)
            samplers, channels = [], []

            def emit(node, path, arr):
                samplers.append({"input": tin, "output": gl.add(arr), "interpolation": "LINEAR"})
                channels.append({"sampler": len(samplers) - 1, "target": {"node": node, "path": path}})
            for i, h in enumerate(H):
                v = tr.get(h)
                if v is None:
                    continue
                rp, rq = rest_loc[i]
                pos = v["pos"]
                rot = qnorm(v["rot"]) if v["rot"] is not None else None
                if rot is not None:
                    rot = fix_continuity(rot)
                if parent[i] < 0 and (pos is not None or rot is not None):      # 根骨骼：并入虚拟父节点
                    Mp = WP[i]
                    qP = mat_to_quat(Mp[:3, :3])
                    pp = pos if pos is not None else np.tile(rp if parent[i] >= 0 else np.zeros(3), (N, 1))
                    rr = rot if rot is not None else np.tile(np.array([0, 0, 0, 1.0]), (N, 1))
                    pos = pp @ Mp[:3, :3].T + Mp[:3, 3]
                    rot = fix_continuity(qnorm(qmul(np.tile(qP, (N, 1)), rr)))
                if pos is not None and np.abs(pos - rp).max() > 1e-4:
                    emit(i, "translation", to_gl_vec(pos).astype(np.float32))
                if rot is not None and qangle(rot, np.tile(rq, (N, 1))).max() > 0.1:
                    emit(i, "rotation", to_gl_quat(rot).astype(np.float32))
                if v["scl"] is not None and np.abs(v["scl"] - 1).max() > 1e-4:
                    emit(i, "scale", v["scl"].astype(np.float32))
            for pname, hm in blend_by_part.items():
                if any(h in bl for h in hm):
                    w = np.zeros((N, len(hm)), np.float32)
                    for h, k in hm.items():
                        if h in bl:
                            w[:, k] = bl[h] / 100.0
                    samplers.append({"input": tin, "output": gl.add(w.reshape(-1)), "interpolation": "LINEAR"})
                    channels.append({"sampler": len(samplers) - 1,
                                     "target": {"node": mesh_node_of[pname], "path": "weights"}})
            if not channels:
                continue
            short = re.sub(rf"^(?:Ani_)?{re.escape(args.character)}_", "", cname)       # 去掉 Ani_<角色>_ 前缀，动作名更短
            anim_names[short] += 1
            nm_ = short if anim_names[short] == 1 else f"{short}_{anim_names[short]}"
            anims.append({"name": nm_, "samplers": samplers, "channels": channels})
            if (ci + 1) % 20 == 0:
                log(f"      … {ci + 1}/{len(sel)}")
        if anims:
            doc["animations"] = anims
    else:
        log("[5/6] 跳过动画")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    glb = out_dir / f"{safe_name(args.character)}.glb"
    write_glb(glb, doc, gl)
    log(f"[6/6] 已写出 {glb}  ({glb.stat().st_size / 1e6:.2f} MB) | 骨骼 {n} | 部件 {len(parts)} | 动画 {len(anims)}")

    # ---- 附带信息 ----
    sk = [{"index": i, "hash": H[i], "name": names[i], "parent": int(parent[i]),
           "parent_hash": H[int(parent[i])] if parent[i] >= 0 else None, "confidence": round(float(conf[i]), 3),
           "world_pos_unity": [round(float(x), 5) for x in W[i][:3, 3]]} for i in range(n)]
    json.dump(sk, open(out_dir / "skeleton.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    json.dump({str(h): hash2path.get(h, "") for h in H}, open(out_dir / "names_template.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    return glb, n, len(anims)


# =========================================================================== #
# GLB -> FBX（借助 Blender，无界面运行）
# =========================================================================== #
BLENDER_SCRIPT = r'''
import bpy, sys
argv = sys.argv[sys.argv.index("--") + 1:]
src, dst = argv[0], argv[1]
fps = float(argv[2]) if len(argv) > 2 else 30.0
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.context.scene.render.fps = int(round(fps))          # 烘焙按场景帧率采样，默认 24 会让 30fps 动画被重采样
bpy.context.scene.render.fps_base = 1.0
bpy.ops.import_scene.gltf(filepath=src)
arm = [o for o in bpy.data.objects if o.type == 'ARMATURE']
# glTF 导入器会给骨骼生成自定义显示形状(Icosphere)，它不属于模型，删除以免被导出成多余网格
shapes = set()
for a in arm:
    for pb in a.pose.bones:
        if pb.custom_shape is not None:
            shapes.add(pb.custom_shape)
            pb.custom_shape = None
for o in shapes:
    bpy.data.objects.remove(o, do_unlink=True)
# 将每个 Action 保证带 fake user，导出时全部烘焙
for a in bpy.data.actions:
    a.use_fake_user = True
bpy.ops.export_scene.fbx(
    filepath=dst, use_selection=False, object_types={'ARMATURE', 'MESH'},
    add_leaf_bones=False, bake_anim=True, bake_anim_use_all_actions=True,
    bake_anim_use_nla_strips=False, bake_anim_force_startend_keying=True,
    bake_anim_simplify_factor=0.0, mesh_smooth_type='FACE',
    axis_forward='-Z', axis_up='Y', path_mode='COPY')
print("FBX_EXPORT_OK", dst, "armatures:", len(arm), "actions:", len(bpy.data.actions))
'''

BLENDER_VERIFY = r'''
import bpy, sys
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.fbx(filepath=sys.argv[sys.argv.index("--") + 1])
arms = [o for o in bpy.data.objects if o.type == 'ARMATURE']
meshes = [o for o in bpy.data.objects if o.type == 'MESH']
print("VERIFY armatures:", len(arms), "bones:", sum(len(a.data.bones) for a in arms))
for m in meshes:
    sk = m.data.shape_keys
    print("VERIFY mesh", m.name, "verts", len(m.data.vertices), "vgroups", len(m.vertex_groups),
          "shape_keys", (len(sk.key_blocks) - 1) if sk else 0)
arm_actions = [a for a in bpy.data.actions if not a.name.startswith("Key|")]
print("VERIFY armature actions:", len(arm_actions), "| shape-key actions:", len(bpy.data.actions) - len(arm_actions))
# 实际播放一个动画，取 Body 变形后的包围盒尺寸（Blender 为 Z 向上）
body = next((m for m in meshes if m.name == "Body"), None)
pick = next((a for a in arm_actions if "Walk" in a.name), arm_actions[0] if arm_actions else None)
if arms and body and pick:
    arm = arms[0]
    arm.animation_data_create().action = pick
    f0, f1 = int(pick.frame_range[0]), int(pick.frame_range[1])
    dg = bpy.context.evaluated_depsgraph_get()
    import mathutils
    dims = []
    for f in (f0, (f0 + f1) // 2, f1):
        bpy.context.scene.frame_set(f)
        dg = bpy.context.evaluated_depsgraph_get()
        me = body.evaluated_get(dg).to_mesh()
        xs = [body.matrix_world @ v.co for v in me.vertices]
        mn = [min(v[i] for v in xs) for i in range(3)]; mx = [max(v[i] for v in xs) for i in range(3)]
        dims.append([round(mx[i] - mn[i], 2) for i in range(3)])
    print("VERIFY play", pick.name[:40], "frames", f0, f1, "Body 包围盒尺寸(x,y,z):", dims)
'''

def find_blender(user_path):
    for c in ([user_path] if user_path else []) + [shutil.which("blender")]:
        if c and os.path.exists(c):
            return c
    return None


def to_fbx(glb, blender, log, fps=30.0, verify=True):
    fbx = glb.with_suffix(".fbx")
    with tempfile.TemporaryDirectory() as td:
        sp = Path(td) / "conv.py"
        sp.write_text(BLENDER_SCRIPT, encoding="utf-8")
        log(f"[FBX] 调用 Blender 转换: {blender}")
        r = subprocess.run([blender, "-b", "--factory-startup", "--python", str(sp), "--", str(glb), str(fbx), str(fps)],
                           capture_output=True, text=True)
        if "FBX_EXPORT_OK" not in r.stdout:
            log("[FBX] 转换失败:\n" + (r.stdout[-1500:] + r.stderr[-1500:]))
            return None
        log("[FBX] " + [l for l in r.stdout.splitlines() if "FBX_EXPORT_OK" in l][0])
        if verify:
            vp = Path(td) / "verify.py"
            vp.write_text(BLENDER_VERIFY, encoding="utf-8")
            r2 = subprocess.run([blender, "-b", "--factory-startup", "--python", str(vp), "--", str(fbx)],
                                capture_output=True, text=True)
            for l in r2.stdout.splitlines():
                if l.startswith("VERIFY"):
                    log("[FBX] " + l)
    log(f"[FBX] 已写出 {fbx} ({fbx.stat().st_size / 1e6:.2f} MB)")
    return fbx


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="从 Unity .uab 重建带骨骼/权重/动画的角色模型 (GLB/FBX)",
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__.split("示例", 1)[1])
    ap.add_argument("inputs", nargs="+", help=".uab 文件或目录（Mesh 包 + 动画包放一起即可）")
    ap.add_argument("-c", "--character", required=True, help="角色标识，如 Parmon_10583（匹配容器路径/资产名）")
    ap.add_argument("-o", "--output", default="./model_out", help="输出目录")
    ap.add_argument("--lod", type=int, default=0, help="使用的 LOD 级别（默认 0 最高精度）")
    ap.add_argument("--parts", help="只导出指定部件，逗号分隔，如 Body,Eye（默认全部）")
    ap.add_argument("--clips", help="只导出名称匹配该正则的动画，如 'Idle|Walk|Run'")
    ap.add_argument("--fps", type=float, default=30.0, help="动画采样帧率（默认 30）")
    ap.add_argument("--no-anim", action="store_true", help="不导出动画")
    ap.add_argument("--names", help="骨骼名映射 JSON：{哈希: 名称} 或 路径列表 ['Bip001/Bip001 Pelvis', ...]")
    ap.add_argument("--platform", nargs="*", help="PlatformData JSON（文件/目录），用于给材质命名")
    ap.add_argument("--fbx", action="store_true", help="同时转出 FBX（需要 Blender）")
    ap.add_argument("--blender", help="Blender 可执行文件路径（默认从 PATH 查找）")
    ap.add_argument("--no-verify", action="store_true", help="转出 FBX 后不回读校验")
    args = ap.parse_args()
    log = lambda s: print(s, flush=True)
    glb, nb, na = build(args, log)
    if args.fbx:
        bl = find_blender(args.blender)
        if not bl:
            log("[FBX] 没找到 Blender：请安装 Blender 后用 --blender 指定路径，或在 Blender 里 导入 glTF → 导出 FBX")
            return 2
        return 0 if to_fbx(glb, bl, log, fps=args.fps, verify=not args.no_verify) else 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
