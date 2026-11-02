#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
uab_extract.py —— 基于 UnityPy 的 Unity AssetBundle (.uab/.ab/...) 资产提取工具

功能
  * 单文件或整个目录（递归）批量提取，输出到指定目录，并保留相对目录结构
  * 优先使用包内 AssetBundle.m_Container 记录的原始资产路径来命名/建目录
  * 自动识别 "一个文件里首尾拼接了多个 UnityFS 包" 的情况并逐个提取
  * 按类型导出：
      Texture2D / Sprite -> .png        Mesh -> .obj（可选再附带 .mesh.json）
      TextAsset -> 原始内容             AudioClip -> .wav 等
      Font -> .ttf/.otf                 Shader -> .shader
      其余类型（MonoBehaviour / AnimationClip / Material / Avatar / ...）
      -> TypeTree JSON（字段名与 Unity 序列化一致）；解析失败时回退为原始 .bin
  * --find 在目录里按关键字查找资产在哪个包里（不写文件），--list 只统计不导出
  * 多进程（-j）、断点续跑（--skip-existing）、错误日志、每个包一份 _index.json

依赖：pip install UnityPy        （可选：pip install tqdm 显示进度条）

示例
  python uab_extract.py cdata.uab -o ./out
  python uab_extract.py ./Assetbundle -o ./out -j 4
  python uab_extract.py ./Assetbundle -o ./out --types Mesh,Texture2D,MonoBehaviour,Avatar
  python uab_extract.py ./Assetbundle --find Parmon_10583
"""
from __future__ import annotations

import argparse
import base64
import json
import mmap
import os
import re
import struct
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

try:
    import UnityPy
except ImportError:  # pragma: no cover
    sys.exit("缺少依赖，请先执行: pip install UnityPy")

try:
    from tqdm import tqdm
except ImportError:  # tqdm 是可选依赖
    tqdm = None

# --------------------------------------------------------------------------- #
# 常量
# --------------------------------------------------------------------------- #
MAGICS = (b"UnityFS", b"UnityWeb", b"UnityRaw", b"UnityArchive")
KNOWN_EXT = {".uab", ".ab", ".bundle", ".assetbundle", ".unity3d", ".u3d"}
# 嗅探文件头时直接跳过的常见非 bundle 扩展名（加快大目录扫描）
SKIP_SNIFF_EXT = {
    ".json", ".txt", ".md", ".png", ".jpg", ".jpeg", ".tga", ".bmp", ".dds", ".psd",
    ".py", ".lua", ".xml", ".csv", ".log", ".ini", ".cfg", ".dll", ".exe", ".so",
    ".mp3", ".wav", ".ogg", ".mp4", ".zip", ".7z", ".rar", ".apk", ".obb", ".bin",
    ".manifest", ".meta", ".bytes", ".xdf",
}
DEFAULT_TYPES = ("Texture2D,Sprite,Mesh,TextAsset,AudioClip,Font,Shader,"
                 "MonoBehaviour,AnimationClip,Material,Avatar")
TEXT_EXTS = {".txt", ".json", ".xml", ".lua", ".csv", ".tsv", ".html", ".yaml", ".yml",
             ".bytes", ".proto", ".md", ".ini", ".cfg", ".sql"}

_BAD_CHARS = re.compile(r'[<>:"|?*\x00-\x1f]')


# --------------------------------------------------------------------------- #
# 通用小工具
# --------------------------------------------------------------------------- #
def safe_part(s: str, maxlen: int = 120) -> str:
    """把单个名字清理成各系统都能用的文件/目录名。"""
    s = _BAD_CHARS.sub("_", str(s).replace("/", "_").replace("\\", "_")).strip(" .")
    return s[:maxlen] or "_"


def safe_rel_parts(path_str: str) -> list[str]:
    """把容器里的 'Assets/a/b.asset' 拆成安全的目录片段，去掉 .. 与绝对路径。"""
    parts = [p for p in re.split(r"[\\/]+", str(path_str)) if p not in ("", ".", "..")]
    return [safe_part(p) for p in parts]


def json_default(o):
    if isinstance(o, (bytes, bytearray, memoryview)):
        return {"__bytes_b64__": base64.b64encode(bytes(o)).decode("ascii")}
    if hasattr(o, "__dict__"):
        return {k: v for k, v in vars(o).items() if not k.startswith("_")}
    return str(o)


def write_json(path: Path, obj) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=json_default)


def write_text(path: Path, text: str) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def with_ext(base: Path, ext: str) -> Path:
    """在末尾追加扩展名（不用 with_suffix，避免名字里带点时被截断）。"""
    return Path(str(base) + ext)


def peek_name(obj) -> str:
    try:
        return obj.peek_name() or ""
    except Exception:
        return ""


# --------------------------------------------------------------------------- #
# 查找 bundle 文件 / 拆分首尾拼接的多个 UnityFS 包
# --------------------------------------------------------------------------- #
def sniff_bundle(path: str) -> bool:
    try:
        with open(path, "rb") as f:
            head = f.read(12)
        return any(head.startswith(m) for m in MAGICS)
    except OSError:
        return False


def find_bundles(inputs, use_sniff: bool, skip_dir: Path | None):
    """返回 [(文件路径, 输入根目录或 None)]。显式传入的单个文件不做类型检查。"""
    found = []
    for inp in inputs:
        p = Path(inp)
        if p.is_file():
            found.append((str(p), None))
            continue
        if not p.is_dir():
            print(f"[!] 路径不存在，已跳过: {inp}", file=sys.stderr)
            continue
        root = str(p.resolve())
        for dirpath, dirnames, filenames in os.walk(root):
            if skip_dir is not None:  # 输出目录放在输入目录里时，避免把自己也扫进去
                dirnames[:] = [d for d in dirnames
                               if Path(dirpath, d).resolve() != skip_dir]
            for fn in filenames:
                full = os.path.join(dirpath, fn)
                ext = os.path.splitext(fn)[1].lower()
                if ext in KNOWN_EXT:
                    found.append((full, root))
                elif use_sniff and ext not in SKIP_SNIFF_EXT and sniff_bundle(full):
                    found.append((full, root))
    found.sort(key=lambda t: t[0].lower())
    return found


def _unityfs_declared_size(mm, off: int):
    """读取 UnityFS 头里声明的 bundle 总大小；失败返回 None。"""
    try:
        p = off + 8 + 4                       # 'UnityFS\0' + 格式版本(u32)
        for _ in range(2):                    # playerVersion / engineVersion 两个 C 字符串
            e = mm.find(b"\x00", p, p + 256)
            if e < 0:
                return None
            p = e + 1
        return struct.unpack(">Q", mm[p:p + 8])[0]
    except Exception:
        return None


def locate_bundles(path: Path):
    """
    返回 [(offset, size), ...]。
    单个 UnityFS 包 -> [(0, 文件大小)]；多个包首尾拼接时按头部声明大小逐个切分。
    """
    fsize = path.stat().st_size
    if fsize < 32:
        return [(0, fsize)]
    with open(path, "rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
        first = mm.find(b"UnityFS\x00", 0, 1 << 20)
        if first < 0:
            return [(0, fsize)]               # 非 UnityFS，交给 UnityPy 整体处理
        spans, off = [], first
        while off is not None and off < fsize:
            size = _unityfs_declared_size(mm, off)
            if not size or size < 64 or off + size > fsize:
                spans.append((off, fsize - off))
                break
            spans.append((off, size))
            nxt = mm.find(b"UnityFS\x00", off + size)
            off = nxt if nxt != -1 else None
        return spans or [(0, fsize)]


def read_slice(path: Path, off: int, size: int) -> bytes:
    with open(path, "rb") as f:
        f.seek(off)
        return f.read(size)


# --------------------------------------------------------------------------- #
# 容器路径（原始资产路径）
# --------------------------------------------------------------------------- #
def container_maps(env):
    """
    读取 AssetBundle.m_Container（一对多：同一路径可对应多个对象，如 fbx 的多个子资产）。
    返回 (by_key, counts)：
      by_key: (序列化文件名, path_id) -> [容器路径...]
      counts: 容器路径 -> 被多少个对象引用
    """
    by_key, counts = {}, Counter()
    for obj in env.objects:
        if obj.type.name != "AssetBundle":
            continue
        try:
            ab = obj.read()
            fname = getattr(obj.assets_file, "name", "")
            for cpath, info in ab.m_Container:
                ptr = info.asset
                if getattr(ptr, "file_id", 0) != 0:
                    continue
                by_key.setdefault((fname, ptr.path_id), []).append(cpath)
                counts[cpath] += 1
        except Exception:
            pass
    return by_key, counts


def obj_key(obj):
    return (getattr(obj.assets_file, "name", ""), obj.path_id)


# --------------------------------------------------------------------------- #
# 各类型的写出函数：handler(obj, payload, base, ctx) -> [写出的文件 Path]
# --------------------------------------------------------------------------- #
def w_texture(obj, data, base, ctx):
    p = with_ext(base, ".png")
    data.image.save(p)
    return [p]


def w_mesh(obj, data, base, ctx):
    txt = data.export()
    if isinstance(txt, (bytes, bytearray)):
        txt = bytes(txt).decode("utf-8", "replace")
    p = with_ext(base, ".obj")
    write_text(p, txt)
    files = [p]
    if ctx["mesh_json"]:                      # 附带完整序列化数据（含蒙皮/BindPose/BlendShape 字段）
        tree = obj.read_typetree()
        if tree:                              # 包内没有类型信息时不写空文件
            p2 = with_ext(base, ".mesh.json")
            write_json(p2, tree)
            files.append(p2)
    return files


def w_text(obj, data, base, ctx):
    s = data.m_Script
    raw = s.encode("utf-8", "surrogateescape") if isinstance(s, str) else bytes(s)
    ext = ctx["cext"] if ctx["cext"] in TEXT_EXTS else None
    if ext is None:
        try:
            raw.decode("utf-8")
            ext = ".txt"
        except UnicodeDecodeError:
            ext = ".bytes"
    p = with_ext(base, ext)
    p.write_bytes(raw)
    return [p]


def w_audio(obj, data, base, ctx):
    samples = data.samples
    if not samples:
        raise ValueError("没有可用的采样数据（流式音频的 .resS 可能不在包内）")
    out = []
    for name, raw in samples.items():
        ext = Path(name).suffix or ".wav"
        stem = base if len(samples) == 1 else Path(str(base) + "_" + safe_part(Path(name).stem))
        p = with_ext(stem, ext)
        p.write_bytes(raw)
        out.append(p)
    return out


def w_font(obj, data, base, ctx):
    fd = data.m_FontData
    raw = bytes(fd) if fd else b""
    if not raw:
        raise ValueError("字体数据为空")
    p = with_ext(base, ".otf" if raw[:4] == b"OTTO" else ".ttf")
    p.write_bytes(raw)
    return [p]


def w_shader(obj, data, base, ctx):
    txt = data.export()
    if isinstance(txt, (bytes, bytearray)):
        txt = bytes(txt).decode("utf-8", "replace")
    p = with_ext(base, ".shader")
    write_text(p, txt)
    return [p]


def w_json(obj, tree, base, ctx):
    if not tree:
        raise ValueError("TypeTree 为空（包内可能没有类型信息）")
    p = with_ext(base, ".json")
    write_json(p, tree)
    return [p]


HANDLERS = {
    "Texture2D": w_texture, "Sprite": w_texture, "Mesh": w_mesh, "TextAsset": w_text,
    "AudioClip": w_audio, "Font": w_font, "Shader": w_shader,
}


def load_payload(obj, tname):
    """返回 (payload, 对象名)。特殊类型用 read()，其余用 TypeTree 字典。"""
    if tname in HANDLERS:
        data = obj.read()
        name = getattr(data, "m_Name", "") or ""
        if not name and tname == "Shader":
            name = getattr(getattr(data, "m_ParsedForm", None), "m_Name", "") or ""
        return data, name
    tree = obj.read_typetree()
    name = tree.get("m_Name", "") if isinstance(tree, dict) else ""
    return tree, name or ""


def plan_base(tname, name, pid, cpaths, counts, used):
    """
    决定输出路径（相对 bundle 输出目录、不含扩展名）与容器路径的原扩展名。
    规则：容器路径只被一个对象引用 -> 沿用原文件名；被多个对象共用(如 fbx 子资产)
          -> 目录沿用容器路径，文件名用对象名；没有容器路径 -> _nopath/<类型>/<对象名>
    """
    cext = ""
    if cpaths:
        cp = cpaths[0]
        parts = safe_rel_parts(cp)
        if not parts:
            parts = ["_"]
        dirs, fname = parts[:-1], parts[-1]
        stem, cext = os.path.splitext(fname)
        cext = cext.lower()
        leaf = (stem or fname) if (counts.get(cp, 1) == 1 or not name) else safe_part(name)
        rel = Path(*dirs, leaf) if dirs else Path(leaf)
    else:
        rel = Path("_nopath", safe_part(tname), safe_part(name or f"{tname}_{pid}"))
    key = str(rel).lower()                    # Windows/macOS 不区分大小写
    if key in used:
        rel = rel.with_name(f"{rel.name}_{pid}")
        key = str(rel).lower()
    used.add(key)
    return rel, cext


# --------------------------------------------------------------------------- #
# 单个 env（一个 bundle）的三种处理模式
# --------------------------------------------------------------------------- #
def do_extract(env, outdir: Path, opts, res, label):
    by_key, counts = container_maps(env)
    want = None if opts["types"] == "all" else {t.lower() for t in opts["types"].split(",") if t}
    used, index = set(), []
    outdir.mkdir(parents=True, exist_ok=True)
    for obj in env.objects:
        tname = obj.type.name
        res["objects"] += 1
        res["types"][tname] = res["types"].get(tname, 0) + 1
        if want is not None and tname.lower() not in want:
            res["skipped"] += 1
            continue
        pid = obj.path_id
        cpaths = by_key.get(obj_key(obj), [])
        entry = {"path_id": pid, "type": tname, "container": cpaths[:1]}
        payload, name, load_err = None, "", None
        try:
            payload, name = load_payload(obj, tname)
        except Exception as e:                # 读取失败 -> 稍后尝试回退为原始数据
            load_err = e
            name = peek_name(obj)
        rel, cext = plan_base(tname, name, pid, cpaths, counts, used)
        base = outdir / rel
        entry["name"] = name
        try:
            base.parent.mkdir(parents=True, exist_ok=True)
            if load_err is not None:
                raise load_err
            ctx = {"cext": cext, "mesh_json": opts["mesh_json"]}
            files = HANDLERS.get(tname, w_json)(obj, payload, base, ctx)
            entry.update(status="ok", files=[str(f.relative_to(outdir)) for f in files])
            res["ok"] += 1
        except Exception as e:
            msg = f"{type(e).__name__}: {e}"
            if opts["raw_fallback"]:
                try:
                    p = with_ext(base, ".bin")
                    p.write_bytes(bytes(obj.get_raw_data()))
                    entry.update(status="raw", files=[str(p.relative_to(outdir))], error=msg)
                    res["raw"] += 1
                    res["errors"].append(f"{label} | {tname} pid={pid} | 回退为原始数据 | {msg}")
                    index.append(entry)
                    continue
                except Exception as e2:
                    msg += f" | 原始数据回退也失败: {type(e2).__name__}: {e2}"
            entry.update(status="failed", error=msg)
            res["failed"] += 1
            res["errors"].append(f"{label} | {tname} pid={pid} | {msg}")
        index.append(entry)
    write_json(outdir / "_index.json", {
        "bundle": label,
        "objects": index,
        "container": [{"path": c, "path_id": k[1]} for k, cs in by_key.items() for c in cs],
    })


def do_list(env, label, res):
    by_key, _ = container_maps(env)
    for obj in env.objects:
        res["objects"] += 1
        res["types"][obj.type.name] = res["types"].get(obj.type.name, 0) + 1
    res["finds"].append({"bundle": label, "container_entries": sum(len(v) for v in by_key.values())})


def do_find(env, label, res, keywords):
    by_key, _ = container_maps(env)
    for obj in env.objects:
        res["objects"] += 1
        cps = by_key.get(obj_key(obj), [])
        names = list(cps) + [peek_name(obj)]
        hay = " ".join(names).lower()
        if any(k in hay for k in keywords):
            res["finds"].append({"bundle": label, "type": obj.type.name,
                                 "path_id": obj.path_id, "where": cps[0] if cps else names[-1]})


# --------------------------------------------------------------------------- #
# 单个文件的处理（可在子进程里运行）
# --------------------------------------------------------------------------- #
def process_file(path_str, input_root, out_root, opts):
    t0 = time.time()
    res = {"file": path_str, "bundles": 0, "objects": 0, "ok": 0, "raw": 0, "failed": 0,
           "skipped": 0, "types": {}, "errors": [], "finds": [], "status": "ok",
           "note": "", "seconds": 0.0}
    try:
        if opts.get("unity_version"):
            UnityPy.config.FALLBACK_UNITY_VERSION = opts["unity_version"]
        p = Path(path_str)
        fsize = p.stat().st_size
        rel_parent = Path(os.path.relpath(p.parent, input_root)) if input_root else Path()
        if str(rel_parent) == ".":
            rel_parent = Path()
        spans = [(0, fsize)] if opts["no_split"] else locate_bundles(p)
        multi = len(spans) > 1
        for i, (off, size) in enumerate(spans):
            label = f"{p.stem}_{i:03d}" if multi else p.stem
            outdir = Path(out_root) / rel_parent / label
            if opts["mode"] == "extract" and opts["skip_existing"] and (outdir / "_index.json").exists():
                res["note"] = "已存在，跳过"
                continue
            try:
                whole = (off == 0 and size == fsize)
                env = UnityPy.load(str(p)) if whole else UnityPy.load(read_slice(p, off, size))
                if not any(True for _ in env.objects):   # UnityPy 对无效数据不抛异常，只会得到空 env
                    raise ValueError("没有解析出任何对象（不是有效的 Unity 包，或被加密/使用了自定义压缩）")
                res["bundles"] += 1
                if opts["mode"] == "extract":
                    do_extract(env, outdir, opts, res, label)
                elif opts["mode"] == "list":
                    do_list(env, label, res)
                else:
                    do_find(env, label, res, opts["keywords"])
            except Exception as e:
                res["status"] = "partial" if res["bundles"] else "failed"
                res["errors"].append(f"{label} | 加载/处理失败 | {type(e).__name__}: {e}")
    except Exception as e:
        res["status"] = "failed"
        res["errors"].append(f"{path_str} | {type(e).__name__}: {e}")
    if res["bundles"] == 0 and res["status"] == "ok" and not res["note"]:
        res["status"] = "failed"
    res["seconds"] = round(time.time() - t0, 2)
    return res


# --------------------------------------------------------------------------- #
# 命令行入口
# --------------------------------------------------------------------------- #
def parse_args():
    ap = argparse.ArgumentParser(
        description="基于 UnityPy 的 .uab/.ab 资产批量提取工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("示例", 1)[1] if "示例" in __doc__ else "")
    ap.add_argument("inputs", nargs="+", help="bundle 文件或目录（可给多个，目录递归扫描）")
    ap.add_argument("-o", "--output", default="./uab_output", help="输出目录（默认 ./uab_output）")
    ap.add_argument("-t", "--types", default=DEFAULT_TYPES,
                    help="要导出的类型，逗号分隔；'all' 导出全部（默认: %(default)s）")
    ap.add_argument("-j", "--jobs", type=int, default=1, help="并行进程数（默认 1）")
    ap.add_argument("--find", metavar="关键字", help="只查找：在容器路径/对象名里搜关键字（逗号分隔多个），不导出")
    ap.add_argument("--list", action="store_true", help="只统计每个包的对象类型，不导出")
    ap.add_argument("--mesh-json", action="store_true", help="导出 Mesh 时额外写出 .mesh.json（含蒙皮/顶点序列化数据）")
    ap.add_argument("--skip-existing", action="store_true", help="输出里已有 _index.json 的包直接跳过（断点续跑）")
    ap.add_argument("--no-sniff", action="store_true", help="目录扫描时只按扩展名识别，不读文件头")
    ap.add_argument("--no-split", action="store_true", help="不拆分首尾拼接的多个 UnityFS 包")
    ap.add_argument("--no-raw-fallback", action="store_true", help="解析失败时不回退保存原始 .bin")
    ap.add_argument("--unity-version", help="包内缺少版本信息时指定，如 2022.3.62f1")
    return ap.parse_args()


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    a = parse_args()
    out_root = Path(a.output).resolve()
    mode = "find" if a.find else ("list" if a.list else "extract")
    opts = {
        "mode": mode, "types": a.types.strip() if a.types.strip().lower() != "all" else "all",
        "mesh_json": a.mesh_json, "skip_existing": a.skip_existing, "no_split": a.no_split,
        "raw_fallback": not a.no_raw_fallback, "unity_version": a.unity_version,
        "keywords": [k.strip().lower() for k in (a.find or "").split(",") if k.strip()],
    }
    files = find_bundles(a.inputs, use_sniff=not a.no_sniff, skip_dir=out_root)
    if not files:
        print("[!] 没有找到任何 bundle 文件。可以去掉 --no-sniff，或直接传入具体文件。")
        return 1
    print(f"[*] 共发现 {len(files)} 个文件 | 模式: {mode} | 输出: {out_root if mode == 'extract' else '(不写文件)'}")
    if mode == "extract":
        out_root.mkdir(parents=True, exist_ok=True)

    results, t_start = [], time.time()
    bar = tqdm(total=len(files), unit="file", desc="处理中") if tqdm else None

    def report(r):
        results.append(r)
        if mode == "find" and r["status"] == "ok" and not r["finds"]:
            if bar:                           # 无命中的文件不刷屏，只推进进度条
                bar.update(1)
            return
        tag = {"ok": "OK  ", "partial": "PART", "failed": "FAIL"}[r["status"]]
        if mode == "find":
            line = f"[{tag}] {r['file']}  （命中 {len(r['finds'])} 处）"
        elif mode == "extract":
            line = (f"[{tag}] {os.path.basename(r['file'])}  包={r['bundles']} 对象={r['objects']} "
                    f"成功={r['ok']} 回退={r['raw']} 失败={r['failed']} 跳过类型={r['skipped']} "
                    f"({r['seconds']}s){'  ' + r['note'] if r['note'] else ''}")
        else:
            line = f"[{tag}] {os.path.basename(r['file'])}  包={r['bundles']} 对象={r['objects']}"
        (bar.write if bar else print)(line)
        if r["status"] != "ok" and r["errors"]:
            (bar.write if bar else print)(f"      原因: {r['errors'][0][:200]}")
        if mode == "find":
            for m in r["finds"]:
                (bar.write if bar else print)(f"      {m['bundle']} :: [{m['type']}] {m['where']}  (path_id={m['path_id']})")
        if mode == "list":
            top = ", ".join(f"{k}×{v}" for k, v in sorted(r["types"].items(), key=lambda kv: -kv[1])[:8])
            if top:
                (bar.write if bar else print)(f"      {top}")
        if bar:
            bar.update(1)

    try:
        if a.jobs > 1 and len(files) > 1:
            with ProcessPoolExecutor(max_workers=a.jobs) as ex:
                futs = [ex.submit(process_file, f, root, str(out_root), opts) for f, root in files]
                for fu in as_completed(futs):
                    report(fu.result())
        else:
            for f, root in files:
                report(process_file(f, root, str(out_root), opts))
    except KeyboardInterrupt:
        print("\n[!] 用户中断，已处理的结果保留。使用 --skip-existing 可继续。")
    finally:
        if bar:
            bar.close()

    # ---- 汇总 ----
    tot = Counter()
    types = Counter()
    errors = []
    for r in results:
        for k in ("bundles", "objects", "ok", "raw", "failed", "skipped"):
            tot[k] += r[k]
        types.update(r["types"])
        errors += r["errors"]
    bad = sum(1 for r in results if r["status"] == "failed")
    print("\n" + "=" * 60)
    print(f"文件 {len(results)}（失败 {bad}） | 包 {tot['bundles']} | 对象 {tot['objects']} | "
          f"成功 {tot['ok']} | 回退原始 {tot['raw']} | 失败 {tot['failed']} | 耗时 {time.time() - t_start:.1f}s")
    if types:
        print("类型分布: " + ", ".join(f"{k}×{v}" for k, v in types.most_common(12)))
    if mode == "extract":
        write_json(out_root / "_summary.json", {"totals": dict(tot), "types": dict(types),
                                                 "files": [{k: r[k] for k in ("file", "status", "bundles", "objects", "ok", "raw", "failed")} for r in results]})
        if errors:
            with open(out_root / "errors.log", "w", encoding="utf-8") as f:
                f.write("\n".join(errors) + "\n")
            print(f"有 {len(errors)} 条错误/回退记录，详见 {out_root / 'errors.log'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
