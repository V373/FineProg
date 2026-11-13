#!/usr/bin/env python3
"""Index validated raw/processed comparison MP4s in Markdown and local HTML.

Only generates indexes beside the source data manifest. It does not render
videos, alter data/cleaning decisions, or run a progress model.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from urllib.parse import quote


GROUPS = (("train", "Train"), ("valid", "Valid"),
          ("calibration", "Calibration"), ("excluded", "隔离示范"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def md_text(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def local_link(path: Path, root: Path) -> str:
    return quote(path.resolve().relative_to(root.resolve()).as_posix(), safe="/_-.")


def source_order(row: dict):
    name = row["source_episode"]
    return int(name.rsplit("_", 1)[-1]) if name.startswith("episode_init_") else name


def collect_rows(manifest_path: Path, video_manifest_path: Path) -> tuple[dict, list[dict]]:
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    media = json.loads(video_manifest_path.read_text(encoding="utf-8"))
    if media.get("source_manifest_sha256") != sha256(manifest_path):
        raise ValueError("Video manifest does not match the frozen data manifest")
    if media.get("layout") != "source_synchronized":
        raise ValueError("Expected source-synchronized comparison videos")
    root = manifest_path.resolve().parent
    expected = {(task, episode["source_episode"]): episode
                for task, details in data["tasks"].items()
                for episode in details["episodes"]}
    videos = media["videos"]
    found = [(item["task"], item["source_episode"]) for item in videos]
    if len(found) != len(set(found)) or set(found) != set(expected):
        raise ValueError("Comparison videos must cover every original episode exactly once")
    rows = []
    for item in videos:
        task, episode_name = item["task"], item["source_episode"]
        source = expected[(task, episode_name)]
        for key in ("status", "split", "video_id", "source_frames", "start_frame",
                    "completion_frame", "end_frame", "retained_frames"):
            if item.get(key) != source.get(key):
                raise ValueError(f"Stale comparison metadata: {task}/{episode_name}/{key}")
        n = source["source_frames"]
        if (not item.get("validation", {}).get("passed") or item.get("frame_count") != n
                or item.get("validation", {}).get("decoded_frames") != n):
            raise ValueError(f"Full MP4 decode/count check missing: {task}/{episode_name}")
        fps = float(item["nominal_fps"])
        if fps != 20.0 or abs(float(item["duration_seconds"]) - n / fps) > 1 / fps:
            raise ValueError(f"Unexpected nominal timing: {task}/{episode_name}")
        path = Path(item["path"])
        sidecar = Path(item["sidecar_path"])
        if not path.is_file() or path.stat().st_size == 0 or not sidecar.is_file():
            raise FileNotFoundError(f"Missing MP4/sidecar: {task}/{episode_name}")
        group = source["split"] if source["status"] == "keep" else "excluded"
        rows.append({**source, "task": task, "group": group,
                     "video_url": local_link(path, root),
                     "sidecar_url": local_link(sidecar, root),
                     "nominal_fps": fps, "duration_seconds": n / fps,
                     "retained_seconds": source["retained_frames"] / fps,
                     "width": item["width"], "height": item["height"]})
    tasks = list(data["tasks"])
    rows.sort(key=lambda row: (tasks.index(row["task"]), source_order(row)))
    return data, rows


def markdown(data: dict, rows: list[dict], video_manifest_path: Path, root: Path) -> str:
    keep = sum(row["status"] == "keep" for row in rows)
    lines = [
        '<a id="top"></a>', "", "# Raw / 裁剪后视频对比索引", "",
        f"共 **{len(rows)} 个 demo、{len(rows)} 个 MP4**：{keep} 条保留示范、{len(rows) - keep} 条隔离示范。每条均已完整解码并核对帧数。", "",
        "**[打开本地快速浏览器](CROP_VIDEO_BROWSER.html)**：按任务／划分筛选、搜索源 demo ID，支持上一条／下一条及起点／完成／终点定位。也可直接使用下面的 Markdown 视频链接。", "",
        "上方左侧播放完整原始 `topFullImg`（标出空间 ROI），右侧显示实际 processed H5 的224×224图像。两侧按**同一原始帧号同步**：右侧仅在保留区间出现，删去的头尾显示提示；隔离示范右侧始终显示排除信息。下方时间轴标明保留区间、语义完成帧和当前帧。", "",
        "视频长度保留原始 demo 的全时长，没有把删去片段压缩跳过。时间为20 FPS名义时间，未声称实测采集间隔；起止帧为零基闭区间。表中“原始/保留秒数”分别按帧数÷20计算。", "",
        "[原清洗图文报告](README.md) · [全量裁剪 CSV](data_manifest.csv) · "
        f"[媒体生成及验证记录]({local_link(video_manifest_path, root)})", "",
        "## 快速跳转", "",
    ]
    for task in data["tasks"]:
        subset = [row for row in rows if row["task"] == task]
        counts = Counter(row["group"] for row in subset)
        links = " · ".join(f"[{label} {counts[key]}](#{task}-{key})" for key, label in GROUPS)
        lines.append(f"- **[{task}：{len(subset)}条](#{task})** — {links}")
    lines.extend(["", "可用编辑器的搜索功能按完整源 demo ID 或 processed ID 查找。", ""])
    for task in data["tasks"]:
        roi = data["tasks"][task]["roi_xy_size"]
        lines.extend([f'<a id="{task}"></a>', "", f"## {task}", "",
                      f"固定空间裁剪：`topFullImg` 的 `(x={roi[0]}, y={roi[1]}, size={roi[2]})`，输出224×224。[返回快速索引](#top)。", ""])
        for group, label in GROUPS:
            selected = [row for row in rows if row["task"] == task and row["group"] == group]
            lines.extend([f'<a id="{task}-{group}"></a>', "", f"### {label}（{len(selected)}条）", ""])
            if not selected:
                lines.extend(["无。", ""])
                continue
            if group == "excluded":
                lines.extend(["| 源 demo / 对比视频 | 原始帧数 | 名义秒数 | 隔离原因 |", "|---|---:|---:|---|"])
                for row in selected:
                    lines.append(f"| [{row['source_episode']}]({row['video_url']}) | {row['source_frames']} | {row['duration_seconds']:.2f} | {md_text(row['reason'])} |")
            else:
                lines.extend(["| 源 demo / 对比视频 | Processed ID | 保留原始帧区间 | 完成帧 | 原始→保留帧数 | 原始/保留秒数 |", "|---|---|---|---:|---:|---:|"])
                for row in selected:
                    lines.append(f"| [{row['source_episode']}]({row['video_url']}) | `{row['video_id']}` | `[{row['start_frame']}, {row['end_frame']}]` | {row['completion_frame']} | {row['source_frames']} → {row['retained_frames']} | {row['duration_seconds']:.2f} / {row['retained_seconds']:.2f} |")
            lines.extend(["", "[返回快速索引](#top)", ""])
    return "\n".join(lines)


HTML_TEMPLATE = r'''<!doctype html>
<html lang="zh-CN">
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Raw / 裁剪后视频快速浏览</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#f4f6f8;color:#182635;font:15px/1.5 system-ui,sans-serif}
header{padding:18px 24px;background:white;border-bottom:1px solid #d8e0e8}h1{font-size:22px;margin:0 0 6px}a{color:#1763a6}
.layout{display:grid;grid-template-columns:330px minmax(0,1fr);gap:18px;padding:18px;max-width:1800px;margin:auto}
aside,main{background:white;border:1px solid #d8e0e8;border-radius:8px;padding:14px;min-width:0}
.filters{display:grid;grid-template-columns:1fr 1fr;gap:8px}input,select,button{font:inherit;border:1px solid #bccbd8;border-radius:5px;padding:7px;background:white;color:inherit}
input{width:100%;margin:8px 0}button{cursor:pointer}button:disabled{opacity:.45;cursor:default}
#list{max-height:68vh;overflow:auto;margin-top:8px}#list button{display:block;text-align:left;width:100%;margin-bottom:5px;font-size:13px}
#list button.active{border-color:#247f50;background:#e4f4e9}small{display:block;color:#526677}#count{color:#526677;font-size:13px}
video{width:100%;max-height:70vh;background:#0e1825;display:block}.nav,.seek{display:flex;gap:8px;flex-wrap:wrap;margin:12px 0}
#title{font-size:19px;margin:0 0 8px}#detail{white-space:pre-line}#reason{margin-top:10px;padding:10px;background:#f3f7fa;border-radius:5px}
.hint{color:#526677;font-size:13px;margin-top:12px}@media(max-width:850px){.layout{grid-template-columns:1fr}#list{max-height:24vh}}
</style>
<header><h1>Raw / 裁剪后视频快速浏览</h1>
<a href="CROP_VIDEO_INDEX.md">Markdown完整索引</a> · <a href="README.md">清洗图文报告</a>
<div class="hint">左右按原始帧号同步。右侧仅在保留区间显示processed图像；删去的头尾及隔离示范显示提示。20 FPS为名义时间。</div></header>
<div class="layout"><aside>
<div class="filters"><select id="task"><option value="">全部任务</option></select>
<select id="split"><option value="">全部划分</option><option value="train">Train</option><option value="valid">Valid</option><option value="calibration">Calibration</option><option value="excluded">隔离示范</option></select></div>
<input id="search" placeholder="搜索源 demo 或 processed ID" aria-label="搜索 demo">
<div id="count"></div><div id="list"></div></aside>
<main><h2 id="title"></h2><video id="video" controls playsinline preload="metadata"></video>
<div class="nav"><button id="prev">← 上一条</button><button id="next">下一条 →</button><a id="download" download>下载 MP4</a><a id="record">来源与验证 JSON</a></div>
<div class="seek"><button id="zero">原始开头</button><button id="start">裁剪起点</button><button id="completion">语义完成</button><button id="end">裁剪终点</button></div>
<div id="detail"></div><div id="reason"></div><div class="hint">建议重点检查：初始抖动是否位于保留区间之前；语义完成后是否及时截断。drawer遮挡处完成帧为图像与状态辅助估计，可能存在数帧定位误差。</div></main></div>
<script>
const rows=__DATA__;
const $=id=>document.getElementById(id), video=$('video');
const groupNames={train:'Train',valid:'Valid',calibration:'Calibration',excluded:'隔离'};
let visible=[], selected=null;
for(const name of [...new Set(rows.map(r=>r.task))]){const o=document.createElement('option');o.value=name;o.textContent=name;$('task').append(o)}
function key(r){return `${r.task}/${r.source_episode}`}
function redrawList(){
 $('list').replaceChildren();
 for(const r of visible){const b=document.createElement('button');b.className=selected&&key(r)===key(selected)?'active':'';
 const name=document.createElement('span');name.textContent=r.source_episode;
 const sub=document.createElement('small');sub.textContent=`${r.task} · ${groupNames[r.group]}${r.video_id?' · '+r.video_id:''}`;
 b.append(name,sub);b.onclick=()=>select(r);$('list').append(b)}
 const i=visible.findIndex(r=>selected&&key(r)===key(selected));$('prev').disabled=i<=0;$('next').disabled=i<0||i>=visible.length-1;
 $('count').textContent=`当前 ${visible.length} / ${rows.length} 条${i>=0?'；第 '+(i+1)+' 条':''}`;
}
function select(r){
 selected=r;video.pause();video.src=r.video_url;video.load();
 $('title').textContent=`${r.task} / ${r.source_episode}`;$('download').href=r.video_url;$('record').href=r.sidecar_url;
 const kept=r.status==='keep';
 $('detail').textContent=kept?`${groupNames[r.group]} / processed ${r.video_id}\n原始 ${r.source_frames} 帧 (${r.duration_seconds.toFixed(2)}s)，保留 [${r.start_frame}, ${r.end_frame}] 共 ${r.retained_frames} 帧 (${r.retained_seconds.toFixed(2)}s)；语义完成帧 ${r.completion_frame}`:`隔离示范：原始 ${r.source_frames} 帧 (${r.duration_seconds.toFixed(2)}s)，没有裁剪后成功片段。`;
 $('reason').textContent=r.reason;
 for(const id of ['start','completion','end'])$(id).disabled=!kept;
 redrawList();
}
function filter(){const text=$('search').value.toLowerCase().trim();visible=rows.filter(r=>(!$('task').value||r.task===$('task').value)&&(!$('split').value||r.group===$('split').value)&&(!text||`${r.task} ${r.source_episode} ${r.video_id||''}`.toLowerCase().includes(text)));
 if(!visible.length){selected=null;video.pause();video.removeAttribute('src');video.load();$('title').textContent='没有匹配的 demo';$('detail').textContent='';$('reason').textContent='';$('download').removeAttribute('href');$('record').removeAttribute('href');for(const id of ['zero','start','completion','end'])$(id).disabled=true;redrawList();return}
 $('zero').disabled=false;const same=visible.find(r=>selected&&key(r)===key(selected));if(!same)select(visible[0]);else redrawList();
}
function seek(frame){if(!selected)return;const t=Math.max(0,frame/selected.nominal_fps);if(video.readyState>=1)video.currentTime=t;else video.addEventListener('loadedmetadata',()=>video.currentTime=t,{once:true})}
$('zero').onclick=()=>seek(0);$('start').onclick=()=>seek(selected.start_frame);$('completion').onclick=()=>seek(selected.completion_frame);$('end').onclick=()=>seek(selected.end_frame);
for(const [id,offset] of [['prev',-1],['next',1]])$(id).onclick=()=>{const i=visible.findIndex(r=>selected&&key(r)===key(selected));if(visible[i+offset])select(visible[i+offset])};
for(const id of ['task','split','search'])$(id).addEventListener(id==='search'?'input':'change',filter);
filter();
</script></html>
'''


def browser(rows: list[dict]) -> str:
    # Embedded JSON keeps the browser usable through file:// without a local server.
    payload = json.dumps(rows, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
    return HTML_TEMPLATE.replace("__DATA__", payload)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--video-manifest", type=Path, required=True)
    args = parser.parse_args()
    manifest_path = args.manifest.expanduser().resolve()
    video_manifest_path = args.video_manifest.expanduser().resolve()
    root = manifest_path.parent
    data, rows = collect_rows(manifest_path, video_manifest_path)
    md_path = root / "CROP_VIDEO_INDEX.md"
    html_path = root / "CROP_VIDEO_BROWSER.html"
    # Keep the reviewed downstream-data appendix when refreshing video links.
    marker = "<!-- subopt-failure-audit -->"
    existing = md_path.read_text(encoding="utf-8") if md_path.exists() else ""
    appendix = existing.partition(marker)[2]
    body = markdown(data, rows, video_manifest_path, root)
    if appendix:
        body = body.replace("## 快速跳转", "[Suboptimal / failure 数据汇总](#subopt-failure)\n\n## 快速跳转", 1)
        body = body.rstrip() + "\n\n" + marker + appendix
    md_path.write_text(body, encoding="utf-8")
    html_path.write_text(browser(rows), encoding="utf-8")
    print(json.dumps({"videos": len(rows), "markdown": str(md_path),
                      "browser": str(html_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
