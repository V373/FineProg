#!/usr/bin/env python3
"""Source-synchronized raw/processed comparison movies for reviewed real demos.

Raw playback covers every source frame at nominal 20 FPS. The right panel shows
the actual processed H5 frame only inside the retained inclusive source range;
removed intervals and excluded demonstrations show explicit slates. No training.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import subprocess

import cv2
import h5py
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont


WIDTH, HEIGHT, FPS = 1280, 800, 20
FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
BG = (17, 24, 39)
PANEL = (30, 41, 59)
WHITE = (236, 242, 250)
MUTED = (170, 185, 205)
GREEN = (65, 204, 139)
RED = (193, 84, 100)
ORANGE = (255, 187, 75)
BLUE = (97, 165, 250)
X0, X1 = 50, 1230


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    partial = path.with_suffix(".json.partial")
    partial.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    partial.replace(path)


class Layout:
    def __init__(self, task, detail, episode):
        self.e = e = episode
        self.roi = detail["roi_xy_size"]
        self.keep = e["status"] == "keep"
        self.n = e["source_frames"]
        self.s, self.end, self.c = (e.get(k) for k in ("start_frame", "end_frame", "completion_frame"))
        self.total = e["retained_frames"]
        self.font = {n: ImageFont.truetype(FONT_PATH, n) for n in (16, 18, 20, 22, 26)}
        im = Image.new("RGB", (WIDTH, HEIGHT), BG)
        d = ImageDraw.Draw(im)
        title = f"{task} / {e['source_episode']}   |   {e.get('split') or 'excluded'}"
        d.text((16, 7), title, font=self.font[26], fill=WHITE)
        d.text((16, 46), "裁剪前 · 完整原始画面（绿框为空间 ROI）", font=self.font[20], fill=WHITE)
        d.text((656, 46), "裁剪后 · 实际 processed H5 输入（224×224）", font=self.font[20], fill=WHITE)
        d.rectangle((16, 82, 623, 537), fill=PANEL)
        d.rectangle((656, 82, 1263, 537), fill=PANEL)
        d.text((16, 584), "原始时间轴：红色为删除，绿色为保留；白色游标为当前原始帧", font=self.font[20], fill=WHITE)
        if self.keep:
            info = f"保留 [{self.s}, {self.end}]（含首尾） / {self.total} 帧    完成参考帧 {self.c}    原始 {self.n} 帧"
        else:
            info = f"整条隔离 / 原始 {self.n} 帧    没有裁剪后专家片段"
        d.text((16, 612), info, font=self.font[18], fill=MUTED)
        d.rectangle((X0, 646, X1, 670), fill=RED)
        if self.keep:
            d.rectangle((self.source_x(self.s), 646, self.source_x(self.end), 670), fill=GREEN)
            d.line((self.source_x(self.c), 641, self.source_x(self.c), 674), fill=ORANGE, width=2)
        for t in np.linspace(0, self.n - 1, 5).round().astype(int):
            x = self.source_x(t)
            d.line((x, 673, x, 677), fill=MUTED)
            label = f"{t}f / {t/FPS:.2f}s"
            box = d.textbbox((0, 0), label, font=self.font[16])
            tx = max(16, min(WIDTH - 16 - (box[2] - box[0]), x - (box[2] - box[0]) / 2))
            d.text((tx, 679), label, font=self.font[16], fill=MUTED)
        label = f"裁剪后片段时间轴：{self.total} 帧 / {self.total/FPS:.2f}s" if self.keep else "裁剪后片段时间轴：不适用（隔离示范）"
        d.text((16, 705), label, font=self.font[18], fill=WHITE)
        d.rectangle((X0, 736, X1, 751), fill=BLUE if self.keep else PANEL)
        if self.keep:
            d.text((X0, 753), "processed 0", font=self.font[16], fill=MUTED)
            last = f"processed {self.total-1}"
            width = d.textbbox((0, 0), last, font=self.font[16])[2]
            d.text((X1 - width, 753), last, font=self.font[16], fill=MUTED)
        d.text((16, 777), "按源帧同步；20 FPS 为名义播放速率。右侧不补帧、不循环；未修改原始数据或 processed 数据。", font=self.font[16], fill=MUTED)
        self.base = np.array(im)
        self.slates = {}
        for status, title, body in (
            ("before", "已删除的录制前段", "等待 / 启动调整不进入裁剪后片段\n到达正式任务起点后，右侧开始播放"),
            ("after", "裁剪后片段已结束", "语义完成附近已截断\n左侧继续展示被删去的收尾 / 回位"),
            ("excluded", "本 demo 已隔离", e.get("reason", "没有生成裁剪后的专家片段")),
        ):
            panel = Image.new("RGB", (608, 456), PANEL)
            draw = ImageDraw.Draw(panel)
            draw.text((28, 112), title, font=self.font[26], fill=ORANGE if status == "excluded" else GREEN)
            lines = []
            for paragraph in body.split("\n"):
                line = ""
                for char in paragraph:
                    if draw.textlength(line + char, font=self.font[20]) > 548:
                        lines.append(line)
                        line = char
                    else:
                        line += char
                lines.append(line)
            for row, line in enumerate(lines):
                draw.text((28, 164 + row * 31), line, font=self.font[20], fill=WHITE)
            self.slates[status] = np.array(panel)

    def source_x(self, frame):
        return int(round(X0 + (X1 - X0) * frame / max(1, self.n - 1)))

    def frame(self, index: int, raw_rgb: np.ndarray, processed: np.ndarray | None):
        out = self.base.copy()
        raw = cv2.resize(raw_rgb, (608, 456), interpolation=cv2.INTER_AREA)
        x, y, size = self.roi
        cv2.rectangle(raw, (round(x * .95), round(y * .95)),
                      (round((x + size) * .95) - 1, round((y + size) * .95) - 1), GREEN, 2)
        out[82:538, 16:624] = raw
        active = self.keep and self.s <= index <= self.end
        if active:
            out[82:538, 732:1188] = cv2.resize(processed, (456, 456), interpolation=cv2.INTER_LINEAR)
            cv2.rectangle(out, (731, 81), (1188, 538), GREEN, 1)
            offset = index - self.s
            message = f"processed f={offset}/{self.total-1}  |  t={offset/FPS:.2f}s  |  raw f={index}"
            cursor = round(X0 + (X1 - X0) * offset / max(1, self.total - 1))
            cv2.line(out, (cursor, 731), (cursor, 755), WHITE, 3)
        else:
            status = "excluded" if not self.keep else "before" if index < self.s else "after"
            out[82:538, 656:1264] = self.slates[status]
            message = "processed: excluded" if not self.keep else "processed: not yet started" if index < self.s else "processed: finished"
        cv2.putText(out, f"raw f={index}/{self.n-1}  |  t={index/FPS:.2f}s", (18, 565),
                    cv2.FONT_HERSHEY_SIMPLEX, .59, WHITE, 1, cv2.LINE_AA)
        cv2.putText(out, message, (658, 565), cv2.FONT_HERSHEY_SIMPLEX, .51, WHITE, 1, cv2.LINE_AA)
        cursor = self.source_x(index)
        cv2.line(out, (cursor, 639), (cursor, 675), WHITE, 3)
        cv2.circle(out, (cursor, 638), 4, WHITE, -1)
        return out


def validate_video(path: Path, count: int, samples: dict) -> dict:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    codec = "".join(chr((fourcc >> (8 * i)) & 255) for i in range(4))
    decoded = 0
    errors = {}
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            if bgr.shape != (HEIGHT, WIDTH, 3):
                raise ValueError(f"Unexpected decoded shape at {decoded}: {bgr.shape}")
            if decoded in samples:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                mae = float(np.mean(np.abs(rgb.astype(np.int16) - samples[decoded].astype(np.int16))))
                errors[str(decoded)] = round(mae, 4)
                if mae > 10:
                    raise ValueError(f"Encoded frame mismatch: {path}/{decoded}: MAE {mae}")
            decoded += 1
    finally:
        cap.release()
    if decoded != count or abs(fps - FPS) > 1e-6 or codec not in ("avc1", "h264") or len(errors) != len(samples):
        raise ValueError(f"Video validation failed: {path}: {decoded}/{count}, {fps}, {codec}")
    return {"passed": True, "decoded_frames": decoded, "fps": fps, "codec": codec,
            "width": WIDTH, "height": HEIGHT, "sample_rgb_mae": errors}


def render_one(task, detail, e, session: Path, source_hash: str, ffmpeg: str) -> dict:
    directory = session / "videos/crop_comparison" / task
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{e['source_episode']}.mp4"
    sidecar = path.with_suffix(".json")
    if path.exists() and sidecar.exists():
        old = json.loads(sidecar.read_text())
        if (old.get("source_manifest_sha256") == source_hash and old.get("render_version") == 1
                and old.get("validation", {}).get("passed") and old.get("video_sha256") == digest(path)):
            return old
        raise ValueError(f"Existing video does not match this source/render: {path}")
    layout = Layout(task, detail, e)
    cleaned = None
    spec = detail["splits"].get(e.get("split"))
    if layout.keep:
        with h5py.File(spec["path"], "r") as h5:
            group = h5[f"videos/{e['video_id']}"]
            if group.attrs["source_episode"] != e["source_episode"]:
                raise ValueError("Processed source episode mismatch")
            if not np.array_equal(group["source_frame_indices"][:], np.arange(layout.s, layout.end + 1)):
                raise ValueError("Processed source-frame mapping mismatch")
            cleaned = group["frames"][:]
            if len(cleaned) != e["retained_frames"]:
                raise ValueError("Processed length mismatch")
    sample_ids = {0, e["source_frames"] - 1}
    if layout.keep:
        sample_ids.update({layout.s, layout.c, layout.end, (layout.s + layout.end) // 2})
    else:
        sample_ids.add(e["source_frames"] // 2)
    samples = {}
    preview_index = (layout.s + layout.end) // 2 if layout.keep else e["source_frames"] // 2
    partial = path.with_suffix(".partial.mp4")
    log_path = session / "work/crop_comparison" / f"{task}_{e['source_episode']}_encode.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    raw_digest = hashlib.sha256()
    command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
               "-vcodec", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{WIDTH}x{HEIGHT}",
               "-r", str(FPS), "-i", "-", "-an", "-c:v", "libx264", "-preset", "veryfast",
               "-crf", "21", "-pix_fmt", "yuv420p", "-threads", "2", "-movflags", "+faststart", str(partial)]
    with log_path.open("wb") as logfile:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=logfile)
        try:
            for index in range(e["source_frames"]):
                source = Path(e["source_path"]) / detail["camera"] / f"{index}.jpg"
                payload = source.read_bytes()
                raw_digest.update(index.to_bytes(8, "little"))
                raw_digest.update(len(payload).to_bytes(8, "little"))
                raw_digest.update(payload)
                raw = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
                if raw is None or raw.shape != (480, 640, 3):
                    raise ValueError(f"Unreadable source image: {source}")
                active = layout.keep and layout.s <= index <= layout.end
                output = layout.frame(index, cv2.cvtColor(raw, cv2.COLOR_BGR2RGB), cleaned[index - layout.s] if active else None)
                if index in sample_ids:
                    samples[index] = output.copy()
                if index == preview_index:
                    Image.fromarray(output).save(path.with_suffix(".jpg"), quality=92)
                process.stdin.write(output.tobytes())
            process.stdin.close()
            if process.wait() != 0:
                raise RuntimeError(f"Encoding failed; see {log_path}")
        except BaseException:
            if process.poll() is None:
                process.kill()
                process.wait()
            if not process.stdin.closed:
                process.stdin.close()
            raise
    validation = validate_video(partial, e["source_frames"], samples)
    partial.replace(path)
    record = {k: e.get(k) for k in ("source_episode", "status", "split", "video_id", "source_frames",
                                  "start_frame", "completion_frame", "end_frame", "retained_frames", "reason")}
    record.update(task=task, path=str(path.resolve()), relative_path=str(path.relative_to(session)),
                  sidecar_path=str(sidecar.resolve()), cover_path=str(path.with_suffix(".jpg").resolve()),
                  frame_count=e["source_frames"], nominal_fps=FPS, duration_seconds=e["source_frames"] / FPS,
                  width=WIDTH, height=HEIGHT, validation=validation, layout="source_synchronized", render_version=1,
                  source_manifest_sha256=source_hash, raw_jpeg_sha256=raw_digest.hexdigest(),
                  processed_h5=spec["path"] if spec else None, processed_h5_sha256=spec["sha256"] if spec else None,
                  roi_xy_size=detail["roi_xy_size"], video_sha256=digest(path))
    atomic_json(sidecar, record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--task", choices=("drawer_place", "towel"))
    parser.add_argument("--episode", help="Render one episode for layout review; does not write full collection manifest")
    args = parser.parse_args()
    cv2.setNumThreads(1)
    source = args.manifest.resolve()
    session = source.parent
    data = json.loads(source.read_text())
    source_hash = digest(source)
    jobs = []
    for task, detail in data["tasks"].items():
        if args.task and args.task != task:
            continue
        for spec in detail["splits"].values():
            if digest(Path(spec["path"])) != spec["sha256"]:
                raise ValueError(f"Processed H5 changed: {spec['path']}")
        for e in detail["episodes"]:
            if args.episode and args.episode != e["source_episode"]:
                continue
            jobs.append((task, detail, e))
    if not jobs:
        raise ValueError("No source episodes selected")
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    records = {}
    progress_path = session / "work/crop_comparison/progress.json"
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(render_one, task, detail, e, session, source_hash, ffmpeg): (task, e["source_episode"])
                   for task, detail, e in jobs}
        for future in as_completed(futures):
            key = futures[future]
            records[key] = future.result()
            print(f"[{len(records)}/{len(jobs)}] {key[0]}/{key[1]}: {records[key]['frame_count']} frames, decoded OK", flush=True)
            atomic_json(progress_path, {"complete": len(records), "total": len(jobs), "latest": list(key)})
    ordered = [records[(task, e["source_episode"])] for task, _, e in jobs]
    if not args.task and not args.episode:
        output = session / "crop_comparison_manifest.json"
        atomic_json(output, {"schema_version": 1, "source_manifest": str(source), "source_manifest_sha256": source_hash,
                             "layout": "source_synchronized", "fps": FPS, "video_count": len(ordered),
                             "all_videos_decoded": True, "total_frames": sum(r["frame_count"] for r in ordered),
                             "videos": ordered})
        print(output, flush=True)


if __name__ == "__main__":
    main()
