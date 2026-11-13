#!/usr/bin/env python3
"""Render a compact, source-linked cleaning report; no model evaluation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import pickle
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args()
    session = args.manifest.resolve().parent
    for key, relative in (("MPLCONFIGDIR", "cache/matplotlib"), ("TMPDIR", "tmp")):
        directory = session / relative
        directory.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(directory)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    import numpy as np
    from PIL import Image

    sys.path.insert(0, str(PROJECT_ROOT))
    from dataset_preparation.prepare_real_progress_data import read_rgb

    manifest = json.loads(args.manifest.read_text())
    visuals = session / "visuals"
    visuals.mkdir(exist_ok=True)

    def original(task, episode, index):
        return PROJECT_ROOT / "datasets/raw" / task / "collect_data" / episode / "topFullImg" / f"{index}.jpg"

    def spatial(task, detail):
        keep = [e for e in detail["episodes"] if e["status"] == "keep"]
        # Distinct acquisition positions and progress phases.
        picks = [keep[0], keep[len(keep) // 2], keep[-1]]
        fig, axes = plt.subplots(3, 3, figsize=(12, 10))
        x, y, size = detail["roi_xy_size"]
        for row, e in enumerate(picks):
            idx = [e["start_frame"], (e["start_frame"] + e["end_frame"]) // 2, e["completion_frame"]][row]
            path = original(task, e["source_episode"], idx)
            raw = np.asarray(Image.open(path).convert("RGB"))
            axes[row, 0].imshow(raw)
            axes[row, 0].add_patch(Rectangle((x, y), size, size, fill=False, edgecolor="#00ef66", linewidth=2))
            axes[row, 0].set_title(f"{e['source_episode']} / raw {idx}\n640 x 480; ROI ({x},{y},{size},{size})", fontsize=9)
            axes[row, 1].imshow(raw[y:y + size, x:x + size])
            axes[row, 1].set_title(f"Square crop: {size} x {size}")
            axes[row, 2].imshow(read_rgb(path, detail["roi_xy_size"]))
            axes[row, 2].set_title("Training RGB: 224 x 224")
            for ax in axes[row]:
                ax.axis("off")
        fig.suptitle(f"{task}: fixed spatial crop (start / middle / completion)")
        fig.tight_layout()
        fig.savefig(visuals / f"spatial_{task}.png", dpi=130)
        plt.close(fig)

    def temporal(task, detail, e, suffix):
        root = Path(e["source_path"])
        q, gripper = [], []
        for index in range(e["source_frames"]):
            with (root / "observation" / f"{index}.pkl").open("rb") as handle:
                obs = pickle.load(handle)
            q.append(obs["joint_positions"])
            gripper.append(obs["gripper_position"])
        q = np.asarray(q)
        gripper = np.asarray(gripper)
        velocity = np.r_[0, np.max(np.abs(np.diff(q[:, [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]], axis=0)), axis=1)]
        s, c, end, n = (e[k] for k in ("start_frame", "completion_frame", "end_frame", "source_frames"))
        prefix_peak = int(np.argmax(velocity[:s])) if s else 0
        points = [0, prefix_peak, max(0, s - 1), s, max(s, c - 10), c, end, min(n - 1, end + 15)]
        labels = ["Recording start", "Prefix motion peak", "Just before clip", "Selected start",
                  "Before completion", "Semantic completion", "Selected end", "After clip"]
        fig = plt.figure(figsize=(18, 7.6))
        gs = fig.add_gridspec(3, 8, height_ratios=[1.35, .7, .7])
        for col, (idx, title) in enumerate(zip(points, labels)):
            ax = fig.add_subplot(gs[0, col])
            ax.imshow(read_rgb(original(task, e["source_episode"], idx), detail["roi_xy_size"]))
            inside = s <= idx <= end
            mapped = f"processed {idx - s}" if inside else "removed"
            ax.set_title(f"{title}\nraw {idx} / {mapped}", fontsize=9)
            ax.axis("off")
        for row, data, ylabel in ((1, velocity, "Max measured joint delta\n(rad / source frame)"), (2, gripper, "Measured gripper\n(source units)")):
            ax = fig.add_subplot(gs[row, :])
            if row == 2:
                ax.plot(data[:, 0], label="left", linewidth=1)
                ax.plot(data[:, 1], label="right", linewidth=1)
            else:
                ax.plot(data, color="#405070", linewidth=1)
            ax.axvspan(s, end, color="#d3efcf", alpha=.65, label="retained interval")
            for value, name, color, style in ((s, "start", "#159447", "--"), (c, "completion", "#ce7600", ":"), (end, "end", "#c72929", "--")):
                ax.axvline(value, color=color, linestyle=style, label=f"{name}={value}", linewidth=1)
            ax.set_ylabel(ylabel, fontsize=9)
            ax.set_xlim(0, n - 1)
            ax.grid(alpha=.2)
            ax.legend(loc="upper right", ncol=5, fontsize=8)
        ax.set_xlabel("Original zero-based frame index (not measured wall-clock time)")
        fig.suptitle(f"{task} / {e['source_episode']} | retain [{s},{end}] inclusive | completion={c}")
        fig.tight_layout()
        filename = f"temporal_{task}_{suffix}.png"
        fig.savefig(visuals / filename, dpi=130)
        plt.close(fig)
        return filename

    def exclusions(manifest):
        excluded = [(task, e) for task, d in manifest["tasks"].items() for e in d["episodes"] if e["status"] != "keep"]
        fig, axes = plt.subplots(len(excluded), 3, figsize=(10, 2.8 * len(excluded)), squeeze=False)
        for row, (task, e) in enumerate(excluded):
            for col, idx in enumerate((0, e["source_frames"] // 2, e["source_frames"] - 1)):
                axes[row, col].imshow(read_rgb(original(task, e["source_episode"], idx), manifest["tasks"][task]["roi_xy_size"]))
                axes[row, col].set_title(f"{task} / {e['source_episode']}\nraw frame {idx}", fontsize=9)
                axes[row, col].axis("off")
        fig.suptitle("Excluded demonstrations: initial / middle / final source images")
        fig.tight_layout()
        fig.savefig(visuals / "excluded.png", dpi=125)
        plt.close(fig)

    media = {}
    rows = []
    details_sections = []
    for task, detail in manifest["tasks"].items():
        spatial(task, detail)
        keep = [e for e in detail["episodes"] if e["status"] == "keep"]
        # Show a typical episode and the largest removed prefix, not cherry-picked
        # model performance (there is no model execution in this report).
        highlight_id = "20260924223006" if task == "towel" else "episode_init_2"
        typical = next((e for e in keep if e["source_episode"] == highlight_id), keep[0])
        longest_prefix = max(keep, key=lambda e: e["start_frame"])
        media[task] = [temporal(task, detail, typical, "typical")]
        if longest_prefix is not typical:
            media[task].append(temporal(task, detail, longest_prefix, "long_prefix"))
        raw_frames = sum(e["source_frames"] for e in detail["episodes"])
        retained = sum(e["retained_frames"] for e in keep)
        prefixes = sum(e["start_frame"] for e in keep)
        tails = sum(e["source_frames"] - e["end_frame"] - 1 for e in keep)
        splits = detail["splits"]
        rows.append(f"| {task} | {len(detail['episodes'])} / {raw_frames:,} | {len(keep)} | {len(detail['episodes']) - len(keep)} | 36 / 4 / {splits.get('calibration', {}).get('count', 0)} | {retained:,} | {prefixes:,} / {tails:,} |")
        lines = [f"# {task} 逐条清洗记录", "", "所有帧号均为原始零基闭区间；状态 `review` 与 `exclude` 均不进入任何成功数据集。", "",
                 "| 源 episode | 原始帧数 | 划分 | 起点 | 语义完成 | 终点 | 保留帧数 | 理由 |", "|---|---:|---|---:|---:|---:|---:|---|"]
        for e in detail["episodes"]:
            lines.append(f"| {e['source_episode']} | {e['source_frames']} | {e.get('split') or e['status']} | {e.get('start_frame')} | {e.get('completion_frame')} | {e.get('end_frame')} | {e['retained_frames']} | {e.get('reason', '')} |")
        lines.extend(["", "## 边界审核依据", ""])
        for e in detail["episodes"]:
            lines.extend([f"### {e['source_episode']}", "", f"- 起点：{e.get('start_reason', '不适用')}",
                          f"- 终点：{e.get('end_reason', '不适用')}", f"- 核验：{e.get('evidence', '')}", ""])
            evidence_files = e.get("evidence_files", [])
            if evidence_files:
                evidence_files = [str(PROJECT_ROOT / p) if p.startswith("datasets/") else p
                                  for p in evidence_files]
                lines.extend(["图像证据：" + " · ".join(f"[{Path(p).stem}]({p})" for p in evidence_files), ""])
        (session / f"{task}_review.md").write_text("\n".join(lines) + "\n")
        details_sections.append(f"[{task} 逐条记录]({task}_review.md)")
    exclusions(manifest)

    excluded_lines = [f"- `{task}/{e['source_episode']}`：{e['reason']}" for task, d in manifest["tasks"].items() for e in d["episodes"] if e["status"] != "keep"]
    retry_links = []
    for stem, label in (("detail_28_right_retries.jpg", "右臂反复抓取窗口"),
                        ("detail_28_left_retries.jpg", "左臂反复抓取窗口")):
        if (visuals / "towel" / stem).exists():
            retry_links.append(f"[{label}](visuals/towel/{stem})")
    dataset_lines = []
    for task, d in manifest["tasks"].items():
        for split, spec in d["splits"].items():
            dataset_lines.append(f"- `{task}/{split}`：[{Path(spec['path']).name}]({spec['path']})，{spec['count']}条、{spec['frames']:,}帧。")
    command = ("conda run --no-capture-output -n fineprog python -u \\\n"
               "  scripts/train_real_single_frame_baselines_serial.py \\\n"
               f"  --dataset-manifest {args.manifest.resolve()} \\\n"
               f"  --output-root {session / 'training'} \\\n"
               "  --gpu 0")
    report = f"""# 真机 progress 数据清洗简报

本轮仅完成数据清洗、划分、processed H5 和训练入口准备；没有启动训练，没有生成 checkpoint。

## 结果

| 任务 | 原始条数 / 帧数 | 合格示范 | 隔离 | train / valid / cali | 保留总帧数 | 合格示范裁去头 / 尾帧数 |
|---|---:|---:|---:|---|---:|---:|
{chr(10).join(rows)}

按源 episode 排序后，分别用 Python `random.Random(42).shuffle` 划分 train36、valid4、剩余 calibration。异常不进入 calibration。原始 HDF5、JPEG、MP4 是同一轨迹的不同表示，只登记一次。**没有 duplicate-frame filter，没有删除中间帧或重采样**。

全量可核对记录：{' · '.join(details_sections)} · [CSV](data_manifest.csv) · [完整 manifest](data_manifest.json)。

## 起止口径与核验

起点根据实测关节/末端运动和邻近图像，取正式接近前最后短暂停顿或持续接近开始，避免把录制后短促调整当任务开始。未观察到独立首次抖动的轨迹不固定删除前N帧。终点取最早清楚完成任务的帧，加2帧（最多3帧），不等回位；允许查看后续帧确认成功，但不将确认窗口全部留入数据。

drawer 的完成条件为玩偶放入并关闭抽屉；towel 为目标折叠完成且松手。逐条检查状态曲线、首/中/尾抽样和边界附近图像；这不是对每一帧的人工语义标注。可读性和索引一致性另外做程序检查。

drawer 部分闭合画面被腕部遮挡，结合右腕图像、前板边缘和实测推入位置估计完成帧，存在数帧定位不确定性。`完成+2帧`是相对审核边界的截取规则，不是物理传感器真值保证。

两任务 `joint_velocities` 实际等于 `joint_positions`，towel `ee_pos_quat` 全零，均未用于边界判断。使用实测 qpos 差分、eef_pose 与夹爪；指令/自动事件只辅助。图像与状态按整数帧号对齐；没有逐帧时间戳，20 FPS仅为名义值。

![drawer 时间裁剪](visuals/{media['drawer_place'][0]})

![towel 时间裁剪](visuals/{media['towel'][0]})

更多边界示例：{' · '.join(f'[{task} 长前段示例](visuals/{items[1]})' for task, items in media.items() if len(items) > 1)}。

## towel 空间裁剪示例

来源为640×480的 `topFullImg`。固定裁 `(x=80,y=0,w=480,h=480)` 后等比例缩小到224×224；绿色框显示保留范围。三行分别展示不同采集时段的起点、中段和完成状态。

![towel 方形裁剪](visuals/spatial_towel.png)

drawer 沿用 `(x=256,y=168,w=224,h=224)`：[查看空间裁剪示例](visuals/spatial_drawer_place.png)。

## 隔离示范

{chr(10).join(excluded_lines)}

929帧示范的排除依据是明显重试而非长度：{' · '.join(retry_links)}。

drawer 20仅按疑似卡边隔离，不断言物理失败：[终态近景证据](visuals/drawer_place/excluded_episode20.jpg)。

![隔离示范首中尾](visuals/excluded.png)

## 数据与验证

{chr(10).join(dataset_lines)}

H5为 `/videos/<id>/frames`，RGB uint8、224×224；逐帧保存原始索引。`T=end-start+1`，processed第i帧对应原始start+i。完整H5读取、checksum、来源集合无交集、每条5个位置与原始裁剪RGB对应检查见 [data_validation.json](data_validation.json)；训练读取器/batch检查见 [loader_validation.json](loader_validation.json)。

## 训练调用（尚未执行）

在 `/home/user/zhangzk/projects/fineprog` 执行以下命令，依次训练 drawer local→frozen HF1→towel local→frozen HF1，各20k epochs。HF1加载同任务刚完成的local。配方沿用can/square，仅做valid Kendall数值监控；本轮只进行了dry-run配置检查。

```bash
{command}
```

输出目录必须为新目录；重复实验请换新的 `--output-root`。本轮生成的 [dry-run配置及manifest](launcher_dry_run/manifest.json) 可直接审阅。
"""
    (session / "README.md").write_text(report)
    print(session / "README.md")


if __name__ == "__main__":
    main()
