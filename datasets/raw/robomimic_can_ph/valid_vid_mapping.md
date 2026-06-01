# robomimic_can_ph valid 20: embd idx 到 raw demo idx 映射

**Embedding H5**: `datasets/embeddings/TCC-robomimic_can_ph-180vid_train-resnet50_conv4c-only_bn-20260508-234455/robomimic_can_ph-20vid_valid-embd.h5`  
**Processed H5**: `datasets/processed/robomimic_can_ph-20vid_valid.h5`  
**Raw video dir**: `datasets/raw/robomimic_can_ph/`

---

## 读取规则

- `embd.h5` 的视频顺序由 `/videos` 下的 key 按字典序决定，即代码实际读取的是 `sorted(f["videos"].keys())`。
- 该 `embd.h5` 中排序后的 key 为 `000001` 到 `000020`。
- 每个 `video_id` 对应的 raw demo 文件通过 `processed H5` 同名视频组的 `path` 属性恢复。
- 因此，`embd_sorted_index (0-based)`、`video_id`、`raw_demo_idx` 三者是一一对应关系。

---

## 完整映射表

| embd_sorted_index (0-based) | video_id | raw_demo_idx | raw_demo_file | embd_seq_len | raw_path |
|---:|:---:|---:|:---|---:|:---|
| 0  | 000001 | 0   | demo_0.mp4   | 118 | /home/user/zhangzk/projects/google-research/mytcc/datasets/raw/robomimic_can_ph/demo_0.mp4 |
| 1  | 000002 | 5   | demo_5.mp4   | 134 | /home/user/zhangzk/projects/google-research/mytcc/datasets/raw/robomimic_can_ph/demo_5.mp4 |
| 2  | 000003 | 14  | demo_14.mp4  | 113 | /home/user/zhangzk/projects/google-research/mytcc/datasets/raw/robomimic_can_ph/demo_14.mp4 |
| 3  | 000004 | 21  | demo_21.mp4  | 100 | /home/user/zhangzk/projects/google-research/mytcc/datasets/raw/robomimic_can_ph/demo_21.mp4 |
| 4  | 000005 | 27  | demo_27.mp4  | 124 | /home/user/zhangzk/projects/google-research/mytcc/datasets/raw/robomimic_can_ph/demo_27.mp4 |
| 5  | 000006 | 32  | demo_32.mp4  | 110 | /home/user/zhangzk/projects/google-research/mytcc/datasets/raw/robomimic_can_ph/demo_32.mp4 |
| 6  | 000007 | 36  | demo_36.mp4  | 103 | /home/user/zhangzk/projects/google-research/mytcc/datasets/raw/robomimic_can_ph/demo_36.mp4 |
| 7  | 000008 | 44  | demo_44.mp4  | 121 | /home/user/zhangzk/projects/google-research/mytcc/datasets/raw/robomimic_can_ph/demo_44.mp4 |
| 8  | 000009 | 56  | demo_56.mp4  | 101 | /home/user/zhangzk/projects/google-research/mytcc/datasets/raw/robomimic_can_ph/demo_56.mp4 |
| 9  | 000010 | 93  | demo_93.mp4  | 96  | /home/user/zhangzk/projects/google-research/mytcc/datasets/raw/robomimic_can_ph/demo_93.mp4 |
| 10 | 000011 | 103 | demo_103.mp4 | 96  | /home/user/zhangzk/projects/google-research/mytcc/datasets/raw/robomimic_can_ph/demo_103.mp4 |
| 11 | 000012 | 105 | demo_105.mp4 | 127 | /home/user/zhangzk/projects/google-research/mytcc/datasets/raw/robomimic_can_ph/demo_105.mp4 |
| 12 | 000013 | 111 | demo_111.mp4 | 106 | /home/user/zhangzk/projects/google-research/mytcc/datasets/raw/robomimic_can_ph/demo_111.mp4 |
| 13 | 000014 | 125 | demo_125.mp4 | 131 | /home/user/zhangzk/projects/google-research/mytcc/datasets/raw/robomimic_can_ph/demo_125.mp4 |
| 14 | 000015 | 131 | demo_131.mp4 | 126 | /home/user/zhangzk/projects/google-research/mytcc/datasets/raw/robomimic_can_ph/demo_131.mp4 |
| 15 | 000016 | 155 | demo_155.mp4 | 144 | /home/user/zhangzk/projects/google-research/mytcc/datasets/raw/robomimic_can_ph/demo_155.mp4 |
| 16 | 000017 | 168 | demo_168.mp4 | 103 | /home/user/zhangzk/projects/google-research/mytcc/datasets/raw/robomimic_can_ph/demo_168.mp4 |
| 17 | 000018 | 176 | demo_176.mp4 | 136 | /home/user/zhangzk/projects/google-research/mytcc/datasets/raw/robomimic_can_ph/demo_176.mp4 |
| 18 | 000019 | 180 | demo_180.mp4 | 140 | /home/user/zhangzk/projects/google-research/mytcc/datasets/raw/robomimic_can_ph/demo_180.mp4 |
| 19 | 000020 | 191 | demo_191.mp4 | 95  | /home/user/zhangzk/projects/google-research/mytcc/datasets/raw/robomimic_can_ph/demo_191.mp4 |

---

## 直接可用的结论

- `selected_video_index = 0` 对应 `video_id = 000001`，raw demo 是 `demo_0.mp4`。
- `selected_video_index = 1` 对应 `video_id = 000002`，raw demo 是 `demo_5.mp4`。
- `selected_video_index = 19` 对应 `video_id = 000020`，raw demo 是 `demo_191.mp4`。
- 如果 heatmap 文件名里出现 `00000k`，就先按上表找到 `video_id`，再映射到对应的 raw `demo_xxx.mp4`。

## 核验依据

- 目标 embedding 文件实际包含 20 个视频 key，排序结果为 `000001` 到 `000020`。
- `algos/eval_task/tcc_eval_tasks/task_latent_distance_heatmap.py` 读取轨迹时使用 `sorted(f["videos"].keys())`。
- `datasets/processed/robomimic_can_ph-20vid_valid.h5` 中同名视频组的 `path` 属性给出了原始 MP4 绝对路径。
