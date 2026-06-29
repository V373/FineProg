"""Activation-map evaluation task for the current TCC encoder.

This task is intentionally narrow: it re-runs a trained encoder on a processed
video H5 and exports two per-target-frame activation maps:

1. backbone Conv4c activations from encoder.backbone
2. temporal activations from the second Conv3D inside encoder.temporal_embedder
"""

from __future__ import annotations

from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from fineprog.algos.eval_task.base_task import BaseTask
from fineprog.dataset_preparation.h5vid_dataset import H5VideoDataset, collate_fn
from fineprog.models.encoder import TCCEncoder


class ActivationMapTask(BaseTask):
    """Export backbone and temporal activation overlays for one dataset."""

    def __init__(self):
        super().__init__(task_name="activation_map", downstream_task=False)
        self.config: dict = {}

    def configure(self, config: dict) -> None:
        self.config = dict(config)

    def evaluate(self, embeddings_dataset=None) -> dict:
        dataset_h5_path = self.config["dataset_h5_path"]
        checkpoint_path = self.config["checkpoint_path"]
        device_name = self.config.get("device", "cuda")
        device = torch.device(device_name if device_name == "cpu" or torch.cuda.is_available() else "cpu")
        context_size = int(self.config.get("context_size", 2))
        batch_size = 1
        num_workers = int(self.config.get("num_workers", 0))
        max_videos = self.config.get("max_videos")
        max_videos = None if max_videos in (None, "all") else int(max_videos)
        output_dir = Path(self.config.get("output_dir", "outputs/activation_map"))
        if not output_dir.is_absolute():
            output_dir = Path(__file__).resolve().parents[3] / output_dir

        print(f"[activation_map] loaded checkpoint path: {checkpoint_path}")
        print(f"[activation_map] dataset_h5_path: {dataset_h5_path}")

        dataset = H5VideoDataset(
            h5_path=dataset_h5_path,
            clip_len=int(self.config.get("clip_len", 20)),
            context_size=context_size,
            context_stride=int(self.config.get("context_stride", 15)),
            sample_all=True,
            sample_all_stride=int(self.config.get("sample_all_stride", 1)),
            transport_frames_as_uint8=False,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            collate_fn=collate_fn,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=(num_workers > 0),
        )
        print(f"[activation_map] number of videos: {len(dataset)}")

        encoder = TCCEncoder(
            clip_len=int(self.config.get("clip_len", 20)),
            context_size=context_size,
            pretrained=False,
            debug=False,
        )
        encoder = encoder.to(device)
        self._load_checkpoint(encoder, checkpoint_path, device)
        encoder.eval()

        conv3d_modules = [m for m in encoder.temporal_embedder.modules() if isinstance(m, nn.Conv3d)]
        if len(conv3d_modules) < 2:
            raise RuntimeError(
                f"[activation_map] Expected at least 2 Conv3d modules, got {len(conv3d_modules)}"
            )

        hook_cache: dict[str, torch.Tensor] = {}
        printed_first_batch = False
        printed_backbone_shape = False
        printed_temporal_shape = False
        printed_backbone_heatmap_shape = False
        printed_temporal_heatmap_shape = False

        def _backbone_hook(_module, _inputs, output):
            hook_cache["backbone"] = output.detach()

        def _temporal_conv2_hook(_module, _inputs, output):
            hook_cache["temporal_conv2"] = output.detach()

        backbone_handle = encoder.backbone.register_forward_hook(_backbone_hook)
        temporal_handle = conv3d_modules[1].register_forward_hook(_temporal_conv2_hook)

        total_processed_frames = 0
        total_processed_videos = 0

        try:
            with torch.no_grad():
                for batch_idx, batch in enumerate(dataloader):
                    if max_videos is not None and total_processed_videos >= max_videos:
                        break

                    frames = batch["frames"]
                    target_steps = batch["target_steps"]
                    seq_len = int(batch["seq_len"][0])
                    action_id = int(batch["action_id"][0])
                    video_id = str(batch["video_id"][0])

                    if frames.shape[0] != 1:
                        raise RuntimeError(
                            f"[activation_map] Expected batch_size=1 in sample_all mode, got frames shape {tuple(frames.shape)}"
                        )

                    if not printed_first_batch:
                        print(f"[activation_map] first batch frames shape: {tuple(frames.shape)}")
                        print(
                            f"[activation_map] first batch frames min/max: "
                            f"{float(frames.min()):.6f} / {float(frames.max()):.6f}"
                        )
                        printed_first_batch = True

                    backbone_heatmaps, temporal_heatmaps = self._run_encoder_chunked(
                        encoder=encoder,
                        frames=frames,
                        device=device,
                        context_size=context_size,
                        hook_cache=hook_cache,
                        print_backbone_shape=not printed_backbone_shape,
                        print_temporal_shape=not printed_temporal_shape,
                        print_backbone_heatmap_shape=not printed_backbone_heatmap_shape,
                        print_temporal_heatmap_shape=not printed_temporal_heatmap_shape,
                    )
                    printed_backbone_shape = True
                    printed_temporal_shape = True
                    printed_backbone_heatmap_shape = True
                    printed_temporal_heatmap_shape = True

                    output_root = self._build_video_output_root(output_dir, checkpoint_path, dataset_h5_path, video_id)
                    output_root.mkdir(parents=True, exist_ok=True)

                    rgb_frames = self._extract_rgb_frames(frames[0], context_size)
                    target_steps_np = target_steps[0].detach().cpu().numpy().astype(np.int64)
                    fps = self._read_video_fps(dataset_h5_path, video_id)
                    print(f"[activation_map] video_id={video_id} render_fps={fps:.4f}")

                    backbone_overlays = self._save_outputs_for_stream(
                        stream_name="backbone",
                        heatmaps=backbone_heatmaps,
                        rgb_frames=rgb_frames,
                        target_steps=target_steps_np,
                        output_root=output_root,
                    )
                    temporal_overlays = self._save_outputs_for_stream(
                        stream_name="temporal_conv2",
                        heatmaps=temporal_heatmaps,
                        rgb_frames=rgb_frames,
                        target_steps=target_steps_np,
                        output_root=output_root,
                    )

                    if bool(self.config.get("save_h5", True)):
                        self._save_h5(
                            output_root=output_root,
                            video_id=video_id,
                            target_steps=target_steps_np,
                            backbone_heatmaps=backbone_heatmaps,
                            temporal_heatmaps=temporal_heatmaps,
                            seq_len=seq_len,
                            action_id=action_id,
                            checkpoint_path=checkpoint_path,
                            dataset_h5_path=dataset_h5_path,
                            context_size=context_size,
                        )

                    if bool(self.config.get("save_mp4", True)):
                        self._save_mp4(output_root / "videos" / "backbone_overlay.mp4", backbone_overlays, fps)
                        self._save_mp4(output_root / "videos" / "temporal_conv2_overlay.mp4", temporal_overlays, fps)

                    total_processed_frames += int(backbone_heatmaps.shape[0])
                    total_processed_videos += 1
                    print(f"[activation_map] final output directory: {output_root}")
                    print(f"[activation_map] processed video {batch_idx}: {video_id}, frames={backbone_heatmaps.shape[0]}")
        finally:
            backbone_handle.remove()
            temporal_handle.remove()

        return {
            "task_name": "activation_map",
            "metric_name": "num_processed_frames",
            "metric_value": float(total_processed_frames),
            "num_processed_videos": total_processed_videos,
            "output_dir": str(output_dir),
        }

    def _run_encoder_chunked(
        self,
        encoder: TCCEncoder,
        frames: torch.Tensor,
        device: torch.device,
        context_size: int,
        hook_cache: dict[str, torch.Tensor],
        print_backbone_shape: bool,
        print_temporal_shape: bool,
        print_backbone_heatmap_shape: bool,
        print_temporal_heatmap_shape: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        frames = frames.to(device)
        total_steps = frames.shape[1]
        frames_per_batch = int(encoder.clip_len)
        backbone_chunks: list[np.ndarray] = []
        temporal_chunks: list[np.ndarray] = []

        for start in range(0, total_steps, frames_per_batch):
            end = min(start + frames_per_batch, total_steps)
            chunk = frames[:, start:end, ...]
            actual_len = end - start
            if actual_len < frames_per_batch:
                pad_len = frames_per_batch - actual_len
                pad = chunk[:, -1:, ...].expand(-1, pad_len, -1, -1, -1, -1)
                chunk = torch.cat([chunk, pad], dim=1)

            hook_cache.clear()
            _ = encoder(chunk)

            if "backbone" not in hook_cache:
                raise RuntimeError("[activation_map] backbone hook did not capture any output")
            if "temporal_conv2" not in hook_cache:
                raise RuntimeError("[activation_map] temporal conv2 hook did not capture any output")

            backbone_out = hook_cache["backbone"]
            temporal_out = hook_cache["temporal_conv2"]

            if print_backbone_shape:
                print(f"[activation_map] backbone hook output shape: {tuple(backbone_out.shape)}")
                print_backbone_shape = False
            if print_temporal_shape:
                print(f"[activation_map] temporal conv2 hook output shape: {tuple(temporal_out.shape)}")
                print_temporal_shape = False

            backbone_heatmaps = self._reduce_backbone(backbone_out, 1, frames_per_batch, context_size)[:actual_len]
            temporal_heatmaps = self._reduce_temporal_conv2(temporal_out, 1, frames_per_batch, context_size)[:actual_len]

            if print_backbone_heatmap_shape:
                print(f"[activation_map] generated backbone heatmap shape: {tuple(backbone_heatmaps.shape)}")
                print_backbone_heatmap_shape = False
            if print_temporal_heatmap_shape:
                print(f"[activation_map] generated temporal_conv2 heatmap shape: {tuple(temporal_heatmaps.shape)}")
                print_temporal_heatmap_shape = False

            backbone_chunks.append(backbone_heatmaps)
            temporal_chunks.append(temporal_heatmaps)

        return np.concatenate(backbone_chunks, axis=0), np.concatenate(temporal_chunks, axis=0)

    def _reduce_backbone(
        self,
        backbone_out: torch.Tensor,
        batch_size: int,
        clip_len: int,
        context_size: int,
    ) -> np.ndarray:
        if backbone_out.ndim != 4:
            raise RuntimeError(
                f"[activation_map] Unexpected backbone hook shape {tuple(backbone_out.shape)}; expected 4D [B*T*context, C, H, W]"
            )
        expected_n = batch_size * clip_len * context_size
        if backbone_out.shape[0] != expected_n or backbone_out.shape[1:] != (1024, 14, 14):
            raise RuntimeError(
                f"[activation_map] Unexpected backbone hook shape {tuple(backbone_out.shape)}; "
                f"expected ({expected_n}, 1024, 14, 14)"
            )
        x = backbone_out.reshape(batch_size, clip_len, context_size, 1024, 14, 14)
        x = x[:, :, context_size - 1, :, :, :]
        x = torch.mean(torch.abs(x), dim=2)
        x = self._normalize_per_frame(x)
        return x[0].detach().cpu().numpy().astype(np.float32)

    def _reduce_temporal_conv2(
        self,
        temporal_out: torch.Tensor,
        batch_size: int,
        clip_len: int,
        context_size: int,
    ) -> np.ndarray:
        if temporal_out.ndim != 5:
            raise RuntimeError(
                f"[activation_map] Unexpected temporal conv2 hook shape {tuple(temporal_out.shape)}; expected 5D [B*T, C, context, H, W]"
            )
        expected_n = batch_size * clip_len
        if temporal_out.shape[0] != expected_n or temporal_out.shape[2] != context_size or temporal_out.shape[3:] != (14, 14):
            raise RuntimeError(
                f"[activation_map] Unexpected temporal conv2 hook shape {tuple(temporal_out.shape)}; "
                f"expected ({expected_n}, C, {context_size}, 14, 14)"
            )
        x = temporal_out.reshape(batch_size, clip_len, temporal_out.shape[1], temporal_out.shape[2], 14, 14)
        x = torch.abs(x)
        x = torch.mean(x, dim=2)
        x = torch.mean(x, dim=2)
        x = self._normalize_per_frame(x)
        return x[0].detach().cpu().numpy().astype(np.float32)

    def _normalize_per_frame(self, x: torch.Tensor) -> torch.Tensor:
        mins = x.amin(dim=(-2, -1), keepdim=True)
        maxs = x.amax(dim=(-2, -1), keepdim=True)
        denom = maxs - mins
        constant_mask = denom <= 1.0e-8
        if bool(constant_mask.any().item()):
            n_constant = int(constant_mask.sum().item())
            print(f"[activation_map] warning: {n_constant} heatmaps were near-constant; emitting zeros")
        y = (x - mins) / (denom + 1.0e-8)
        return torch.where(constant_mask, torch.zeros_like(y), y)

    def _extract_rgb_frames(self, frames: torch.Tensor, context_size: int) -> np.ndarray:
        rgb = frames[:, context_size - 1].detach().cpu().numpy()
        rgb = np.transpose(rgb, (0, 2, 3, 1))
        if rgb.min() < 0.0 or rgb.max() > 1.5:
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            rgb = rgb * std[None, None, None, :] + mean[None, None, None, :]
        rgb = np.clip(rgb, 0.0, 1.0)
        return (rgb * 255.0).astype(np.uint8)

    def _save_outputs_for_stream(
        self,
        stream_name: str,
        heatmaps: np.ndarray,
        rgb_frames: np.ndarray,
        target_steps: np.ndarray,
        output_root: Path,
    ) -> list[np.ndarray]:
        overlays: list[np.ndarray] = []
        stream_dir = output_root / stream_name
        if bool(self.config.get("save_png", True)):
            stream_dir.mkdir(parents=True, exist_ok=True)

        resize_to = int(self.config.get("resize_to", 224))
        alpha = float(self.config.get("overlay_alpha", 0.45))
        color_code = getattr(cv2, f"COLORMAP_{str(self.config.get('colormap', 'jet')).upper()}", cv2.COLORMAP_JET)

        for frame_idx, target_step in enumerate(target_steps.tolist()):
            heatmap = heatmaps[frame_idx]
            heatmap_u8 = np.clip(np.round(heatmap * 255.0), 0, 255).astype(np.uint8)
            heatmap_resized = cv2.resize(heatmap_u8, (resize_to, resize_to), interpolation=cv2.INTER_LINEAR)
            heatmap_color = cv2.applyColorMap(heatmap_resized, color_code)

            rgb = rgb_frames[frame_idx]
            if rgb.shape[0] != resize_to or rgb.shape[1] != resize_to:
                rgb = cv2.resize(rgb, (resize_to, resize_to), interpolation=cv2.INTER_LINEAR)
            rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            overlay = cv2.addWeighted(rgb_bgr, 1.0 - alpha, heatmap_color, alpha, 0.0)
            overlays.append(overlay)

            if bool(self.config.get("save_png", True)):
                out_path = stream_dir / f"frame_{int(target_step):06d}.png"
                cv2.imwrite(str(out_path), overlay)

        return overlays

    def _save_h5(
        self,
        output_root: Path,
        video_id: str,
        target_steps: np.ndarray,
        backbone_heatmaps: np.ndarray,
        temporal_heatmaps: np.ndarray,
        seq_len: int,
        action_id: int,
        checkpoint_path: str,
        dataset_h5_path: str,
        context_size: int,
    ) -> None:
        raw_h5_dir = output_root / "raw_h5"
        raw_h5_dir.mkdir(parents=True, exist_ok=True)
        out_path = raw_h5_dir / "activations.h5"
        with h5py.File(out_path, "w") as f:
            videos_grp = f.create_group("videos")
            grp = videos_grp.create_group(video_id)
            grp.create_dataset("target_steps", data=target_steps.astype(np.int64))
            grp.create_dataset("backbone_heatmaps", data=backbone_heatmaps.astype(np.float32))
            grp.create_dataset("temporal_conv2_heatmaps", data=temporal_heatmaps.astype(np.float32))
            grp.attrs["seq_len"] = seq_len
            grp.attrs["action_id"] = action_id
            grp.attrs["checkpoint_path"] = checkpoint_path
            grp.attrs["dataset_h5_path"] = dataset_h5_path
            grp.attrs["context_size"] = context_size

    def _save_mp4(self, output_path: Path, frames_bgr: list[np.ndarray], fps: float) -> None:
        if not frames_bgr:
            return
        output_path.parent.mkdir(parents=True, exist_ok=True)
        safe_fps = float(fps) if float(fps) > 1.0e-6 else 1.0

        # Prefer ffmpeg/libx264 (same strategy as expert_projection) for
        # maximum player compatibility in VS Code.
        try:
            import imageio  # noqa: PLC0415

            writer = imageio.get_writer(
                str(output_path),
                fps=safe_fps,
                format="ffmpeg",
                codec="libx264",
                macro_block_size=1,
            )
            try:
                for frame_bgr in frames_bgr:
                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    writer.append_data(frame_rgb)
            finally:
                writer.close()
            return
        except Exception as exc:
            print(f"[activation_map] imageio ffmpeg writer failed ({exc}); falling back to OpenCV mp4v")

        height, width = frames_bgr[0].shape[:2]
        writer_cv2 = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            safe_fps,
            (width, height),
        )
        if not writer_cv2.isOpened():
            raise RuntimeError(f"[activation_map] Failed to open VideoWriter for {output_path}")
        try:
            for frame in frames_bgr:
                writer_cv2.write(frame)
        finally:
            writer_cv2.release()

    def _build_video_output_root(
        self,
        output_dir: Path,
        checkpoint_path: str,
        dataset_h5_path: str,
        video_id: str,
    ) -> Path:
        ckpt_path = Path(checkpoint_path)
        run_or_ckpt_stem = ckpt_path.parent.name if ckpt_path.parent.name else ckpt_path.stem
        dataset_stem = Path(dataset_h5_path).stem
        return output_dir / run_or_ckpt_stem / dataset_stem / video_id

    def _read_video_fps(self, dataset_h5_path: str, video_id: str) -> float:
        with h5py.File(dataset_h5_path, "r") as f:
            grp = f["videos"][video_id]
            raw_video_path = str(grp.attrs.get("path", ""))
            h5_fps = float(grp.attrs.get("fps", 1.0))

        # Prefer FPS measured from the source video to preserve original
        # (potentially non-integer) frame rate.
        if raw_video_path and Path(raw_video_path).exists():
            cap = cv2.VideoCapture(raw_video_path)
            try:
                fps = float(cap.get(cv2.CAP_PROP_FPS))
            finally:
                cap.release()
            if fps > 1.0e-6:
                return fps

        if h5_fps > 1.0e-6:
            return h5_fps
        return 1.0

    def _load_checkpoint(
        self,
        model: torch.nn.Module,
        checkpoint_path: str,
        device: torch.device,
    ) -> None:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if isinstance(checkpoint, dict):
            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        cleaned_state_dict = {}
        for key, value in state_dict.items():
            if not isinstance(value, torch.Tensor):
                continue
            cleaned_key = key[7:] if key.startswith("module.") else key
            cleaned_state_dict[cleaned_key] = value

        missing_keys, unexpected_keys = model.load_state_dict(cleaned_state_dict, strict=False)
        print(f"[activation_map] loaded checkpoint path: {checkpoint_path}")
        print(f"[activation_map] missing_keys: {missing_keys}")
        print(f"[activation_map] unexpected_keys: {unexpected_keys}")