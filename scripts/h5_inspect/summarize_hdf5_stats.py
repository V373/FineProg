"""Summarize aggregate statistics of the robomimic HDF5 dataset."""
import h5py
import numpy as np
import collections

path = "/home/user/zhangzk/projects/fineprog/third_party/robomimic/datasets/can/mh/image_v15.hdf5"

with h5py.File(path, "r") as f:
    demo_keys = list(f["data"].keys())
    n_demos = len(demo_keys)
    demo_ids = sorted(int(k.split("_")[1]) for k in demo_keys)
    print(f"Total demos: {n_demos}")
    print(f"Demo id range: {demo_ids[0]} ... {demo_ids[-1]}")
    print(f"Demo ids are contiguous: {demo_ids == list(range(demo_ids[0], demo_ids[-1]+1))}")

    # Step counts
    steps = [f[f"data/{k}/actions"].shape[0] for k in demo_keys]
    print(f"\nSteps per demo:")
    print(f"  min: {min(steps)}, max: {max(steps)}, mean: {np.mean(steps):.2f}, median: {np.median(steps):.0f}")
    print(f"  total timesteps: {sum(steps)}")
    print(f"  step distribution (counter): {dict(sorted(collections.Counter(steps).items()))}")

    # Reward statistics
    all_rewards = np.concatenate([f[f"data/{k}/rewards"][...] for k in demo_keys])
    print(f"\nRewards:")
    print(f"  shape total: {all_rewards.shape}")
    print(f"  min: {all_rewards.min()}, max: {all_rewards.max()}, mean: {all_rewards.mean():.4f}")
    print(f"  nonzero count: {(all_rewards != 0).sum()}  ({(all_rewards != 0).mean()*100:.2f}%)")
    print(f"  unique non-zero: {np.unique(all_rewards[all_rewards != 0])}")

    # Final-step dones (terminal signal)
    last_dones = [f[f"data/{k}/dones"][...][-1] for k in demo_keys]
    print(f"\nFinal dones: min={min(last_dones)}, max={max(last_dones)}, "
          f"sum={sum(last_dones)}/{n_demos} end with done=1")

    # Action summary across whole dataset
    all_actions = np.concatenate([f[f"data/{k}/actions"][...] for k in demo_keys], axis=0)
    print(f"\nActions overall shape: {all_actions.shape}")
    print(f"  per-dim min: {all_actions.min(0)}")
    print(f"  per-dim max: {all_actions.max(0)}")
    print(f"  per-dim mean: {np.round(all_actions.mean(0), 4)}")

    # All unique obs sub-keys (should be consistent across demos)
    obs_keys_per_demo = [sorted(f[f"data/{k}/obs"].keys()) for k in demo_keys]
    unique_obs_key_sets = {tuple(ks) for ks in obs_keys_per_demo}
    print(f"\nDistinct obs key sets across all demos: {len(unique_obs_key_sets)}")
    for ks in unique_obs_key_sets:
        print(f"  {ks}")

    # Check action dim and obs dimensions (sum low-dim)
    one = demo_keys[0]
    print(f"\nFor {one}:")
    print(f"  actions shape: {f[f'data/{one}/actions'].shape}")
    obs_lowdim_keys = [k for k in f[f"data/{one}/obs"].keys() if k != "agentview_image"]
    total_lowdim = sum(int(np.prod(f[f'data/{one}/obs/{k}'].shape[1:])) for k in obs_lowdim_keys)
    print(f"  obs low-dim keys ({len(obs_lowdim_keys)}): {obs_lowdim_keys}")
    print(f"  total low-dim obs dim: {total_lowdim}")
    img_keys = [k for k in f[f"data/{one}/obs"].keys() if k == "agentview_image"]
    if img_keys:
        img_shape = f[f"data/{one}/obs/agentview_image"].shape[1:]
        print(f"  image obs: agentview_image  shape per step: {img_shape}  (H,W,C)  per-step pixels: {int(np.prod(img_shape))}")
