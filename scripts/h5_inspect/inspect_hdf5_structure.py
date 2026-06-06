"""Inspect the structure of a robomimic HDF5 dataset."""
import h5py
import numpy as np

path = "/home/user/zhangzk/projects/fineprog/third_party/robomimic/datasets/can/mh/image_v15.hdf5"

def walk(g, prefix=""):
    """Recursively print the structure of an h5py Group/Dataset."""
    for key in g.keys():
        item = g[key]
        p = f"{prefix}/{key}" if prefix else key
        if isinstance(item, h5py.Group):
            print(f"GROUP   {p}")
            walk(item, p)
        else:
            ds = item
            print(f"DATASET {p}  shape={ds.shape}  dtype={ds.dtype}")

with h5py.File(path, "r") as f:
    print("=" * 80)
    print(f"FILE: {path}")
    print(f"FILE attrs: {dict(f.attrs)}")
    print("=" * 80)
    print("\n--- TREE ---")
    walk(f)

    print("\n--- TOP-LEVEL GROUPS ---")
    for key in f.keys():
        item = f[key]
        if isinstance(item, h5py.Group):
            print(f"\n[{key}]  attrs: {dict(item.attrs)}")
            for sub in item.keys():
                sub_item = item[sub]
                if isinstance(sub_item, h5py.Dataset):
                    print(f"  - {sub}: shape={sub_item.shape} dtype={sub_item.dtype}")
                else:
                    print(f"  - {sub}/ (group)")

    if "data" in f:
        demo_keys = list(f["data"].keys())
        print(f"\n--- DEMOS ---")
        print(f"Total demos: {len(demo_keys)}")
        print(f"First 5: {demo_keys[:5]}")
        print(f"Last 5:  {demo_keys[-5:]}")

        first_demo_key = demo_keys[0]
        demo = f[f"data/{first_demo_key}"]
        print(f"\n--- SINGLE DEMO: data/{first_demo_key} ---")
        print(f"attrs: {dict(demo.attrs)}")
        for sub in demo.keys():
            sub_item = demo[sub]
            if isinstance(sub_item, h5py.Dataset):
                print(f"  - {sub}: shape={sub_item.shape} dtype={sub_item.dtype}")
            else:
                print(f"  - {sub}/ (group)")

        for sub_name in ["actions", "rewards", "dones", "obs", "states"]:
            if sub_name in demo:
                obj = demo[sub_name]
                if isinstance(obj, h5py.Dataset):
                    arr = obj[...]
                    print(f"\n  {sub_name} sample (first 3 rows):")
                    print(arr[:3])
                else:
                    print(f"\n  {sub_name} sub-keys:")
                    for k in obj.keys():
                        ds = obj[k]
                        if isinstance(ds, h5py.Dataset):
                            arr = ds[...]
                            print(f"    {k}: shape={ds.shape} dtype={ds.dtype} min={arr.min()} max={arr.max()} mean={float(arr.mean()):.4f}")

        first = demo_keys[0]
        obs = f[f"data/{first}/obs"]
        print(f"\n--- OBS sub-dataset attrs ---")
        for k in obs.keys():
            ds = obs[k]
            if isinstance(ds, h5py.Dataset):
                print(f"  {k}: shape={ds.shape} dtype={ds.dtype} attrs={dict(ds.attrs)}")

    # Check for global metadata datasets often present in robomimic
    print("\n--- GLOBAL METADATA ---")
    for k in f.keys():
        item = f[k]
        if isinstance(item, h5py.Dataset) and not isinstance(item, h5py.Group):
            arr = item[...]
            print(f"  /{k}: shape={arr.shape} dtype={arr.dtype}")
            if arr.size < 30:
                print(f"    value: {arr}")
    # Check attrs on data group
    if "data" in f and "attrs" in f["data"]:
        print(f"  /data/attrs sample: {list(f['data/attrs'][:5])}")
    if "data" in f and "mask" in f["data"]:
        print(f"  /data/mask shape: {f['data/mask'].shape} dtype: {f['data/mask'].dtype}")
    if "data" in f and "total" in f["data"]:
        print(f"  /data/total: {f['data/total'][...]}")
