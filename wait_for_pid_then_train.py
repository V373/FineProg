#!/usr/bin/env python3
"""Wait for a target PID to exit, then run train_encoder.py once in conda env.
"""

import errno
import os
import subprocess
import time

TARGET_PID = 1685702
POLL_INTERVAL_SECONDS = 5


def is_process_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError as exc:
        if exc.errno == errno.ESRCH:
            return False
        if exc.errno == errno.EPERM:
            return True
        raise
    return True


def main() -> None:
    while is_process_running(TARGET_PID):
        print("waiting", flush=True)
        time.sleep(POLL_INTERVAL_SECONDS)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    print("start training", flush=True)
    subprocess.run(
        ["conda", "run", "-n", "fineprog", "python", "train_encoder.py"],
        cwd=script_dir,
        check=True,
    )


if __name__ == "__main__":
    main()
