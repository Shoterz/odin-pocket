"""Linux exec trampoline: a training child must die if its supervising process dies."""
import ctypes
import os
import signal
import sys


def main():
    parent = int(sys.argv[1])
    command = sys.argv[2:]
    if not command:
        raise ValueError('Missing child command')
    libc = ctypes.CDLL(None, use_errno=True)
    # PR_SET_PDEATHSIG persists across our non-privileged exec. Use a fresh Python
    # process rather than preexec_fn after importing multithreaded PyTorch.
    if libc.prctl(1, signal.SIGKILL, 0, 0, 0) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    # Cover the race where the owner died before prctl armed the signal.
    if os.getppid() != parent:
        raise RuntimeError('Supervisor exited before child startup')
    os.execv(command[0], command)


if __name__ == '__main__':
    main()
