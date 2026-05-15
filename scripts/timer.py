"""
Script to measure the execution time of a function.
"""

import statistics
import time


def timer(func, repeats=5, warmup=1):
    """
    Decorator to measure the execution time of a function.

    Runs ``warmup`` untimed iterations first (to absorb lazy allocator /
    kernel-compile / cache effects), then ``repeats`` timed iterations,
    and prints mean / median / min over the timed runs. Returns the last
    result so the caller can keep using it.
    """

    def wrapper(*args, **kwargs):
        for _ in range(warmup):
            result = func(*args, **kwargs)
        times = []
        for _ in range(repeats):
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            end_time = time.perf_counter()
            times.append(end_time - start_time)
        mean = statistics.fmean(times)
        median = statistics.median(times)
        best = min(times)
        print(
            f"Execution time over {repeats} runs: "
            f"mean={mean:.4g}s, median={median:.4g}s, min={best:.4g}s"
        )
        return result

    return wrapper


if __name__ == "__main__":
    from lgt.lattice import SU, build_transport_sums, random_links

    timed_bts = timer(build_transport_sums, repeats=10)
    SU3 = SU(3)
    U = random_links(group=SU3, L=16, D=4)

    T = timed_bts(U, R=4, group=SU3)
