"""
Script to measure the execution time of a function.
"""

import statistics
import time

if __name__ == "__main__":
    from gelt.lattice import SU, build_transport_sums, random_links

    SU3 = SU(3)
    warmup = 2
    repeats = 10

    def func():
        U = random_links(gaugegroup=SU3, L=16, D=4).to("mps")
        T = build_transport_sums(U, R=4, gaugegroup=SU3)

    for _ in range(warmup):
        func()
    times = []
    for _ in range(repeats):
        start_time = time.perf_counter()
        func()
        end_time = time.perf_counter()
        times.append(end_time - start_time)
    mean = statistics.fmean(times)
    median = statistics.median(times)
    best = min(times)
    print(
        f"Execution time over {repeats} runs: "
        f"mean={mean:.4g}s, median={median:.4g}s, min={best:.4g}s"
    )
