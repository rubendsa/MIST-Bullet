import pstats


stats = pstats.Stats("profilingResults")
stats.sort_stats("tottime")
stats.print_stats(10)