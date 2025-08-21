# Layer normalization in CUDA

This is a guide to building a layer normalization kernel in CUDA.

# 1. Welford Algorithm

Let's start by implementing Welford's algorithm for computing variance. This method is quite important, as it is a numerically stable method of computing variance which can be computed online (ie we don't need all of the data points in advance to compute variance).
