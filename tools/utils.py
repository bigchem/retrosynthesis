
def get_batch_idx(N, batch_size):
    num_batches = (N + batch_size - 1) / batch_size

    for i in range(int(num_batches)):
        start, end = i * batch_size, (i + 1) * batch_size
        idx = slice(start, end)

        yield idx


