##


def _unfold_parallel(lists, n_jobs):
    full_list = []
    for i in range(n_jobs):
        full_list.extend(lists[i])
    return full_list
