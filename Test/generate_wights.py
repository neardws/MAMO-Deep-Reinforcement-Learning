import numpy as np

def generate_weights(count=1, n=3, m=1):
    all_weights = []

    target = np.random.dirichlet(np.ones(n), 1)[0]
    prev_t = target
    for _ in range(count // m):
        target = np.random.dirichlet(np.ones(n), 1)[0]
        if m == 1:
            all_weights.append(target)
        else:
            for i in range(m):
                i_w = target * (i + 1) / float(m) + prev_t * \
                    (m - i - 1) / float(m)
                all_weights.append(i_w)
        prev_t = target + 0.

    return all_weights

if __name__ == '__main__':
    weights = generate_weights(count=1, n=3, m=1)
    print(weights)
    print(sum(sum(weights)))