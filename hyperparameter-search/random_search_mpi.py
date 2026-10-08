import time
import numpy as np
from mpi4py import MPI

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def sample_C(rng: np.random.Generator) -> float:
    # C ~ 10^{U[-5, -1]}
    return 10.0 ** rng.uniform(-5.0, -1.0)

def main(n_trials=32, seed_data=42, seed_rng=0):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # 1) Όλοι φτιάχνουν το ίδιο dataset (ίδια random_state => ίδια δεδομένα)
    X, y = make_classification(
        n_samples=100000,
        n_features=200,
        n_informative=10,
        n_redundant=10,
        class_sep=1.2,
        flip_y=0.08,
        random_state=seed_data,
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=seed_data
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    # 2) Δημιουργία C τιμών deterministically (ίδιες για όλους)
    rng = np.random.default_rng(seed_rng)
    Cs = [sample_C(rng) for _ in range(n_trials)]

    # 3) Timing (ταυτόχρονη έναρξη)
    comm.Barrier()
    t0 = time.perf_counter()

    # 4) Κάθε rank εκτελεί subset: trial = rank, rank+size, ...
    local_best_c = None
    local_best_acc = -1.0

    for i in range(rank, n_trials, size):
        C = Cs[i]

        clf = LogisticRegression(
            C=C,
            penalty="l1",
            solver="saga",
            max_iter=3000,
        )
        clf.fit(X_train, y_train)
        acc = accuracy_score(y_val, clf.predict(X_val))

        if acc > local_best_acc:
            local_best_acc = acc
            local_best_c = C

    # 5) Reduce για global best (max accuracy)
    # Στέλνουμε (acc, rank) για να ξέρουμε ποιος έχει το καλύτερο
    local_pair = (local_best_acc, rank)
    global_pair = comm.allreduce(local_pair, op=MPI.MAXLOC)
    best_acc, best_rank = global_pair

    # 6) Ο νικητής στέλνει το best C στον rank 0
    best_c = None
    if rank == best_rank:
        best_c = local_best_c

    best_c = comm.bcast(best_c, root=best_rank)

    comm.Barrier()
    t1 = time.perf_counter()

    if rank == 0:
        print("\nFINAL RESULTS (MPI)")
        print(f"Ranks: {size}")
        print(f"Trials: {n_trials}")
        print(f"Elapsed time: {t1 - t0:.3f} s")
        print(f"Best: C={best_c:.3e}, acc={best_acc:.4f}")

if __name__ == "__main__":
    main(n_trials=32)

