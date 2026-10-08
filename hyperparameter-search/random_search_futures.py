import time
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def sample_C(rng: np.random.Generator) -> float:
    return 10.0 ** rng.uniform(-5.0, -1.0)

def run_one_trial(args):
    """Run one (C -> train -> acc) trial. Executed in a worker process."""
    (trial_id, C, X_train, y_train, X_val, y_val) = args

    clf = LogisticRegression(
        C=C,
        penalty="l1",
        solver="saga",
        max_iter=3000,
    )
    clf.fit(X_train, y_train)
    acc = accuracy_score(y_val, clf.predict(X_val))
    return trial_id, C, acc

def main(n_trials=32, seed_data=42, seed_rng=0, n_workers=4, verbose=False):
    # 1) Data
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

    # 2) Sample Cs (deterministic)
    rng = np.random.default_rng(seed_rng)
    Cs = [sample_C(rng) for _ in range(n_trials)]

    t0 = time.perf_counter()

    best_c = None
    best_acc = -1.0
    # 3) Parallel trials
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futures = []
        for i, C in enumerate(Cs):
            futures.append(ex.submit(run_one_trial, (i, C, X_train, y_train, X_val, y_val)))

        for fut in as_completed(futures):
            trial_id, C, acc = fut.result()
            if verbose:
                print(f"{trial_id:02d}: C={C:.3e}, acc={acc:.4f}")
            if acc > best_acc:
                best_acc = acc
                best_c = C

    t1 = time.perf_counter()

    print("\nFINAL RESULTS (FUTURES)")
    print(f"Workers: {n_workers}")
    print(f"Trials: {n_trials}")
    print(f"Elapsed time: {t1 - t0:.3f} s")
    print(f"Best: C={best_c:.3e}, acc={best_acc:.4f}")

if __name__ == "__main__":
    # αλλαγή n_workers για πειράματα
    main(n_trials=32, n_workers=3, verbose=False)

