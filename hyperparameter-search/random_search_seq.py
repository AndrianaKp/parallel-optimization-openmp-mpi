import time
import warnings
import numpy as np

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Κρύβουμε τα warnings της LogisticRegression 
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.linear_model")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.linear_model")

def sample_C(rng: np.random.Generator) -> float:
    # C ~ 10^{U[-5, -1]}
    return 10.0 ** rng.uniform(-5.0, -1.0)

def main(
    n_trials: int = 32,
    seed_data: int = 42,
    seed_rng: int = 0,
    verbose: bool = False,   # για σωστά timings: False
) -> None:
    # 1) Dataset 
    X, y = make_classification(
        n_samples=100000,
        n_features=200,
        n_informative=10,
        n_redundant=10,
        class_sep=1.2,
        flip_y=0.08,
        random_state=seed_data,
    )

    # 2) Split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=seed_data
    )

    # 3) Scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    rng = np.random.default_rng(seed_rng)

    best_c = None
    best_acc = -1.0

    t0 = time.perf_counter()

    for i in range(n_trials):
        C = sample_C(rng)
        # Πιο συμβατό setup για L1 + saga
        clf = LogisticRegression(
            C=C,
            penalty="l1",
            solver="saga",
            max_iter=3000,
        )

        clf.fit(X_train, y_train)
        acc = accuracy_score(y_val, clf.predict(X_val))

        if verbose:
            print(f"{i:02d}: C={C:.3e}, acc={acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            best_c = C

    t1 = time.perf_counter()

    print("\nFINAL RESULTS (SEQUENTIAL)")
    print(f"Trials: {n_trials}")
    print(f"Elapsed time: {t1 - t0:.3f} s")
    print(f"Best: C={best_c:.3e}, acc={best_acc:.4f}")

if __name__ == "__main__":
    main(n_trials=32, verbose=False)

