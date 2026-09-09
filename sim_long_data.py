import numpy as np
import pandas as pd

np.random.seed(123)

n_subjects = 100
min_times, max_times = 3, 20

beta0, beta1, beta2 = 1.0, 0.5, 1.5
sigma = 0.5

alpha0, alpha1 = -0.5, 1.0

z = np.random.normal(0, 1, n_subjects)

rows = []
for i in range(n_subjects):
    n_times = np.random.randint(min_times, max_times + 1)
    times = np.sort(np.random.uniform(0, 10, n_times))

    for t in times:
        x = beta0 + beta1 * t + beta2 * z[i] + np.random.normal(0, sigma)
        rows.append([i, t, x, z[i]])

long_data = pd.DataFrame(rows, columns=["subject", "time", "x", "z"])

p = 1 / (1 + np.exp(-(alpha0 + alpha1 * z)))
event = np.random.binomial(1, p)

subject_data = pd.DataFrame({
    "subject": np.arange(n_subjects),
    "z": z,
    "event": event,
    "event_probability": p
})

long_data.to_csv("longitudinal_data.csv", index=False)
subject_data.to_csv("subject_data.csv", index=False)