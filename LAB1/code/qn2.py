import numpy as np
import matplotlib.pyplot as plt

a = int(input("A: "))
b = int(input("B: "))
n = int(input("N: "))

uniform = np.random.uniform(a, b, n)

plt.figure(figsize=(6, 4))
plt.plot(uniform)
plt.title("Uniform Distribution")
plt.xlabel("Value")
plt.ylabel("Density")
plt.show()

mu = np.mean(uniform)
sigma = np.std(uniform)

gaussian = []

while len(gaussian) < n:
    u1 = np.random.uniform(-1, 1)
    u2 = np.random.uniform(-1, 1)
    s = u1**2 + u2**2
    if s > 0 and s < 1:
        k = np.sqrt((-2 * np.log(s)) / s)
        x = u1 * k
        y = u2 * k
        gaussian.append(mu + sigma * x)
        if len(gaussian) < n:
            gaussian.append(mu + sigma * y)

gaus_dist = np.array(gaussian)

plt.figure(figsize=(6, 4))
plt.hist(gaus_dist, bins=30, density=True, edgecolor='black')
plt.title("Gaussian Distribution")
plt.xlabel("Value")
plt.ylabel("Density")
plt.show()