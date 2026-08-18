import numpy as np

a_list = []
b_list = []
m, n, p = [int(x) for x in input().split()]
print(f"=============================
A: {m}*{n}")
for _ in range(m):
  t = []
  for _ in range(n):
    t.append(int(input()))
  a_list.append(t)
a = np.array(a_list)

print(f"=============================
B: {n}*{p}")
for _ in range(n):
  t = []
  for _ in range(p):
    t.append(int(input()))
  b_list.append(t)
b = np.array(b_list)

print(f"=============================
A:
{a}
B:
{b}")
result = a @ b # NumPy matrix multiplication
print("====================================")
print(result)
print("====================================")