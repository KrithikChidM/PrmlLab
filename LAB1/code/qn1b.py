import numpy as np

a_list = []
b_list = []
n = int(input())
print(f"=============================
A: {1}*{n}")
for _ in range(1):
  t = []
  for _ in range(n):
    t.append(int(input()))
  a_list.append(t)
a = np.array(a_list)

print(f"=============================
B: {n}*{1}")
for _ in range(n):
  t = []
  for _ in range(1):
    t.append(int(input()))
  b_list.append(t)
b = np.array(b_list)

print(f"=============================
A:
{a}
B:
{b}")
result_np = a @ b # NumPy dot product, result is a 1x1 array
result = result_np.item()
print("====================================")
print(result)
print("====================================")