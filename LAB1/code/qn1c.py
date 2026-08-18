import numpy as np

a_list = []
m, n = [int(x) for x in input().split()]
print(f"=============================
A: {m}*{n}")
for _ in range(m):
  row = []
  for _ in range(n):
    row.append(int(input()))
  a_list.append(row)
a = np.array(a_list)

print(f"=============================
A:
{a}")
result = a.T
print("====================================")
print(result)
print("====================================")