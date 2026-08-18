import numpy as np
upper = 0
lower = 0
a_list = []
m = int(input())
print(f"=============================
A: {m}*{m}")
for _ in range(m):
  row = []
  for _ in range(m):
    row.append(int(input()))
  a_list.append(row)
a = np.array(a_list)
for i in range(len(a)):
  for j in range(len(a[i])):
    if j > i:
      if a[i][j] != 0 or upper != 0:
        upper = 1
    if j < i:
      if a[i][j] != 0 or lower != 0:
        lower = 1
print(f"=============================
{a}")
print("=============================
1. ",end="")
print("Upper triangular
2. " if upper == 0 else "Not upper triangular
2. ",end = "")
print("Lower triangular" if lower == 0 else "Not lower triangular")
print("=============================")