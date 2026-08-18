import numpy as np

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

print(f"=============================
A:
{a}
")

result = a.T

print("====================================")
print(result)
print("====================================")

if np.array_equal(a, result):
  print("Symmetric")
else:
  print("Non Symmetric")
print("====================================")