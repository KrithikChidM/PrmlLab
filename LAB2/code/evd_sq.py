import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

image_path = "../img/cat_07_sq.png"

img = Image.open(image_path).convert("L")
A = np.array(img, dtype=float)

n = A.shape[0]
print("Matrix shape:", A.shape)
eigenvalues, Q = np.linalg.eig(A)
idx = np.argsort(np.abs(eigenvalues))[::-1]
eigenvalues = eigenvalues[idx]
Q = Q[:, idx]

used = np.zeros(len(eigenvalues), dtype=bool)
ordered_indices = []
conjugate_pair_products = []

for i in range(len(eigenvalues)):
    if used[i]:
        continue
    lam = eigenvalues[i]
    if np.abs(lam.imag) < 1e-10:
        ordered_indices.append(i)
        used[i] = True
    else:
        conjugate_target = np.conj(lam)
        remaining = np.where(~used)[0]
        differences = np.abs(
            eigenvalues[remaining] - conjugate_target
        )
        j = remaining[np.argmin(differences)]
        lam_conjugate = eigenvalues[j]
        pair_product = lam * lam_conjugate
        conjugate_pair_products.append(
            np.real(pair_product)
        )
        ordered_indices.append(i)
        ordered_indices.append(j)
        used[i] = True
        used[j] = True
eigenvalues = eigenvalues[ordered_indices]
Q = Q[:, ordered_indices]
eigenvalue_magnitude = np.abs(eigenvalues)
total_energy = np.sum(eigenvalue_magnitude)
cumulative_explained_variance = (
    np.cumsum(eigenvalue_magnitude) /
    total_energy
) * 100
k_values = [180, 190, 200, 210, 220, 230, 240, 250, 260]
results = []
for requested_k in k_values:
    k = requested_k
    if k < len(eigenvalues):
        current = eigenvalues[k - 1]
        if np.abs(current.imag) > 1e-10:
            if np.abs(
                eigenvalues[k] -
                np.conj(current)
            ) < 1e-8:
                k += 1
    Lambda_k = np.zeros(
        (n, n),
        dtype=complex
    )
    Lambda_k[
        np.arange(k),
        np.arange(k)
    ] = eigenvalues[:k]
    A_k = (
        Q @ Lambda_k @ np.linalg.inv(Q)
    )
    A_k = np.real(A_k)
    A_k = np.clip(A_k, 0, 255)
    error_image = np.abs(A - A_k)
    frobenius_error = np.linalg.norm(
        A - A_k,
        ord="fro"
    )
    cev = cumulative_explained_variance[k - 1]
    original_storage = n * n
    compressed_storage = (
        2 * n * k + k
    )
    compression_factor = (
        original_storage /
        compressed_storage
    )
    results.append({
        "k": k,
        "Frobenius Error": frobenius_error,
        "Compression Factor": compression_factor
    })
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(A, cmap="gray")
    plt.title("Original Image")
    plt.axis("off")
    plt.subplot(1, 3, 2)
    plt.imshow(A_k, cmap="gray")
    plt.title(f"EVD Reconstruction (k={k})")
    plt.axis("off")
    plt.subplot(1, 3, 3)
    plt.imshow(error_image, cmap="gray")
    plt.title(f"Error Image (k={k})")
    plt.axis("off")
    plt.tight_layout()
    plt.show()
results_df_evd = pd.DataFrame(results)
results_df_evd["Frobenius Error"] = (
    results_df_evd["Frobenius Error"]
    .round(4)
)
results_df_evd["Compression Factor"] = (
    (1/results_df_evd["Compression Factor"])
    .round(4)
)
print("\nEVD Reconstruction Results:")
print(
    results_df_evd.to_string(index=False)
)
