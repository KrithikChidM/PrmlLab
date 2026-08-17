import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd

image_path = "../img/cat_07_sq.png"
img = Image.open(image_path)
gray = img.convert("L")

A = np.array(gray, dtype=float)
print("Matrix shape:", A.shape)
U, singular_values, Vt = np.linalg.svd(
    A,
    full_matrices=False
)

singular_values_squared = singular_values ** 2
total_energy = np.sum(singular_values_squared)
cumulative_explained_variance = (
    np.cumsum(singular_values_squared)
    / total_energy
) * 100

k_values = [180,220,240]
results = []

for k in k_values:
    U_k = U[:, :k]
    Sigma_k = np.diag(singular_values[:k])
    Vt_k = Vt[:k, :]
    A_k = U_k @ Sigma_k @ Vt_k
    A_k = np.clip(A_k, 0, 255)
    frobenius_error = np.linalg.norm(A - A_k, ord="fro")
    cev = cumulative_explained_variance[k - 1]
    results.append({
        "k": k,
        "Frobenius Error": frobenius_error,
        "Cumulative Explained Variance (%)": cev
    })
    error_image = np.abs(A - A_k)
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(A, cmap="gray")
    plt.title("Original Image")
    plt.axis("off")
    plt.subplot(1, 3, 2)
    plt.imshow(A_k, cmap="gray")
    plt.title(f"SVD Reconstruction (k={k})")
    plt.axis("off")
    plt.subplot(1, 3, 3)
    plt.imshow(error_image, cmap="gray")
    plt.title(f"SVD Error Image (k={k})")
    plt.axis("off")
    plt.tight_layout()
    plt.show()
n = A.shape[0]
results_df_svd = pd.DataFrame(results)
results_df_svd["Compressed Storage"] = (
    2 * n * results_df_svd["k"] + results_df_svd["k"]
)
original_storage = n * n
results_df_svd["Compression Factor"] = (
    original_storage /
    results_df_svd["Compressed Storage"]
)
results_df_svd["Frobenius Error"] = \
    results_df_svd["Frobenius Error"].round(4)
results_df_svd["Cumulative Explained Variance (%)"] = \
    results_df_svd["Cumulative Explained Variance (%)"].round(4)
results_df_svd["Compression Factor"] = \
    (1/results_df_svd["Compression Factor"]).round(4)
print("\nSVD Reconstruction Results:")
print(
    results_df_svd[
        [
            "k",
            "Frobenius Error",
            "Compression Factor"
        ]
    ].to_string(index=False)
)