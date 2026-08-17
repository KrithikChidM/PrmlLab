import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

image_path = "../img/cat_07_rect.png"

img = Image.open(image_path).convert("L")
A = np.array(img, dtype=float)

m, n = A.shape
print("Image shape:", A.shape)
U, singular_values, Vt = np.linalg.svd(
    A,
    full_matrices=False
)
max_k = min(m, n)

k_values = np.arange(180, 251, 10)
k_values = k_values[k_values <= max_k]
results = []
for k in k_values:

    U_k = U[:, :k]
    Sigma_k = np.diag(
        singular_values[:k]
    )
    Vt_k = Vt[:k, :]
    A_k = U_k @ Sigma_k @ Vt_k
    A_k = np.real(A_k)
    A_k = np.clip(A_k, 0, 255)
    error_image = np.abs(A - A_k)
    frobenius_error = np.linalg.norm(
        A - A_k,
        ord="fro"
    )

    original_storage = m * n
    compressed_storage = (
        m * k + k + k * n
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
    plt.title(f"SVD Reconstruction (k={k})")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(error_image, cmap="gray")
    plt.title(f"Error Image (k={k})")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

results_df_svd_rect = pd.DataFrame(results)

results_df_svd_rect["Frobenius Error"] = (
    results_df_svd_rect["Frobenius Error"].round(4)
)

results_df_svd_rect["Compression Factor"] = (
    (1/results_df_svd_rect["Compression Factor"]).round(4)
)

print("\nSVD Reconstruction Results - Rectangular Image:")

print(
    results_df_svd_rect.to_string(index=False)
)

results2_df_svd_rect = results_df_svd_rect.copy()

min_fro_error = results2_df_svd_rect['Frobenius Error'].min()
max_fro_error = results2_df_svd_rect['Frobenius Error'].max()
results2_df_svd_rect['Frobenius Error'] = (
    results2_df_svd_rect['Frobenius Error'] - min_fro_error
) / (max_fro_error - min_fro_error)

min_comp_factor = results2_df_svd_rect['Compression Factor'].min()
max_comp_factor = results2_df_svd_rect['Compression Factor'].max()
results2_df_svd_rect['Compression Factor'] = (
    results2_df_svd_rect['Compression Factor'] - min_comp_factor
) / (max_comp_factor - min_comp_factor)

print("\nMin-Max Scaled SVD Results:")
results2_df_svd_rect = results2_df_svd_rect.sort_values(by="Frobenius Error", ascending=True)
print(results2_df_svd_rect[[ "k", "Frobenius Error", "Compression Factor"]].to_string(index=False))