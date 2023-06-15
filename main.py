import numpy as np
import matplotlib.pyplot as plt
import os


def threshold_selection(coefficients, ratio):
    max_coef = np.max(np.abs(coefficients))
    threshold = ratio * max_coef
    return threshold


def EZW_encode(coefficients, threshold):
    significant_coeffs = []
    stack = [(0, 0, coefficients.shape[0])]  # Stack to track submatrices

    while stack:
        row, col, size = stack.pop()
        submatrix = coefficients[row : row + size, col : col + size]

        if np.all(np.abs(submatrix) <= threshold):
            continue

        max_coef_idx = np.unravel_index(np.argmax(np.abs(submatrix)), submatrix.shape)
        max_coef = submatrix[max_coef_idx]

        significant_coeffs.append(
            (row + max_coef_idx[0], col + max_coef_idx[1], max_coef)
        )

        if size > 1:
            half_size = size // 2
            stack.append((row, col, half_size))
            stack.append((row, col + half_size, half_size))
            stack.append((row + half_size, col, half_size))
            stack.append((row + half_size, col + half_size, half_size))

    return significant_coeffs


def EZW_decode(significant_coeffs, size):
    coefficients = np.zeros((size, size))

    for coeff in significant_coeffs:
        row, col, value = coeff
        coefficients[row, col] = value

    return coefficients


# Generate a random 4x4 image
random_image = np.random.randint(0, 256, size=(8, 8), dtype=np.uint8)

# Calculate threshold based on the ratio
threshold_ratio = 0
threshold = threshold_selection(random_image, threshold_ratio)

# Plot original image
plt.subplot(131)
plt.imshow(random_image, cmap="gray")
plt.title("Original Image")

# Encode using EZW
significant_coeffs = EZW_encode(random_image, threshold)

# Reconstruct the image
reconstructed_image = EZW_decode(significant_coeffs, random_image.shape[0])

# Plot reconstructed image
plt.subplot(132)
plt.imshow(reconstructed_image, cmap="gray")
plt.title("Reconstructed Image")

# Calculate the difference between original and reconstructed coefficients
difference = random_image - reconstructed_image

# Plot the difference
plt.subplot(133)
plt.imshow(difference, cmap="gray")
plt.title("Difference (Original - Reconstructed)")

# Save the original and reconstructed images to files
plt.imsave("original_image.png", random_image, cmap="gray")
plt.imsave("reconstructed_image.png", reconstructed_image, cmap="gray")

# Compare image sizes
original_size = os.path.getsize("original_image.png")
reconstructed_size = os.path.getsize("reconstructed_image.png")

print("Original Image Size:", original_size, "bytes")
print("Reconstructed Image Size:", reconstructed_size, "bytes")

plt.tight_layout()
plt.show()
