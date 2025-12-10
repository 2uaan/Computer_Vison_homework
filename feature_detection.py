import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist

def load_data_standard():
    print("📥 Đang tải dữ liệu MNIST từ thư viện...")
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    print(f"✅ Tải xong! Kích thước Train: {x_train.shape}, Test: {x_test.shape}")
    return (x_train, y_train), (x_test, y_test)

class HoG_FromScratch:
    def __init__(self, cell_size=4, bin_count=9, block_size=2):
        self.cell_size = cell_size
        self.bin_count = bin_count
        self.block_size = block_size

    def convolve2d(self, image, kernel):
        """Hàm tích chập thủ công"""
        k_h, k_w = kernel.shape
        h, w = image.shape
        out_h, out_w = h - k_h + 1, w - k_w + 1
        output = np.zeros((out_h, out_w))

        # Tối ưu hóa một chút bằng cách tính toán vector thay vì loop từng điểm
        for i in range(out_h):
            for j in range(out_w):
                region = image[i:i+k_h, j:j+k_w]
                output[i, j] = np.sum(region * kernel)
        return output

    def compute_gradients(self, image):
        """Tính Magnitude và Orientation (Góc) dùng Sobel"""
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

        # Tích chập để tìm biên
        g_x = self.convolve2d(image, sobel_x)
        g_y = self.convolve2d(image, sobel_y)

        # Tính độ lớn và góc
        magnitude = np.sqrt(g_x**2 + g_y**2)
        orientation = np.rad2deg(np.arctan2(g_y, g_x)) % 180

        return magnitude, orientation

    def compute_cell_histograms(self, magnitude, orientation):
        """Chia ảnh thành các Cell và tính Histogram"""
        h_grad, w_grad = magnitude.shape
        n_cells_y = h_grad // self.cell_size
        n_cells_x = w_grad // self.cell_size

        histograms = np.zeros((n_cells_y, n_cells_x, self.bin_count))
        bin_width = 180 / self.bin_count

        for i in range(n_cells_y):
            for j in range(n_cells_x):
                # Vùng dữ liệu của cell
                r, c = i * self.cell_size, j * self.cell_size
                cell_mag = magnitude[r:r+self.cell_size, c:c+self.cell_size]
                cell_ang = orientation[r:r+self.cell_size, c:c+self.cell_size]

                # Bỏ phiếu vào các bin
                bin_idx = (cell_ang / bin_width).astype(int)
                bin_idx[bin_idx >= self.bin_count] = 0

                # Cộng dồn magnitude vào bin tương ứng
                flat_bin = bin_idx.ravel()
                flat_mag = cell_mag.ravel()
                for k in range(len(flat_bin)):
                    histograms[i, j, flat_bin[k]] += flat_mag[k]

        return histograms

    def block_normalization(self, histograms):
        """Chuẩn hóa khối (L2 Norm) để tạo Feature Vector cuối cùng"""
        n_cells_y, n_cells_x, _ = histograms.shape
        n_blocks_y = n_cells_y - self.block_size + 1
        n_blocks_x = n_cells_x - self.block_size + 1

        normalized_blocks = []
        epsilon = 1e-5

        for i in range(n_blocks_y):
            for j in range(n_blocks_x):
                # Lấy block (gồm nhiều cell)
                block = histograms[i:i+self.block_size, j:j+self.block_size, :]

                # Trải phẳng và chuẩn hóa
                v = block.ravel()
                v = v / np.sqrt(np.sum(v**2) + epsilon)
                normalized_blocks.append(v)

        # Nối tất cả lại thành 1 vector duy nhất
        if len(normalized_blocks) > 0:
            return np.concatenate(normalized_blocks)
        return np.array([])

    def extract(self, image):
        """Hàm API chính để gọi từ bên ngoài"""
        mag, ang = self.compute_gradients(image)
        hist = self.compute_cell_histograms(mag, ang)
        vector = self.block_normalization(hist)
        return vector

# 1. Tải dữ liệu bằng thư viện
(x_train, y_train), (x_test, y_test) = load_data_standard()

# 2. Lấy thử 1 ảnh ngẫu nhiên để test
idx = 100
sample_img = x_train[idx]
label = y_train[idx]

print(f"\n🧪 Đang trích xuất đặc trưng cho ảnh số: {label} (Index: {idx})")

# 3. Khởi tạo và chạy HoG tự viết
    # MNIST 28x28 -> HoG parameters này sẽ cho ra vector 900 chiều
hog = HoG_FromScratch(cell_size=4, bin_count=9, block_size=2)
feature_vector = hog.extract(sample_img)

print("="*40)
print(f"KẾT QUẢ TRÍCH XUẤT:")
print(f"- Input shape: {sample_img.shape}")
print(f"- Output Feature Vector shape: {feature_vector.shape}")
print("="*40)

# 4. Hiển thị trực quan
plt.figure(figsize=(12, 4))

# Ảnh gốc
plt.subplot(1, 2, 1)
plt.imshow(sample_img, cmap='gray')
plt.title(f"Original Image: Label {label}")
plt.axis('off')

# Vector đặc trưng
plt.subplot(1, 2, 2)
plt.plot(feature_vector, color='blue', linewidth=0.8)
plt.title("HoG Feature Descriptor")
plt.xlabel("Feature Index")
plt.ylabel("Normalized Strength")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()