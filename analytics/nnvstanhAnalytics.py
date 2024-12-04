import matplotlib.pyplot as plt

# Text dosyalarından verileri okuma
def read_accuracies(file_path):
    with open(file_path, "r") as file:
        return [float(line.strip()) for line in file.readlines()]

# Dosya yollarını belirleyin
tanh_file = "10tanhAccuracy.txt"
nn_file = "nnAccuracy.txt"

# Verileri oku
tanh_accuracies = read_accuracies(tanh_file)
nn_accuracies = read_accuracies(nn_file)

# Epoch numaraları
epochs = list(range(1, len(tanh_accuracies) + 1))

# 10tanh accuracy verisini 0-100 aralığına ölçekle
tanh_accuracies_scaled = [x * 100 for x in tanh_accuracies]

# Grafik oluşturma
plt.figure(figsize=(10, 6))
plt.plot(epochs, tanh_accuracies_scaled, label='10tanh Accuracy (Scaled)', marker='o', linestyle='-')
plt.plot(epochs, nn_accuracies, label='nn Accuracy', marker='x', linestyle='--')

# Y ekseni aralıklarını özelleştirme
plt.yticks(range(0, 101, 5))

plt.title("Model Accuracies Over Epochs (0-100 Scale)")
plt.xlabel("Epochs")
plt.ylabel("Accuracy (0-100 Scale)")
plt.legend()
plt.grid(True)

# PDF olarak kaydet
plt.savefig("accuracy_comparison_scaled.pdf", format="pdf")
plt.close()
