import matplotlib.pyplot as plt

# Veriyi okuyacak fonksiyon
def read_data(file_path):
    seconds, epochs, train_loss, test_loss = [], [], [], []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.split()
            if len(parts) == 4:
                seconds.append(float(parts[0]))
                epochs.append(float(parts[1]))
                train_loss.append(float(parts[2]))
                test_loss.append(float(parts[3]))
    return seconds, epochs, train_loss, test_loss

# Karşılaştırmalı grafik çizme fonksiyonu
def plot_comparison(sgd_data, gd_data, adam_data):
    # SGD, GD ve Adam verilerini ayır
    sgd_seconds, sgd_epochs, sgd_train_loss, sgd_test_loss = sgd_data
    gd_seconds, gd_epochs, gd_train_loss, gd_test_loss = gd_data
    adam_seconds, adam_epochs, adam_train_loss, adam_test_loss = adam_data

    # Epochs karşılaştırması
    plt.figure(figsize=(10, 5))
    plt.plot(sgd_epochs, sgd_train_loss, label='SGD Train Loss', marker='o', linestyle='-')
    plt.plot(sgd_epochs, sgd_test_loss, label='SGD Test Loss', marker='o', linestyle='-')
    plt.plot(gd_epochs, gd_train_loss, label='GD Train Loss', marker='s', linestyle='-')
    plt.plot(gd_epochs, gd_test_loss, label='GD Test Loss', marker='s', linestyle='-')
    plt.plot(adam_epochs, adam_train_loss, label='Adam Train Loss', marker='^', linestyle='-')
    plt.plot(adam_epochs, adam_test_loss, label='Adam Test Loss', marker='^', linestyle='-')
    plt.title("Loss vs Epochs: SGD vs GD vs Adam")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig('comparison_epoch.pdf', format='pdf', bbox_inches='tight', dpi=300)
    # Or for vector graphics
    plt.savefig('comparison_epoch.eps', format='eps', bbox_inches='tight')


    plt.show()


    # Time (Seconds) karşılaştırması (ilk 30 saniye)
    plt.figure(figsize=(10, 5))
    plt.plot(sgd_seconds, sgd_train_loss, label='SGD Train Loss', marker='o', linestyle='-')
    plt.plot(sgd_seconds, sgd_test_loss, label='SGD Test Loss', marker='o', linestyle='-')
    plt.plot(gd_seconds, gd_train_loss, label='GD Train Loss', marker='s', linestyle='-')
    plt.plot(gd_seconds, gd_test_loss, label='GD Test Loss', marker='s', linestyle='-')
    plt.plot(adam_seconds, adam_train_loss, label='Adam Train Loss', marker='^', linestyle='-')
    plt.plot(adam_seconds, adam_test_loss, label='Adam Test Loss', marker='^', linestyle='-')
    plt.title("Loss vs Time (Seconds): SGD vs GD vs Adam (First 30 Seconds)")
    plt.xlabel("Time (Seconds)")
    plt.ylabel("Loss")
    plt.xlim(0, 30)  # X eksenini 0 ile 30 arasında sınırla
    plt.legend()
    plt.grid(True)

    plt.savefig('comparison_time.pdf', format='pdf', bbox_inches='tight', dpi=300)
    # Or for vector graphics
    plt.savefig('comparison_time.eps', format='eps', bbox_inches='tight')

    plt.show()

    



# Dosya yollarını belirt
sgd_file_path = 'SGD_T_E_TL_TEL.txt'  # SGD sonuçlarını içeren dosya
gd_file_path = 'GD_T_E_TL_TEL.txt'    # GD sonuçlarını içeren dosya
adam_file_path = 'ADAM_T_E_TL_TEL.txt' # Adam sonuçlarını içeren dosya

# Verileri oku
sgd_data = read_data(sgd_file_path)
gd_data = read_data(gd_file_path)
adam_data = read_data(adam_file_path)

# Karşılaştırmalı grafikleri çiz
plot_comparison(sgd_data, gd_data, adam_data)

