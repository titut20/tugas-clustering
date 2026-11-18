import tkinter as tk
from tkinter import messagebox, ttk
from sklearn.cluster import KMeans
import pandas as pd

# Dataset contoh untuk data mahasiswa (Fitur: Nilai Akademik, Keaktifan, Absensi)
data_mahasiswa = {
    'Nama': ['Andi', 'Budi', 'Citra', 'Dodi', 'Eka', 'Fani', 'Gita', 'Hadi', 'Indra', 'Joko'],
    'Nilai_Akademik': [85, 60, 90, 70, 50, 95, 75, 65, 88, 72],
    'Keaktifan': [80, 50, 85, 60, 45, 90, 70, 55, 82, 68],
    'Absensi': [90, 70, 95, 75, 60, 98, 80, 65, 92, 78]
}

# Membuat DataFrame pandas
df = pd.DataFrame(data_mahasiswa)

class KMeansApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Aplikasi Clustering K-Means Data Mahasiswa")
        self.root.geometry("650x500")

        # Label Judul Utama
        self.label_title = ttk.Label(root, text="Clustering K-Means Data Mahasiswa (3 Centroid)", font=("Arial", 14, "bold"))
        self.label_title.pack(pady=10)

        # Tombol untuk Menjalankan Proses Clustering
        self.btn_proses = ttk.Button(root, text="Jalankan Proses Clustering", command=self.jalankan_clustering)
        self.btn_proses.pack(pady=5)

        # Tabel (Treeview) untuk menampilkan hasil data dan cluster
        self.tree = ttk.Treeview(root, columns=("Nama", "Nilai_Akademik", "Keaktifan", "Absensi", "Cluster"), show="headings")
        self.tree.heading("Nama", text="Nama Mahasiswa")
        self.tree.heading("Nilai_Akademik", text="Nilai Akademik")
        self.tree.heading("Keaktifan", text="Keaktifan")
        self.tree.heading("Absensi", text="Absensi")
        self.tree.heading("Cluster", text="Cluster (K=3)")
        
        # Pengaturan lebar kolom
        self.tree.column("Nama", width=100, anchor=tk.CENTER)
        self.tree.column("Nilai_Akademik", width=110, anchor=tk.CENTER)
        self.tree.column("Keaktifan", width=90, anchor=tk.CENTER)
        self.tree.column("Absensi", width=90, anchor=tk.CENTER)
        self.tree.column("Cluster", width=100, anchor=tk.CENTER)

        self.tree.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)

    def jalankan_clustering(self):
        try:
            # Memilih kolom fitur numerik untuk proses K-Means
            X = df[['Nilai_Akademik', 'Keaktifan', 'Absensi']]

            # Inisialisasi dan pelatihan model K-Means dengan 3 cluster (centroid)
            kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
            df['Cluster'] = kmeans.fit_predict(X)

            # Membersihkan tampilan tabel sebelum menampilkan data terbaru
            for item in self.tree.get_children():
                self.tree.delete(item)

            # Memasukkan data hasil clustering ke dalam tabel GUI
            for index, row in df.iterrows():
                self.tree.insert("", tk.END, values=(
                    row['Nama'], 
                    row['Nilai_Akademik'], 
                    row['Keaktifan'], 
                    row['Absensi'], 
                    f"Cluster {row['Cluster']}"
                ))

            messagebox.showinfo("Sukses", "Algoritma K-Means berhasil dikelompokkan ke dalam 3 centroid!")
        
        except Exception as e:
            messagebox.showerror("Error", f"Terjadi kesalahan: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = KMeansApp(root)
    root.mainloop()
