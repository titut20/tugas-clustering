import tkinter as tk
from tkinter import messagebox, ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

# Konfigurasi Tema Warna Modern (Dark & Clean)
BG_COLOR = "#0f172a"      # Slate 900
CARD_BG = "#1e293b"       # Slate 800
TEXT_COLOR = "#f8fafc"    # Slate 50
ACCENT_COLOR = "#38bdf8"  # Sky 400
SUCCESS_COLOR = "#4ade80" # Green 500
DANGER_COLOR = "#ef4444"  # Red 500
MUTED_COLOR = "#94a3b8"   # Slate 400

class CompetitivePredictor:
    def __init__(self, root):
        self.root = root
        self.root.title("Sistem Prediksi Kelulusan Mahasiswa - JST Kompetitif")
        self.root.geometry("900" "x" "700")
        self.root.configure(bg=BG_COLOR)

        # Bobot / Prototaip Jaringan Saraf Tiruan Kompetitif (3 Cluster / Neuron)
        # Bentuk: [IPK Sem 1, IPK Sem 2, IPK Sem 3, IPK Sem 4]
        self.weights = np.array([
            [3.70, 3.75, 3.80, 3.85], # Neuron 0: Tepat Waktu (Sangat Memuaskan)
            [3.00, 3.05, 3.10, 3.15], # Neuron 1: Tepat Waktu (Standar/Cukup)
            [2.20, 2.10, 2.05, 2.00]  # Neuron 2: Terlambat / Berisiko
        ])
        
        self.cluster_labels = {
            0: "Lulus Tepat Waktu (Cumlaude / Sangat Memuaskan)",
            1: "Lulus Tepat Waktu (Standar)",
            2: "Berisiko Terlambat / Perlu Perhatian Akademik"
        }
        
        self.cluster_colors = [SUCCESS_COLOR, ACCENT_COLOR, DANGER_COLOR]

        # Header Frame
        header_frame = tk.Frame(root, bg=BG_COLOR)
        header_frame.pack(fill=tk.X, padx=20, pady=15)

        title_label = tk.Label(header_frame, text="Prediksi Kelulusan Mahasiswa - JST Kompetitif", font=("Arial", 16, "bold"), bg=BG_COLOR, fg=TEXT_COLOR)
        title_label.pack(anchor="w")
        subtitle_label = tk.Label(header_frame, text="Algoritma Jaringan Saraf Tiruan Kompetitif (Winner-Take-All / Euclidean Distance)", font=("Arial", 10), bg=BG_COLOR, fg=MUTED_COLOR)
        subtitle_label.pack(anchor="w")

        # Main Container (Grid Layout)
        main_container = tk.Frame(root, bg=BG_COLOR)
        main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        main_container.columnconfigure(0, weight=1)
        main_container.columnconfigure(1, weight=2)
        main_container.rowconfigure(0, weight=1)

        # Panel Kiri: Input dan Hasil
        left_panel = tk.Frame(main_container, bg=CARD_BG, relief=tk.RIDGE, bd=2)
        left_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 10))

        input_frame = tk.Frame(left_panel, bg=CARD_BG)
        input_frame.pack(fill=tk.X, padx=15, pady=15)

        tk.Label(input_frame, text="Masukkan IPK Mahasiswa:", font=("Arial", 11, "bold"), bg=CARD_BG, fg=TEXT_COLOR).pack(anchor="w", pady=(0, 10))

        self.entries = []
        semesters = ["Semster 1", "Semster 2", "Semster 3", "Semster 4"]
        for sem in semesters:
            row_frame = tk.Frame(input_frame, bg=CARD_BG)
            row_frame.pack(fill=tk.X, pady=5)
            tk.Label(row_frame, text=f"{sem}:", width=12, anchor="w", bg=CARD_BG, fg=TEXT_COLOR, font=("Arial", 10)).pack(side=tk.LEFT)
            ent = tk.Entry(row_frame, font=("Arial", 10), width=15)
            ent.pack(side=tk.RIGHT, expand=True, fill=tk.X)
            self.entries.append(ent)

        btn_predict = tk.Button(input_frame, text="Jalankan Prediksi JST", bg=ACCENT_COLOR, fg="#0f172a", font=("Arial", 10, "bold"), relief=tk.FLAT, cursor="hand2", command=self.predict_graduation)
        btn_predict.pack(fill=tk.X, pady=(15, 10))

        # Output / Hasil Frame
        output_frame = tk.Frame(left_panel, bg=CARD_BG)
        output_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=10)

        tk.Label(output_frame, text="Hasil Analisis:", font=("Arial", 11, "bold"), bg=CARD_BG, fg=TEXT_COLOR).pack(anchor="w", pady=(5, 5))
        
        self.result_title = tk.Label(output_frame, text="Status: Belum Diproses", font=("Arial", 10, "bold"), bg=CARD_BG, fg=MUTED_COLOR, justify=tk.LEFT)
        self.result_title.pack(anchor="w", pady=5)

        self.result_status = tk.Label(output_frame, text="Masukkan IPK lalu klik tombol prediksi.", font=("Arial", 9), bg=CARD_BG, fg=TEXT_COLOR, justify=tk.LEFT, wraplength=320)
        self.result_status.pack(anchor="w", pady=5)

        # Panel Kanan: Grafik Matplotlib
        right_panel = tk.Frame(main_container, bg=CARD_BG, relief=tk.RIDGE, bd=2)
        right_panel.grid(row=0, column=1, sticky="nsew")

        # Inisialisasi Grafik Matplotlib
        self.fig, self.ax = plt.subplots(figsize=(5, 4), dpi=100)
        self.fig.patch.set_facecolor(CARD_BG)
        self.ax.set_facecolor(BG_COLOR)

        self.canvas = FigureCanvasTkAgg(self.fig, master=right_panel)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.draw_initial_plot()

    def draw_initial_plot(self):
        self.ax.clear()
        self.ax.set_facecolor(BG_COLOR)
        semesters = ['Sem 1', 'Sem 2', 'Sem 3', 'Sem 4']
        
        # Plot prototipe bobot awal
        for idx, w in enumerate(self.weights):
            self.ax.plot(semesters, w, marker='o', linestyle='--', label=f"Prototipe Cluster {idx+1}")
            
        self.ax.set_title("Grafik Standar Cluster JST", color=TEXT_COLOR, fontsize=11)
        self.ax.set_ylabel("Nilai IPK", color=TEXT_COLOR)
        self.ax.set_ylim(1.5, 4.0)
        self.ax.tick_params(colors=TEXT_COLOR)
        
        for spine in self.ax.spines.values():
            spine.set_color(MUTED_COLOR)
            
        self.ax.legend(facecolor=CARD_BG, edgecolor=MUTED_COLOR, labelcolor=TEXT_COLOR, fontsize=8)
        self.fig.tight_layout()
        self.canvas.draw()

    def predict_graduation(self):
        input_ipk = []
        for i, ent in enumerate(self.entries):
            val_str = ent.get().strip()
            if not val_str:
                messagebox.showerror("Error", f"IPK Semester {i+1} tidak boleh kosong!")
                return
            try:
                val = float(val_str)
                if not (0.0 <= val <= 4.0):
                    raise ValueError
                input_ipk.append(val)
            except ValueError:
                messagebox.showerror("Error", f"IPK Semester {i+1} harus berupa angka antara 0.0 sampai 4.0!")
                return

        # Algoritma Jaringan Saraf Tiruan Kompetitif (Winner-Take-All / Euclidean Distance)
        input_vector = np.array(input_ipk)

        # Hitung jarak Euclidean antara input dengan setiap bobot/prototipe cluster
        distances = [np.linalg.norm(input_vector - w) for w in self.weights]

        # Neuron pemenang adalah yang jaraknya paling minim (Competitive Learning)
        winner_index = np.argmin(distances)

        # Tampilkan Hasil di GUI
        status_text = self.cluster_labels[winner_index]
        color = self.cluster_colors[winner_index]

        self.result_title.config(text=f"Status: Terklasifikasi ke Cluster {winner_index + 1}", fg=color)
        self.result_status.config(text=f"{status_text}\n\nJarak Euclidean terdekat: {distances[winner_index]:.4f}", fg=TEXT_COLOR)

        # Update Visualisasi Grafik
        self.ax.clear()
        self.ax.set_facecolor(BG_COLOR)
        semesters = ['Sem 1', 'Sem 2', 'Sem 3', 'Sem 4']

        # Plot garis prototipe pemenang (tebal)
        for idx, w in enumerate(self.weights):
            if idx == winner_index:
                self.ax.plot(semesters, w, marker='o', linewidth=2.5, color=color, label=f"Cluster Pemenang ({winner_index + 1})")
            else:
                self.ax.plot(semesters, w, marker='o', linestyle='--', alpha=0.4, color=MUTED_COLOR, label=f"Cluster {idx + 1}")

        # Plot data mahasiswa yang diinput
        self.ax.plot(semesters, input_vector, marker='s', markersize=8, linestyle='-', linewidth=2, color=ACCENT_COLOR, label="IPK Mahasiswa")

        self.ax.set_title(f"Hasil Klasifikasi: Cluster {winner_index + 1}", color=TEXT_COLOR, fontsize=11)
        self.ax.set_ylabel("Nilai IPK", color=TEXT_COLOR)
        self.ax.set_ylim(1.5, 4.0)
        self.ax.tick_params(colors=TEXT_COLOR)

        for spine in self.ax.spines.values():
            spine.set_color(MUTED_COLOR)

        self.ax.legend(facecolor=CARD_BG, edgecolor=MUTED_COLOR, labelcolor=TEXT_COLOR, fontsize=8)
        self.fig.tight_layout()
        self.canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = CompetitivePredictor(root)
    root.mainloop()
