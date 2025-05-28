import tkinter as tk
from tkinter import filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image, ImageTk
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr
from statsmodels.graphics.tsaplots import plot_acf

# Initialize GUI
root = tk.Tk()
root.title("TimeGAN Synthetic Data Viewer")
root.state("zoomed")
root.resizable(True, True)

# Screen dimensions
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

# Canvas
canvas_bg = tk.Canvas(root, width=screen_width, height=screen_height)
canvas_bg.pack(fill="both", expand=True)

# Load and display background
bg_image = Image.open("timegan_background.jpg")
bg_image = bg_image.resize((screen_width, screen_height), Image.LANCZOS)
bg_photo = ImageTk.PhotoImage(bg_image)
canvas_bg.create_image(0, 0, image=bg_photo, anchor="nw")

# GUI Elements
apply_smoothing = tk.BooleanVar(root, value=True)  # ✅ Fixed master assignment

# Title
title_label = tk.Label(root, text="TimeGAN Synthetic Data Visualizer",
                       bg="black", fg="white", font=("Helvetica", 18, "bold"))
canvas_bg.create_window(screen_width//2, 30, window=title_label)

# Button row
frame_buttons = tk.Frame(root, bg="black")
canvas_bg.create_window(screen_width//2, 80, window=frame_buttons)

btn_style = {"font": ("Arial", 10), "bg": "#007acc", "fg": "white", "padx": 10, "pady": 5, "bd": 0}

# Metrics Label
metrics_label = tk.Label(root, text="", font=("Arial", 10), bg="black", fg="white")
canvas_bg.create_window(screen_width//2, screen_height - 40, window=metrics_label)

# Frame for displaying plots
frame_canvas = tk.Frame(root, bg="white", bd=2, relief="sunken")
canvas_bg.create_window(screen_width//2, screen_height//2 + 30, window=frame_canvas, width=screen_width-80, height=screen_height-250)

# Button Actions
tk.Button(frame_buttons, text="Load Data", command=lambda: load_data(), **btn_style).pack(side=tk.LEFT, padx=5)
tk.Button(frame_buttons, text="Show KDE Plot", command=lambda: show_plot(plot_kde), **btn_style).pack(side=tk.LEFT, padx=5)
tk.Button(frame_buttons, text="Show Time Series Sample", command=lambda: show_plot(plot_timeseries_sample), **btn_style).pack(side=tk.LEFT, padx=5)
tk.Checkbutton(frame_buttons, text="Apply Smoothing", variable=apply_smoothing,
               bg="black", fg="white", font=("Arial", 10), selectcolor="black").pack(side=tk.LEFT, padx=10)  # ✅ Visible checkbox tick

# Load .npy arrays
data_real, data_synth = None, None

def load_data():
    global data_real, data_synth
    stock_path = filedialog.askopenfilename(title="Select Real Data", filetypes=[("NumPy files", "*.npy")])
    synth_path = filedialog.askopenfilename(title="Select Synthetic Data", filetypes=[("NumPy files", "*.npy")])
    try:
        data_real = np.load(stock_path)
        data_synth = np.load(synth_path)
        messagebox.showinfo("Success", "Data loaded successfully!")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load: {e}")

def plot_kde():
    real_flat = data_real[:, :, 0].flatten()
    synth_flat = data_synth[:, :, 0].flatten()
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.kdeplot(real_flat, label="Real", fill=True, color="navy", ax=ax)
    sns.kdeplot(synth_flat, label="Synthetic", fill=True, color="cyan", ax=ax)
    ax.set_title("Real vs Synthetic KDE Plot")
    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    ax.legend()
    return fig

def plot_timeseries_sample():
    idxs = np.random.choice(len(data_real), size=5, replace=False)
    real_avg = data_real[idxs, :, 0].mean(axis=0)
    synth_avg = data_synth[idxs, :, 0].mean(axis=0)
    if apply_smoothing.get():
        synth_avg = pd.Series(synth_avg).rolling(window=3, min_periods=1).mean().values

    mae = mean_absolute_error(real_avg, synth_avg)
    mse = mean_squared_error(real_avg, synth_avg)
    corr, _ = pearsonr(real_avg, synth_avg)
    metrics_label.config(text=f"MAE: {mae:.4f}  MSE: {mse:.4f}  Pearson Corr: {corr:.4f}")

    fig, axs = plt.subplots(2, 3, figsize=(14, 7))
    fig.suptitle("Real vs Synthetic Time Series Analysis")

    axs[0, 0].plot(real_avg, label="Real", marker='o')
    axs[0, 0].plot(synth_avg, label="Synthetic", linestyle='--', marker='x')
    axs[0, 0].set_title("Time Series")
    axs[0, 0].legend()

    axs[0, 1].hist(real_avg, bins=10, alpha=0.6, label="Real", color='navy')
    axs[0, 1].hist(synth_avg, bins=10, alpha=0.6, label="Synthetic", color='orange')
    axs[0, 1].set_title("Histogram")
    axs[0, 1].legend()

    axs[0, 2].scatter(real_avg, synth_avg, alpha=0.7, edgecolors='k')
    axs[0, 2].set_title("Scatter Comparison")
    axs[0, 2].set_xlabel("Real")
    axs[0, 2].set_ylabel("Synthetic")

    plot_acf(real_avg, ax=axs[1, 0], lags=10)
    axs[1, 0].set_title("Autocorrelation (Real)")

    plot_acf(synth_avg, ax=axs[1, 1], lags=10)
    axs[1, 1].set_title("Autocorrelation (Synthetic)")

    axs[1, 2].axis('off')

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig

def show_plot(plot_func):
    for widget in frame_canvas.winfo_children():
        widget.destroy()
    fig = plot_func()
    if fig:
        canvas = FigureCanvasTkAgg(fig, master=frame_canvas)
        canvas.draw()
        canvas.get_tk_widget().pack()

root.mainloop()
