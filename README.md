 # 📈Time Series Data Generation using Deep Learning
 _Generating realistic synthetic time series data using deep learning models, with a focus on TimeGAN for applications in finance and beyond._
---
## 📌 Table of Contents
- <a href="#overview">Overview</a>

- <a href="#Problem-Statement">Problem Statement</a>

- <a href="#dataset">Dataset</a>

- <a href="#tools--technologies">Tools & Technologies</a>

- <a href="#project-structure">Project Structure</a>

- <a href="#methodology">Methodology</a>

- <a href="#results-and-visualizations">Results and Visualizations</a>

- <a href="#gui-dashboard">GUI Dashboard</a>

- <a href="#how-to-run-this-project">How to Run This Project</a>

- <a href="#conclusion--future-scope">Conclusion & Future Scope</a>

- <a href="#author--contact">Author & Contact</a>

---
<h2><a class="anchor" id="overview"></a>Overview</h2>
This project focuses on the generation of synthetic time series data using deep learning, specifically the TimeGAN (Time-series Generative Adversarial Network) architecture. The objective is to produce high-fidelity synthetic data that accurately reflects the temporal dynamics and statistical properties of real-world time series data. This is particularly valuable in domains where data is scarce, private, or expensive to obtain, such as finance, healthcare, and IoT.

---
<h2><a class="anchor" id="Problem-Statement"></a>Problem Statement</h2>
Many industries rely on time series data but face significant limitations such as missing values, privacy constraints, and restricted access,especially in critical fields like finance, healthcare, and IoT. These limitations can hinder the development of reliable machine learning models that require large and diverse datasets to learn complex temporal patterns.This project aims to address this by:

- Generating realistic synthetic time series data to augment existing datasets.

- Enabling the development of more accurate and reliable models without compromising data privacy.

- Providing a solution for creating large, diverse datasets for training and testing purposes.

- Improving Model Resilience: Producing a range of diverse and uncommon event scenarios that may be less frequent in actual data to develop machine learning models that are more robust and generalizable.

- Promoting Data Sharing and Collaboration: Developing shareable, privacy-respecting synthetic datasets to encourage cooperation and speed up research and development in both academic and industrial settings without breaching confidentiality.

---
<h2><a class="anchor" id="dataset"></a>Dataset</h2>
- The project uses a financial time series dataset, aadr.us.txt sourced from kaggle containing daily stock data.  (https://www.kaggle.com/datasets/borismarjanovic/price-volume-data-for-all-us-stocks-etfs) -- DataSet link for reference

- Univariate time series (single feature per sequence).

- Data saved in .npy format for fast access and compatibility.

---
<h2><a class="anchor" id="tools--technologies"></a>Tools & Technologies</h2>

- **Python**: The core programming language for the project.

**Libraries**:

- **TensorFlow & Keras**: For building and training the TimeGAN model.

- **Pandas & NumPy**: For data manipulation and numerical operations.

- **Matplotlib & Seaborn**: For creating visualizations.

- **Scikit-learn**: For data preprocessing, including MinMaxScaler.

- **Statsmodels**: For time series analysis, such as the ADF test.

- **ydata-synthetic**: For the implementation of the TimeGAN model.

- **Tkinter**: For the development of a graphical user interface (GUI).

---
<h2><a class="anchor" id="project-structure"></a>Project Structure</h2>

```
time-series-data-generation/
│
├── README.md
├── requirements.txt

│
├── notebooks/                  # Jupyter notebooks
│   ├──Time_Series_synthetic_data_generation_with_TimeGAN.ipynb 
│
├── data/
│   └── aadr.us.txt

├── scripts/
│   └── gui.py
```

---
<h2><a class="anchor" id="methodology"></a>Methodology</h2>

- **Data Loading and Preprocessing**: The stock price data is loaded, and the 'Close' prices are extracted. The data is then scaled using MinMaxScaler.

- **Stationarity Check**: The Augmented Dickey-Fuller (ADF) test is used to check for stationarity, and differencing is applied to make the data stationary.

- **Model Training**: The TimeGAN model is configured and trained on the preprocessed data.

- **Synthetic Data Generation**: The trained model is used to generate synthetic time series data.

- **Evaluation**: The quality of the synthetic data is assessed by visually and statistically comparing it to the real data using plots and metrics like ACF and PACF plots,stationarity test.

---
<h2><a class="anchor" id="results-and-visualizations"></a>Results and Visualizations</h2>

- **Real vs. Synthetic Data**: The generated data closely mimics the patterns of the real stock price data.

- **Distribution Comparison**: Histograms and KDE plots show that the distributions of the real and synthetic data are similar.

- **Autocorrelation**: ACF and PACF plots confirm that the temporal dependencies in the real data are preserved in the synthetic data.

- **Metrics**: MAE, MSE, Pearson correlation, PCA/TSNE for diversity.

- **Performance**: Generated data closely matched real data distribution (MAE: 0.0082, MSE: 0.0080, Pearson: 0.4991)

- **Visualization**: Tkinter GUI to load .npy files and plot outputs.

---
<h2><a class="anchor" id="gui-dashboard"></a>GUI Dashboard</h2>

- An interactive GUI built with Tkinter provides a user-friendly way to interact with the model.

- The workflow is straightforward:

1) **Load Data**: Upload a time series file in NumPy (.npy) format.

2) **Train Model**: Train the TimeGAN model on the loaded data.

3) **Generate Data**: Produce synthetic samples from the trained model.

4) **Visualize Results**: Click "Show Time Series Samples" to view a smoothed plot comparing the real data against a generated synthetic sample. Continuously clicking this button will cycle through different generated samples, allowing for a dynamic evaluation of the model's output.
<img width="1363" height="722" alt="image" src="https://github.com/user-attachments/assets/659d398c-7ba6-4c0d-a131-b937bb80d978" />
<img width="1364" height="724" alt="image" src="https://github.com/user-attachments/assets/ad318bf1-573c-41ba-abfc-c3d2292bf48b" />
<img width="1362" height="726" alt="image" src="https://github.com/user-attachments/assets/e2bef32b-2dad-4b45-a1c4-c4cafb5ff966" />



---
<h2><a class="anchor" id="how-to-run-this-project"></a>How to Run This Project</h2>

1. **Clone the repository**:
```bash
git clone https://github.com/yourusername/time-series-data-generation.git
```
2. **Install the required libraries**:
```bash
pip install -r requirements.txt
```
3. **Run the Jupyter Notebook for a detailed walkthrough**:
```bash
notebooks/Time_Series_synthetic_data_generation_with_TimeGAN.ipynb
```
4. **To use the GUI, run the script**:
```bash
python scripts/gui.py
```
---

<h2><a class="anchor" id="conclusion--future-scope"></a>Conclusion & Future Scope</h2>

- **Conclusion**: The project successfully demonstrates the use of TimeGAN to generate high-quality synthetic time series data, providing a viable solution for data augmentation.

- **Future Scope**:

   - Extend the model to handle multivariate time series data.

   - Experiment with other generative models like DoppelGANger.

   - Apply the framework to different domains, such as healthcare and IoT.

---
<h2><a class="anchor" id="author--contact"></a>Author & Contact</h2>

**B Rahul Sharma**  
Data Analyst  
📧 Email: brahulsharma02@gmail.com  
🔗 [LinkedIn](https://www.linkedin.com/in/bukkapatnam-rahulsharma/)












