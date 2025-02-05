import tkinter as tk
from tkinter import filedialog
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from processing import extract_features
from processing import features, features_labels




# Function to load CSV file & extract features from df
def load_csv():
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if file_path:
        # try:
            df = pd.read_csv(file_path)
            window_size = 10
            df = df.iloc[:, 1:]
            df['Label'] = float('nan') #This adds the label column at the end

            pred_features = pd.DataFrame(index=range(int(len(df) / 500)),
                                    columns=['x_mean', 'x_min', 'x_max', 'x_std', 'x_kurtosis', 'x_skew', 'x_median',
                                             'x_quantile', 'x_range', 'x_variance', 'y_mean', 'y_min', 'y_max', 'y_std',
                                             'y_kurtosis', 'y_skew', 'y_median', 'y_quantile', 'y_range', 'y_variance',
                                             'z_mean', 'z_min', 'z_max', 'z_std', 'z_kurtosis', 'z_skew', 'z_median',
                                             'z_quantile', 'z_range', 'z_variance', 'Label'])


            # Extract features from Test Data
            pred_features = extract_features(pred_features, df, window_size)

            pred_features = pred_features.iloc[1:, :]
        # except Exception as e:
        #     print("Error loading CSV file:", e)
    return pred_features


# Function to plot predicted response
def plot_response(data):
    data = data.iloc[:, :-1]
    #print(data)
    #NORMALIZE DATA HERE

    sc = StandardScaler()
    data = sc.fit_transform(data)

    #print(data.shape)

    l_reg = LogisticRegression(max_iter=10000)
    clf = make_pipeline(l_reg)

    print(data[:, :-1])



    clf.fit(features, features_labels)

    y_pred = clf.predict(data[:, :-1])

    print("y_pred new", y_pred)

    # Clear any existing plots
    plt.close()

    plt.plot(y_pred, marker='o', linestyle='-')

    plt.ylabel('Walking (0) or Jumping (1)')
    plt.title('Predicted Response Plot')


    # Convert plot to Tkinter canvas
    canvas = FigureCanvasTkAgg(plt.gcf(), master=root)
    canvas.draw()

    # Place the canvas on the GUI
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    # Destroy the old canvas widget
    if hasattr(plot_response, 'canvas'):
        plot_response.canvas.get_tk_widget().destroy()

    # Save the canvas reference to be used for destruction later
    plot_response.canvas = canvas

    export_csv(y_pred)

# Function to export predicted data to CSV
def export_csv(y_pred):
    file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
    if file_path:
        df = pd.DataFrame()
        df['Predicted_Label'] = y_pred
        df.to_csv(file_path, index=False)

# Function to handle button click
def on_button_click():
    features = load_csv()
    if features is not None:
        plot_response(features)


# Create main window
root = tk.Tk()
root.title("CSV File Loader")

# Set the size of the window
root.geometry("600x400")

# Create a button to load CSV file
load_button = tk.Button(root, text="Load CSV File", command=on_button_click)
load_button.pack(pady=20)

# Start the GUI event loop
root.mainloop()
