import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)


def train_model(learning_rates, num_iterations, error_threshold):
    data = np.genfromtxt('datos.csv', delimiter=';', skip_header=1)
    features = data[:, 1:5]
    output = data[:, 4]
    # synaptic_weights = 2 * np.random.random((4, 1)) - 1

    # Normalizar las características (features)
    min_vals = np.min(features, axis=0)
    max_vals = np.max(features, axis=0)
    normalized_features = (features - min_vals) / (max_vals - min_vals)

    # Normalizar la variable de salida (output)
    min_output = np.min(output)
    max_output = np.max(output)
    normalized_output = (output - min_output) / (max_output - min_output)

    training_inputs = normalized_features
    training_outputs = normalized_output.reshape(-1, 1)

    np.random.seed(1)

    results = []
    
    for learning_rate in learning_rates:
        errors = []
        synaptic_weights = 2 * np.random.random((4, 1)) - 1
        prev_error = np.inf

        for iteration in range(num_iterations):
            input_layer = training_inputs

            outputs = sigmoid(np.dot(input_layer, synaptic_weights))

            error = training_outputs - outputs
            adjustments = error * sigmoid_derivative(outputs)

            synaptic_weights += learning_rate * np.dot(input_layer.T, adjustments)
            average_error = np.mean(np.abs(error))
            errors.append(average_error)

            if average_error < error_threshold:
                break

            # Verificar si el error comienza a aumentar
            if average_error > prev_error:
                break

            prev_error = average_error

        results.append({
            'learning_rate': learning_rate,
            'initial_weights': synaptic_weights,
            'final_weights': synaptic_weights,
            'error': average_error,
            'iterations': iteration,
            'error_history': errors
        })

    return results


def run_training():
    learning_rates = [float(learning_rate_entry.get()) for learning_rate_entry in learning_rate_entries]
    num_iterations = int(num_iterations_entry.get())
    error_threshold = float(error_threshold_entry.get())
    training_results = train_model(learning_rates, num_iterations, error_threshold)

    result_window = tk.Toplevel(window)
    result_window.title("Resultados del entrenamiento")
    result_window.geometry("800x400")

    fig = plt.Figure(figsize=(6, 4), dpi=100)
    ax = fig.add_subplot(111)

    for i, result in enumerate(training_results):
        learning_rate_label = tk.Label(result_window, text="Learning Rate {}: {:.2f}".format(i+1, result['learning_rate']))
        learning_rate_label.pack()

        initial_weights_label = tk.Label(result_window, text="Pesos Iniciales: {}".format(result['initial_weights'].flatten()))
        initial_weights_label.pack()

        final_weights_label = tk.Label(result_window, text="Pesos Finales: {}".format(result['final_weights'].flatten()))
        final_weights_label.pack()

        error_label = tk.Label(result_window, text="Error: {:.6f}".format(result['error']))
        error_label.pack()

        iterations_label = tk.Label(result_window, text="Iteraciones: {}".format(result['iterations']))
        iterations_label.pack()

        ax.plot(result['error_history'], label='Learning Rate: {}'.format(result['learning_rate']))

    ax.set_xlabel('Iteración')
    ax.set_ylabel('Error')
    ax.set_title('Evolución del error durante el entrenamiento')
    ax.legend()

    canvas = FigureCanvasTkAgg(fig, master=result_window)
    canvas.draw()
    canvas.get_tk_widget().pack()


# Crear la ventana principal de la interfaz gráfica
window = tk.Tk()
window.title("Entrenamiento del modelo")
window.geometry("400x300")

# Etiquetas y campos de entrada para los parámetros de entrenamiento
learning_rate_labels = []
learning_rate_entries = []

for i in range(5):
    label = tk.Label(window, text="Learning Rate {}: ".format(i+1))
    label.pack()
    learning_rate_labels.append(label)

    entry = tk.Entry(window)
    entry.pack()
    learning_rate_entries.append(entry)

num_iterations_label = tk.Label(window, text="Cantidad de iteraciones: ")
num_iterations_label.pack()
num_iterations_entry = tk.Entry(window)
num_iterations_entry.pack()

error_threshold_label = tk.Label(window, text="Umbral de error: ")
error_threshold_label.pack()
error_threshold_entry = tk.Entry(window)
error_threshold_entry.pack()

# Botón de entrenamiento
train_button = tk.Button(window, text="Entrenar", command=run_training)
train_button.pack()

# Ejecutar la interfaz gráfica
window.mainloop()
