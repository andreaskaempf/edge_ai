# Plots created after model training

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_gesture_data(filename, gesture_name):
    """Plot acceleration and gyroscope data for a gesture."""
    # Read the data
    df = pd.read_csv("data/" + filename)
    print(df.head())
    index = range(1, len(df['aX']) + 1)
    
    # Set figure size
    plt.rcParams["figure.figsize"] = (20, 10)
    
    # Plot acceleration data
    plt.figure()
    for axis, color in [('aX', 'g'), ('aY', 'b'), ('aZ', 'r')]:
        plt.plot(index, df[axis], f'{color}.', label=axis[1].lower(), 
                linestyle='solid', marker=',')
    plt.title(f"{gesture_name} Acceleration")
    plt.xlabel(f"{gesture_name} Sample #")
    plt.ylabel(f"{gesture_name} Acceleration (G)")
    plt.legend()
    plt.show()
    
    # Plot gyroscope data
    plt.figure()
    for axis, color in [('gX', 'g'), ('gY', 'b'), ('gZ', 'r')]:
        plt.plot(index, df[axis], f'{color}.', label=axis[1].lower(), 
                linestyle='solid', marker=',')
    plt.title(f"{gesture_name} Gyroscope")
    plt.xlabel(f"{gesture_name} Sample #")
    plt.ylabel(f"{gesture_name} Gyroscope (deg/sec)")
    plt.legend()
    plt.show()


# Load and preprocess gesture data from CSV files
def load_and_preprocess_data(gestures, samples_per_gesture):
    inputs = []
    outputs = []
    one_hot_encoded_gestures = np.eye(len(gestures))
    
    for gesture_index, gesture in enumerate(gestures):
        print(f"Processing index {gesture_index} for gesture '{gesture}'.")
        output = one_hot_encoded_gestures[gesture_index]
        
        df = pd.read_csv("data/" + gesture + ".csv")
        num_recordings = int(df.shape[0] / samples_per_gesture)
        print(f"\tThere are {num_recordings} recordings of the {gesture} gesture.")
        
        for i in range(num_recordings):
            tensor = []
            for j in range(samples_per_gesture):
                index = i * samples_per_gesture + j
                # Normalize the input data
                tensor += [
                    (df['aX'][index] + 4) / 8,
                    (df['aY'][index] + 4) / 8,
                    (df['aZ'][index] + 4) / 8,
                    (df['gX'][index] + 2000) / 4000,
                    (df['gY'][index] + 2000) / 4000,
                    (df['gZ'][index] + 2000) / 4000
                ]
            inputs.append(tensor)
            outputs.append(output)
    
    return np.array(inputs), np.array(outputs)


def plot_training_history(history, skip_epochs=100):
    """Plot training history including loss and MAE."""
    plt.rcParams["figure.figsize"] = (20, 10)
    
    # Plot loss
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    
    plt.figure()
    plt.plot(epochs, loss, 'g.', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    
    # Plot loss (skipping initial epochs)
    plt.figure()
    plt.plot(epochs[skip_epochs:], loss[skip_epochs:], 'g.', label='Training loss')
    plt.plot(epochs[skip_epochs:], val_loss[skip_epochs:], 'b.', label='Validation loss')
    plt.title('Training and validation loss (after initial epochs)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    
    # Plot MAE
    plt.figure()
    mae = history.history['mae']
    val_mae = history.history['val_mae']
    plt.plot(epochs[skip_epochs:], mae[skip_epochs:], 'g.', label='Training MAE')
    plt.plot(epochs[skip_epochs:], val_mae[skip_epochs:], 'b.', label='Validation MAE')
    plt.title('Training and validation mean absolute error')
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.legend()
    plt.show()

def plot_predictions(model, inputs_test, outputs_test, gestures):
    """Plot model predictions against actual values."""
    predictions = model.predict(inputs_test)
    print("predictions =\n", np.round(predictions, decimals=3))
    print("actual =\n", outputs_test)
    
    plt.figure(figsize=(20, 10))
    for i, gesture in enumerate(gestures):
        plt.subplot(2, 3, i+1)
        plt.title(f'Gesture: {gesture}')
        sample_indices = range(len(outputs_test))
        plt.plot(sample_indices, outputs_test[:, i], 'b.', label='Actual', alpha=0.5)
        plt.plot(sample_indices, predictions[:, i], 'r.', label='Predicted', alpha=0.5)
        plt.xlabel('Sample Index')
        plt.ylabel('Probability')
        plt.legend()
    plt.tight_layout()
    plt.show()

