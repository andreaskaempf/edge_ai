# Ingest the data files created moving device

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

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

# List of gestures to plot
gestures_old = [
    ("flex.csv", "Flex"),
    ("punch.csv", "Punch"),
    ("curl.csv", "Curl"),
    ("side_lift.csv", "Sidelift Curl"),
    ("rot_curl.csv", "Rotating Curl"),
]

gestures = [
    ("circle.csv", "Circle"),
    ("leftright.csv", "Left-right"),
    ("punch.csv", "Punch"),
    ("updown.csv", "Up-down"),
]

GESTURES = ["circle", "leftright", "punch", "updown"]

# Plot data for each gesture
#for filename, gesture_name in gestures:
#    plot_gesture_data(filename, gesture_name)

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

def split_data(inputs, outputs, train_split=0.6, test_split=0.2):
    """Split data into training, testing, and validation sets."""
    num_inputs = len(inputs)
    randomize = np.arange(num_inputs)
    np.random.shuffle(randomize)
    
    inputs = inputs[randomize]
    outputs = outputs[randomize]
    
    train_size = int(train_split * num_inputs)
    test_size = int(test_split * num_inputs + train_size)
    
    inputs_train, inputs_test, inputs_validate = np.split(inputs, [train_size, test_size])
    outputs_train, outputs_test, outputs_validate = np.split(outputs, [train_size, test_size])
    
    return (inputs_train, inputs_test, inputs_validate), (outputs_train, outputs_test, outputs_validate)

def create_and_train_model(inputs_train, outputs_train, inputs_validate, outputs_validate, num_gestures, epochs=600):
    """Create and train the neural network model."""
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(50, activation='relu'),
        tf.keras.layers.Dense(15, activation='relu'),
        tf.keras.layers.Dense(num_gestures, activation='softmax')
    ])
    
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    history = model.fit(
        inputs_train, outputs_train,
        epochs=epochs,
        batch_size=1,
        validation_data=(inputs_validate, outputs_validate)
    )
    
    return model, history

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

def generate_header_file(tflite_model_path, header_path):
    """Generate a C header file from a TFLite model."""
    # Read the TFLite model binary
    with open(tflite_model_path, 'rb') as f:
        model_bytes = f.read()
    
    # Convert bytes to hex representation
    hex_lines = []
    for i, byte in enumerate(model_bytes):
        if i % 12 == 0:
            hex_lines.append('\n  ')
        hex_lines.append(f'0x{byte:02x},')
    
    # Write the header file
    with open(header_path, 'w') as f:
        f.write('const unsigned char model[] = {')
        f.write(''.join(hex_lines))
        f.write('\n};\n')
        f.write(f'const unsigned int model_len = {len(model_bytes)};\n')

def main():
    # Set random seeds for reproducibility
    SEED = 1337
    #np.random.seed(SEED)
    #tf.random.set_seed(SEED)
    
    # Define gestures and parameters
    SAMPLES_PER_GESTURE = 119
    
    print(f"TensorFlow version = {tf.__version__}\n")
    
    # Load and preprocess data
    inputs, outputs = load_and_preprocess_data(GESTURES, SAMPLES_PER_GESTURE)
    print("Data set parsing and preparation complete.\n")
    
    # Split data
    (inputs_train, inputs_test, inputs_validate), (outputs_train, outputs_test, outputs_validate) = split_data(inputs, outputs)
    print("Data set randomization and splitting complete.")
    
    # Create and train model
    model, history = create_and_train_model(inputs_train, outputs_train, inputs_validate, outputs_validate, len(GESTURES))
    print("Training and Model building is complete.\n")
    
    # Plot training history
    print("Graphing the model loss function ...\n")
    plot_training_history(history)
    
    # Plot predictions
    print("Testing: Using the model to predict the gesture from the test data set ...")
    plot_predictions(model, inputs_test, outputs_test, GESTURES)
    print("Testing is complete.")
    
    # Convert model to TFLite
    print("\nConverting model to TFLite format...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    
    # Save the model, required for conversion
    open("gesture_model.tflite", "wb").write(tflite_model)
    basic_model_size = os.path.getsize("gesture_model.tflite")
    print("Model is %d bytes" % basic_model_size)
    
    # Generate header file
    header_path = 'src/model.h'
    generate_header_file('gesture_model.tflite', header_path)
    model_h_size = os.path.getsize(header_path)
    print(f"Model converted to {header_path} src/model.h ({model_h_size:,} bytes)")

if __name__ == "__main__":
    main()
