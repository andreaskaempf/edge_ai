# Gesture sensor, using Arduino Nano BLE Sense Lite, and Tensorflow Lite.
#
# Model trained on five gestures, all created by moving the Arduino device,
# with device held flat, with USB port pointing away from you, and copying
# the serial output into a separate CSV file for each movement:
# - Flex: a [flat horizontal] simple circular movement in front of the body,
#   like a hook
# - Punch: a forward, straight punch
# - Curl: vertical lift with the elbow only [i.e., vertical up then 
#   horizontal towards you]
# - Side lift: Arm held sideways, rigid and moved up and down from the 
#   shoulder [device USB port pointing away from you, different from curl 
#   because just up & down]
# - Rotating curl: A curl, but with rotating the wrist while lifting [i.e., 
#   device starts flat, then twisted 90 degrees right during lift, so long 
#   edge ends up on top]
#
# Simpler four gestures
# - updown
# - leftright
# - fwdback
# - circle (horizontal plane, clockwise, starting close to body)
#
# Sources:
# https://medium.com/@traiano_1008/a-tinyml-journey-neural-networks-on-cheap-tiny-embedded-microcontrollers-020ea0f30c44
# https://github.com/spaziochirale/ArduTFLite/tree/main/examples/ArduinoNano33BLE_GestureClassifier
#
# This Python file ingest the CSV files of the data created by making each
# of these gestures 10 times, trains a TensorFlow model, then converts that
# model to a tflite/micro model and outputs the model in hex format to
# src/model.h

import os
import numpy as np
import pandas as pd
import tensorflow as tf

from plots import *

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

# Split data into training, testing, and validation sets
def split_data(inputs, outputs, train_split=0.6, test_split=0.2):

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


# Create and train the neural network model
def create_and_train_model(inputs_train, outputs_train, inputs_validate, outputs_validate, num_gestures, epochs=600):

    # Define the model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(50, activation='relu'),
        tf.keras.layers.Dense(15, activation='relu'),
        tf.keras.layers.Dense(num_gestures, activation='softmax')
    ])
    
    # Compile and train the model
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    history = model.fit(
        inputs_train, outputs_train,
        epochs=epochs,
        batch_size=1,   # why?
        validation_data=(inputs_validate, outputs_validate)
    )
    
    return model, history

# Generate a C header file from a TFLite model
def generate_header_file(tflite_model_path, header_path):

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
