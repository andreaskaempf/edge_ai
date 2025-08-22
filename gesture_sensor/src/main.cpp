// Gesture sensor, using Arduino Nano BLE Sense Lite, and Tensorflow Lite
//
// Model trained on five gestures, all with device held flat, with USB port pointing away from you:
// Flex: a [flat horizontal] simple circular movement in front of the body, like a hook
// Punch: a forward, straight punch
// Curl: vertical lift with the elbow only [i.e., vertical up then horizontal towards you]
// Side lift: Arm held sideways, rigid and moved up and down from the shoulder [device USB 
//   port pointing away from you, different from curl because just up & down]
// Rotating curl: A curl, but with rotating the wrist while lifting [i.e., device starts flat, 
//   then twisted 90 degrees right during lift, so long edge ends up on top]
//
// Simpler four gestures
// - updown
// - leftright
// - fwdback
// - circle (horizontal plane, clockwise, starting close to body)
//
// Sources:
// https://medium.com/@traiano_1008/a-tinyml-journey-neural-networks-on-cheap-tiny-embedded-microcontrollers-020ea0f30c44
// https://github.com/spaziochirale/ArduTFLite/tree/main/examples/ArduinoNano33BLE_GestureClassifier
//
// AK, 19/08/2025

// Need to install Arduino_LSM9DS1 1.1.1 using Library manager
#include <Arduino_LSM9DS1.h>

// Provided by ArduTFLite, which needs to be installed by Library manager
#include <ArduTFLite.h>

// The model, created by Python script from Keras model
#include "model.h"

// Global variables, for reading input
const float accelerationThreshold = 2.5; // threshold of significant in G's
const int numSamples = 119; // sensors have 119 readings per second, a single gesture
int samplesRead = numSamples;

// For setting up model
const int inputLength = 714; // Dimension of input tensor (6 values * 119 samples)
constexpr int tensorArenaSize = 8 * 1024; // Tensor Arena size
alignas(16) byte tensorArena[tensorArenaSize]; // Tensor Arena memory

// 5 gym gestures
//const char* GESTURES[] = { "Flex", "Punch", "Curl", "Sidelift Curl", "Rotating Curl" };
//#define NUM_GESTURES 5  // Number of gestures

// 4 simpler gestures
const char* GESTURES[] = { "Circle", "Left-right", "Punch", "Up-down" };
#define NUM_GESTURES 4  // Number of gestures

// Set to 1 to just output data for creating training set from serial monitor
// output, or 0 to run inference
#define TRAINING_DATA 0

void setup() {

  // Initialize serial port
  Serial.begin(9600);
  while (!Serial)
    ;

  // Initialize IMU sensor
  if ( !IMU.begin() ) {
    Serial.println("Failed to initialize IMU!");
    while (1);
  }

  // print IMU sampling frequencies
  Serial.print("Accelerometer sampling frequency = ");
  Serial.print(IMU.accelerationSampleRate());
  Serial.println(" Hz");
  Serial.print("Gyroscope sampling frequency = ");
  Serial.print(IMU.gyroscopeSampleRate());
  Serial.println(" Hz");
  Serial.println();

  // If creating input files, print the header, otherwise initialize the model
  #if TRAINING_DATA
  Serial.println("aX,aY,aZ,gX,gY,gZ");
  #else
  Serial.println("Init model..");
  if (!modelInit(model, tensorArena, tensorArenaSize)){
    Serial.println("Model initialization failed!");
    while(true);
  }
  Serial.println("Model initialization done.");
  #endif
}

// Print one sample, for creating data files
void printSample(float aX, float aY, float aZ, float gX, float gY, float gZ) {
  Serial.print(aX, 3);
  Serial.print(',');
  Serial.print(aY, 3);
  Serial.print(',');
  Serial.print(aZ, 3);
  Serial.print(',');
  Serial.print(gX, 3);
  Serial.print(',');
  Serial.print(gY, 3);
  Serial.print(',');
  Serial.print(gZ, 3);
  Serial.println();
}

void loop() {

  float aX, aY, aZ, gX, gY, gZ;

  // Wait for significant motion
  while ( true ) {
    if ( IMU.accelerationAvailable() ) {
      IMU.readAcceleration(aX, aY, aZ);  // variables are treated as pointers
      float aSum = fabs(aX) + fabs(aY) + fabs(aZ);
      if ( aSum >= accelerationThreshold )
        //samplesRead = 0;
        break;
    }
  }

  // Check if the all the required samples have been read since
  // the last time the significant motion was detected
  samplesRead = 0;
  while ( samplesRead < numSamples ) {

    // check if both new acceleration and gyroscope data is available
    if ( IMU.accelerationAvailable() && IMU.gyroscopeAvailable() ) {

      // Read the acceleration and gyroscope data
      IMU.readAcceleration(aX, aY, aZ);
      IMU.readGyroscope(gX, gY, gZ);

      

      // Populate the input tensor
      #if !TRAINING_DATA
      // Normalize sensor data for inference, not for creating training data
      aX = (aX + 4.0) / 8.0;
      aY = (aY + 4.0) / 8.0;
      aZ = (aZ + 4.0) / 8.0;
      gX = (gX + 2000.0) / 4000.0;
      gY = (gY + 2000.0) / 4000.0;
      gZ = (gZ + 2000.0) / 4000.0;
      modelSetInput(aX, samplesRead * 6 + 0);
      modelSetInput(aY, samplesRead * 6 + 1);
      modelSetInput(aZ, samplesRead * 6 + 2); 
      modelSetInput(gX, samplesRead * 6 + 3);
      modelSetInput(gY, samplesRead * 6 + 4);
      modelSetInput(gZ, samplesRead * 6 + 5);
      #endif

      samplesRead++;

      // If creating training data, print the data in CSV format
      #if TRAINING_DATA
      printSample(aX, aY, aZ, gX, gY, gZ);
      if ( samplesRead == numSamples )
        Serial.println(); 

      // If a full set of samples, run inference
      #else
      if ( samplesRead == numSamples ) {

        if( !modelRunInference() ) {
          Serial.println("RunInference Failed!");
          return;
        }

        // Get output values and print as percentages
        for (int i = 0; i < NUM_GESTURES; i++) {
          Serial.print(GESTURES[i]);
          Serial.print(": ");
          Serial.print(modelGetOutput(i)*100, 2);
          Serial.println("%");
        }
        // Blank line after inference, also when outputting data
        Serial.println(); 
      }
      #endif
    }
  }
}


