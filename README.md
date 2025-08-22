# Edge AI Applications

This repository contains experimental implementations of AI models that
run on edge devices such as microcontrollers. Will include implementations of
the exercises the TinyML course on EdX, among others. I am using the
Arduino Nano 33 BLE Sense Lite, which comes with the TinyML Learning Kit
available on Amazon, but intend to use the ESP32 and others as well.

I terms of software stack, the Arduino IDE can be used, but the following
instructions assume that Platform IO is installed as well.

## Set-up on Arch Linux

Install Microsoft VS Code
* uninstall default Arch Linux version of 'code' if installed
* download binary VS code as .tar.gz, install in /opt
* add path to ~/.bashrc: export PATH=$PATH:/opt/VSCode-linux-x64/bin

Install Arch Linux platformio packages
* pacman -S platformio-core platformio-core-udev
* type 'pio' to test
* do this BEFORE running code

Install Platform IO extension in VS Code
* run ./code
* click extensions icon, search for 'platformio ide' and install it
* exit code

Create a project
* create and enter project directory
* find board ID using: pio boards <keyword>
* pio project init --board <board ID>
* if you use IDE to create project, it will be in ~/Documents/PlatformIO/Projects

Build and upload
* edit src/main.cpp
* add any Arduino libraries to platformio.ini, e.g.,:
    lib_deps = Arduino_LSM9DS1
* to build: `pio run`
* to upload: `pio run -t upload`
* or use buttons in VSCode, including the one for serial monitor

