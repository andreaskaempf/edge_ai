// Send text to a 16x2 LCD with I2C adapter
//
// Pins I2C -> Uno board:
// SCL -> A5
// SDA -> A4
// VCC -> 5V (3.3 V does not work)
// Gnd -> Gnd
//
// AK, 24/08/2025

#include <Arduino.h>

#include <LiquidCrystal_I2C.h>

// Create LCD object: I2C address 0x27, 16 column and 2 rows
LiquidCrystal_I2C lcd(0x27, 16, 2);

// Counter
int n = 0;

void setup() {

  lcd.init();           // initialize the lcd
  lcd.backlight();      // comment out this line for no backlight

  lcd.setCursor(5, 0);  // column 0-15, row 0-1
  lcd.print("Counter"); // text centered on top row
}

void loop() {
  lcd.setCursor(0, 1); // column, row
  lcd.print(n);
  delay(500);
  n += 1;
}
