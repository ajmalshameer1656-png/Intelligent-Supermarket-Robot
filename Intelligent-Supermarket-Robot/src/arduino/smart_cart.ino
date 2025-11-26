// Define pins 
const int trigPin = 9; 
const int echoPin = 10; 
// Define pins 
const int leftIRPin = 2;   // Digital pin for left IR sensor 
const int rightIRPin = 3;  // Digital pin for right IR sensor 
const int leftMotorPin1 = 4; 
const int leftMotorPin2 = 5; 
const int rightMotorPin1 = 6; 
const int rightMotorPin2 = 7; 
// Define variables 
long duration; 
int distance; 
void setup() { 
// Initialize serial communication 
Serial.begin(9600); 
// Set the trigPin as an OUTPUT 
pinMode(trigPin, OUTPUT); 
// Set the echoPin as an INPUT 
pinMode(echoPin, INPUT); 
// Set the motor control pins as OUTPUT 
pinMode(leftMotorPin1, OUTPUT); 
pinMode(leftMotorPin2, OUTPUT); 
pinMode(rightMotorPin1, OUTPUT); 
pinMode(rightMotorPin2, OUTPUT); 
// Set IR sensor pins as INPUT 
pinMode(leftIRPin, INPUT); 
pinMode(rightIRPin, INPUT); 
// Set motor control pins as OUTPUT 
pinMode(leftMotorPin1, OUTPUT); 
pinMode(leftMotorPin2, OUTPUT); 
pinMode(rightMotorPin1, OUTPUT); 
pinMode(rightMotorPin2, OUTPUT); 
} 
void loop() { 
// Clear the trigPin  
digitalWrite(trigPin, LOW); 
delayMicroseconds(2); 
// Set the trigPin HIGH for 10 microseconds 
digitalWrite(trigPin, HIGH); 
delayMicroseconds(10); 
digitalWrite(trigPin, LOW); 
// Read the echoPin, calculate the distance in centimeters 
duration = pulseIn(echoPin, HIGH); 
distance = duration * 0.034 / 2; 
// Print the distance on the serial monitor 
Serial.print("Distance: "); 
Serial.print(distance); 
Serial.println(" cm"); 
// Check if the distance is within the specified range 
if (distance >= 5 && distance <= 10) { 
// If the distance is within the range, run the line follower 
lineFollower(); 
}  
else { 
// If the distance is not within the range, stop the motors 
stopMotors(); 
}  
// Delay before the next reading 
} 
void lineFollower() { 
// Write your line follower logic here 
// Read IR sensor values 
int leftIRValue = digitalRead(leftIRPin); 
int rightIRValue = digitalRead(rightIRPin); 
// Print sensor values (for debugging) 
Serial.print("Left IR Value: "); 
Serial.println(leftIRValue); 
Serial.print("Right IR Value: "); 
Serial.println(rightIRValue); 
// Check sensor readings and control motors accordingly 
if (leftIRValue == HIGH && rightIRValue == HIGH) { 
// Both sensors detect the line, move forward 
moveForward(); 
} else if (leftIRValue == HIGH && rightIRValue == LOW) { 
// Only left sensor detects the line, turn right 
turnRight(); 
} else if (leftIRValue == LOW && rightIRValue == HIGH) { 
// Only right sensor detects the line, turn left 
turnLeft(); 
}  
else { 
// Both sensors do not detect the line, stop 
turnLeft(); 
delay(100); 
turnRight(); 
} 
} 
void moveForward() { 
digitalWrite(leftMotorPin1, HIGH); 
digitalWrite(leftMotorPin2, LOW); 
digitalWrite(rightMotorPin1, HIGH); 
digitalWrite(rightMotorPin2, LOW); 
} 
void turnLeft() { 
digitalWrite(leftMotorPin1, HIGH); 
digitalWrite(leftMotorPin2, LOW); 
digitalWrite(rightMotorPin1, LOW); 
digitalWrite(rightMotorPin2, LOW); 
} 
void turnRight() { 
digitalWrite(leftMotorPin1, LOW); 
digitalWrite(leftMotorPin2, LOW); 
digitalWrite(rightMotorPin1, HIGH); 
digitalWrite(rightMotorPin2, LOW);   
} 
void stopMotors() { 
digitalWrite(leftMotorPin1, LOW); 
digitalWrite(leftMotorPin2, LOW); 
digitalWrite(rightMotorPin1, LOW); 
digitalWrite(rightMotorPin2, LOW); 
}