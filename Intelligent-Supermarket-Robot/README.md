# Intelligent Supermarket Service Robot 🤖

**A Smart Shopping Cart with Human-Following, Automated Billing, and Demand Prediction.**

## 📄 Project Overview
This project is an **Intelligent Supermarket Service Robot** designed to revolutionize the shopping experience. It integrates a smart shopping cart that autonomously follows the customer, detects products using computer vision, generates bills instantly, and uses machine learning for inventory demand forecasting.

## 🌟 Key Features

| Feature | Technology Used | Screenshot/Visualization |
| :--- | :--- | :--- |
| **Smart Product Detection** | Python, OpenCV, Edge Impulse (Raspberry Pi 4) | The system achieved high accuracy in identifying items like Coke, Lays, and Kurkure. |
| **Automated Billing System** | C Programming (Console Application) | The C program manages transactions, invoices, and cart modification. |
| **Demand Prediction** | Python, Scikit-Learn (Linear Regression) | Analyzes historical sales data to forecast required stock levels for the next month. |
| **Human Following** | Arduino Uno, Ultrasonic & IR Sensors | Code manages the motor movement based on sensor inputs to follow the customer. |

## 📊 Results and Performance

### Model Accuracy
The object detection model was rigorously tested, yielding high accuracy for various products.
![Model Accuracy Results](images/Screenshot%202024-03-14%20013212.png)

### Demand Forecasting
The Linear Regression model generates a forecast for daily product demand, visualized here for clarity.
![Predicted Daily Demand Bar Chart](images/Screenshot%202024-03-14%20013250.png)
Demand prediction console output:
![Predicted Demand Console Output](images/Screenshot%202024-03-14%20013440.png)

### Billing System Interface
The user interface for the console-based billing system.
![Billing System Interface](images/Screenshot%202024-03-14%20013057.png)

## 🛠️ Tech Stack & Hardware
* **Code:** Python, C, C++ (Arduino)
* **ML/Vision:** Scikit-Learn, Pandas, OpenCV, Edge Impulse
* **Hardware:** Raspberry Pi 4 Model B, Arduino Uno, Ultrasonic (HC-SR04), IR Sensors, L298N Motor Driver.

## 🚀 How to Run the Code

1.  **Clone the Repository:** Download all the files to your local machine.
2.  **Install Python Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run Prediction Model:**
    ```bash
    python src/python/predictionforMarch.py
    ```
4.  **Run Detection Model:** (Requires the Edge Impulse model file and live camera feed)
    ```bash
    python src/python/detection.py <path_to_model.eim>
    ```
5.  **Run Billing System:** (Compile using GCC on Windows/Linux)
    ```bash
    gcc src/billing/billing_system.c -o billing
    ./billing
    ```

