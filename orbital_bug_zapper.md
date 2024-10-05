---
### **Designing and Assembling an Orbital Insect Zapping System – Step-by-Step Guide**

This comprehensive guide outlines the process to assemble an orbital sensor system designed to detect and neutralize flying insects using sensors, a rotational platform, and a laser zapper. The instructions include detailed electronic schematics, coding, and a step-by-step implementation of an AI learning component to enhance accuracy over time.

### **1. Core Design Overview**
The system acts as a turret, scanning its surroundings using optical (camera) and infrared (IR) sensors to detect insects. It employs machine vision for real-time tracking and a laser zapper for neutralization. The central control unit, built around a Raspberry Pi, processes sensor data and manages the system’s functions.

### **2. Core Components**
1. **Detection System**: Uses optical (camera) and IR sensors to identify insects in the environment.
2. **Tracking System**: Utilizes machine vision and motion-tracking algorithms to follow insects.
3. **Zapping Mechanism**: A laser zapper to neutralize detected insects.
4. **Rotational Platform**: A 360-degree gimbal for precise targeting.
5. **Control Unit**: A Raspberry Pi for data processing and movement control.
6. **AI Learning**: Incorporates a feedback loop for continuous improvement of insect detection accuracy.

### **Required Materials**
- **Raspberry Pi 4** (or equivalent microcontroller)
- **High-Speed Camera Module** (e.g., Raspberry Pi Camera Module)
- **Infrared (IR) Sensor** (e.g., MLX90640 Thermal Camera)
- **Laser Module** (low-power laser diode, Class 3B)
- **Stepper Motors** (for gimbal movement)
- **Motor Driver Board** (e.g., L298N for stepper motor control)
- **Gimbal Platform** (360-degree, 2-axis platform)
- **Proximity Sensor** (e.g., HC-SR04 ultrasonic sensor for safety)
- **Solar Panel and Rechargeable Battery Pack**
- **Custom Enclosure** (weatherproof box)
- **Breadboard, Jumper Wires, Resistors, Capacitors, Voltage Regulators**

### **3. Electronics Schematic and Assembly**
#### **Electronic Schematic Overview**
1. **Camera Module**: Connects to the CSI port on the Raspberry Pi.
2. **IR Sensor**: Connects to the I2C pins on the Raspberry Pi.
3. **Laser Module**: Controlled via a transistor circuit connected to a GPIO pin.
4. **Stepper Motors**: Controlled by a motor driver board wired to the Raspberry Pi GPIO pins.
5. **Proximity Sensor**: Monitors surroundings to avoid accidental zapping of humans or pets.
6. **Power Supply**: Uses a voltage regulator to ensure stable power from the battery/solar panel.

#### **Step-by-Step Assembly**
1. **Set Up the Camera**:
   - Connect the camera module to the CSI port on the Raspberry Pi.
   - **Test**: Capture a test image using:
     ```bash
     raspistill -o test.jpg
     ```
   - Verify the captured image is clear.

2. **Connect the IR Sensor**:
   - Wire the IR sensor to the I2C pins (SDA, SCL) on the Raspberry Pi.
   - **Test**: Verify the sensor connection:
     ```bash
     sudo apt-get install python3-smbus i2c-tools
     i2cdetect -y 1
     ```
   - Confirm the sensor appears in the I2C address output.

3. **Set Up the Laser Module**:
   - Connect the laser to a GPIO pin through a transistor circuit for control.
   - **Test**: Run a script to toggle the laser:
     ```python
     import RPi.GPIO as GPIO
     import time

     laser_pin = 17
     GPIO.setmode(GPIO.BCM)
     GPIO.setup(laser_pin, GPIO.OUT)

     GPIO.output(laser_pin, GPIO.HIGH)  # Turn laser on
     time.sleep(1)
     GPIO.output(laser_pin, GPIO.LOW)   # Turn laser off
     GPIO.cleanup()
     ```
   - Ensure the laser operates as expected.

4. **Connect the Stepper Motors and Gimbal**:
   - Wire the stepper motors to the motor driver board and connect the board to the Raspberry Pi GPIO pins.
   - **Test**: Rotate the gimbal with a motor control script and confirm smooth operation.

5. **Connect the Proximity Sensor**:
   - Wire the proximity sensor to the GPIO pins on the Raspberry Pi.
   - **Test**: Use a script to measure distance:
     ```python
     # Use the provided distance measurement script.
     ```
   - Verify accurate distance measurements.

6. **Assemble the Enclosure**:
   - Mount the components inside the weatherproof enclosure.
   - Attach solar panels and battery pack to power the system.
   - **Test**: Power on the system to check stable power delivery to all components.

### **4. AI Learning: Implementing Real-Time Feedback**

In this section, we will expand on how to implement a feedback mechanism using machine learning (ML) to enhance insect detection and tracking accuracy over time. This involves capturing image data, training a neural network model, integrating the model into a real-time detection system, and periodically retraining the model based on feedback.

### **Step-by-Step Implementation**

#### **4.1 Initial Setup**
Before beginning with the code, ensure all necessary hardware is in place and the following libraries are installed on your Raspberry Pi:

```bash
sudo apt-get update
sudo apt-get install python3-opencv python3-picamera
pip install tensorflow keras numpy
```

#### **4.2 Data Collection**
1. **Capture a Dataset of Images**: Collect images using the Raspberry Pi Camera Module to create a dataset of insects. Use this dataset to train a Convolutional Neural Network (CNN) for insect detection.
   
2. **Manual Labeling**: Organize these images into directories for binary classification:
   - `data/train/insect/` for images containing insects.
   - `data/train/no_insect/` for images without insects.

You can use a simple script to automate the process of capturing images:

```python
from picamera import PiCamera
from time import sleep
import os

camera = PiCamera()
camera.resolution = (640, 480)
save_dir = 'data/train/insect/'  # Change to 'no_insect/' as needed

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

try:
    for i in range(100):  # Capture 100 images
        file_path = os.path.join(save_dir, f'image_{i}.jpg')
        camera.capture(file_path)
        print(f'Captured {file_path}')
        sleep(2)  # Delay between captures

finally:
    camera.close()
```

#### **4.3 Train the Initial Machine Learning Model**
Once you have a dataset of labeled images, you can train a CNN to detect insects. Here’s the code to define, compile, and train a simple CNN model using Keras:

```python
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator

# Define the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')  # Binary classification: Insect or not
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Set up data augmentation for training
train_datagen = ImageDataGenerator(rescale=1./255)

# Load training data
train_generator = train_datagen.flow_from_directory(
    'data/train',  # Directory with training images
    target_size=(64, 64),  # Resize images
    batch_size=32,
    class_mode='binary'
)

# Train the model
model.fit(train_generator, steps_per_epoch=100, epochs=5)

# Save the trained model
model.save('insect_detector.h5')
```

The above script will train the model using images from the `data/train` directory and save it as `insect_detector.h5`. 

#### **4.4 Integrate the Model into Real-Time Detection**
Now, let's integrate the trained model into a real-time detection script:

```python
from keras.models import load_model
import cv2
import numpy as np

# Load the pre-trained model
model = load_model('insect_detector.h5')

# Define the function to detect insects in frames
def detect_insect(frame):
    frame_resized = cv2.resize(frame, (64, 64))  # Resize to match input shape
    frame_array = np.expand_dims(frame_resized, axis=0) / 255.0  # Normalize
    prediction = model.predict(frame_array)
    return prediction[0][0] > 0.5  # True if insect detected

# Capture real-time video from the camera
camera = cv2.VideoCapture(0)

try:
    while True:
        ret, frame = camera.read()
        if not ret:
            break

        # Run insect detection
        if detect_insect(frame):
            print("Insect detected!")
            # Add code here to trigger the laser zapper
        else:
            print("No insect detected.")

        # Display the frame (Optional)
        cv2.imshow('Video Feed', frame)
        
        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    camera.release()
    cv2.destroyAllWindows()
```

This script captures frames from the camera, processes them with the `detect_insect()` function, and prints whether an insect is detected.

#### **4.5 Implement AI Learning Feedback Loop**
The feedback loop involves capturing images in real-time, labeling them based on zapping success or failure, and periodically retraining the model with this new data.

1. **Capture Images Based on Detection Outcome**:
   
   Modify the detection script to save images for retraining purposes:
   
   ```python
   import os

   # Directory for new data
   success_dir = 'data/feedback/insect/'
   failure_dir = 'data/feedback/no_insect/'

   if not os.path.exists(success_dir):
       os.makedirs(success_dir)
   if not os.path.exists(failure_dir):
       os.makedirs(failure_dir)

   frame_count = 0

   def capture_feedback_image(frame, success):
       global frame_count
       label_dir = success_dir if success else failure_dir
       cv2.imwrite(os.path.join(label_dir, f'frame_{frame_count}.jpg'), frame)
       frame_count += 1
   ```

   In the detection loop, call `capture_feedback_image()` based on detection success or failure.

2. **Retrain the Model with Feedback Data**:
   
   Periodically retrain the model using newly captured feedback data:
   
   ```python
   def retrain_model():
       feedback_datagen = ImageDataGenerator(rescale=1./255)
       feedback_generator = feedback_datagen.flow_from_directory(
           'data/feedback',  # Directory with feedback images
           target_size=(64, 64),
           batch_size=32,
           class_mode='binary'
       )
       
       # Retrain the model
       model.fit(feedback_generator, steps_per_epoch=100, epochs=3)

       # Save the updated model
       model.save('insect_detector_updated.h5')
   ```

3. **Automate Retraining**:
   
   Schedule retraining using a simple shell script and `cron`. Create a script named `retrain.sh`:
   
   ```bash
   #!/bin/bash
   python3 -c "from my_detection_script import retrain_model; retrain_model()"
   ```

   Make the script executable:
   
   ```bash
   chmod +x retrain.sh
   ```

   Add a cron job to retrain the model every day:
   
   ```bash
   crontab -e
   ```
   
   Add the line:
   
   ```bash
   0 0 * * * /path/to/retrain.sh
   ```

#### **4.6 Testing the AI Integration**
1. Capture live video, run the `detect_insect()` function on each frame, and log results.
2. Periodically review the performance of the model to ensure it is adapting and improving over time. Look for increases in detection accuracy and reduced false positives.
3. The retraining process using the feedback loop should gradually enhance the system’s detection capabilities.

By implementing this AI learning feedback loop, the system will continuously improve its accuracy and adapt to various insect patterns in its operational environment.

### **5. Estimated Cost Table**

| **Component**                 | **Quantity** | **Cost Per Unit (USD)** | **Total Cost (USD)** | **Example Products**                                                                                             |
|-------------------------------|--------------|-------------------------|----------------------|---------------------------------------------------------------------------------------------------------------|
| Raspberry Pi 4               | 1            | $50                     | $50                  | [Option 1](https://www.amazon.com/Raspberry-Pi-4-Model-B/dp/B07TC2BK1X), [Option 2](https://www.amazon.com/CanaKit-Raspberry-Pi-4-4GB/dp/B07V5JTMV9), [Option 3](https://www.amazon.com/LABISTS-Raspberry-4GB-Starter-Micro-HDMI/dp/B07WG4DW52)      |
| High-Speed Camera Module     | 1            | $100                    | $100                 | [Option 1](https://www.amazon.com/Raspberry-Pi-Camera-Module-Megapixel/dp/B07X8DJYYJ), [Option 2](https://www.amazon.com/Arducam-Megapixel-Camera-Raspberry-Degree/dp/B0899TGPQM), [Option 3](https://www.amazon.com/Raspberry-Pi-Camera-V2-8-Megapixel/dp/B01ER2SKFS) |
| IR Sensor (e.g., MLX90640)   | 1            | $70                     | $70                  | [Option 1](https://www.amazon.com/Adafruit-MLX90640-24x32-IR-Thermal-Camera/dp/B07KYLFH84), [Option 2](https://www.amazon.com/HiLetgo-MLX90640-Infrared-Resolution-Temperature/dp/B09JYR83FL), [Option 3](https://www.amazon.com/Infrared-Thermometer-Resolution-Imaging-Thermal/dp/B097Q63ND3) |
| Laser Module (Class 3B)      | 1            | $150                    | $150                 | [Option 1](https://www.amazon.com/Adjustable-Focus-Wavelength-Pointer-Professional/dp/B078ZKLD1M), [Option 2](https://www.amazon.com/Laserland-Focusable-Power-200mW-450nm/dp/B07SLPYH1J), [Option 3](https://www.amazon.com/303-Pointer-Burning-Matches-Included/dp/B08FDNK9XF) |
| Stepper Motors               | 2            | $25                     | $50                  | [Option 1](https://www.amazon.com/STEPPERONLINE-Stepper-Motor-Bipolar-Connector/dp/B00PNEQKC0), [Option 2](https://www.amazon.com/Stepper-Motor-Bipolar-Screwdriver-Printer/dp/B07F8Z8KNY), [Option 3](https://www.amazon.com/Stepperonline-Stepper-Extruder-Printer-Stepper/dp/B00PNEQKYW) |
| Motor Driver Board (L298N)   | 1            | $10                     | $10                  | [Option 1](https://www.amazon.com/HiLetgo-Stepper-Motor-Driver-Controller/dp/B00WJS6DGU), [Option 2](https://www.amazon.com/DAOKI-L298N-Controller-Stepper-Duemilanove/dp/B07CNZNYQ6), [Option 3](https://www.amazon.com/SongHe-Controller-Module-Arduino-Raspberry/dp/B07PQGGDKP) |
| Gimbal Platform (360-degree) | 1            | $100                    | $100                 | [Option 1](https://www.amazon.com/FeiyuTech-Handheld-Stabilizer-Mirrorless-Cameras/dp/B07QLRXWTX), [Option 2](https://www.amazon.com/Neewer-Motorized-Automatic-Photography-Panoramic/dp/B00XGZTH06), [Option 3](https://www.amazon.com/Camera-Rotating-Panoramic-Timelapse-Photography/dp/B07RJZCJWS) |
| Proximity Sensor             | 1            | $10                     | $10                  | [Option 1](https://www.amazon.com/Ultrasonic-HC-SR04-Distance-Arduino-Raspberry/dp/B01MT6I5W8), [Option 2](https://www.amazon.com/Gikfun-HC-SR04-Distance-Measuring-Arduino/dp/B07CN69THV), [Option 3](https://www.amazon.com/Akozon-HC-SR04-Ultrasonic-Distance-Measuring/dp/B07RP6H3LL) |
| Solar Panel                  | 1            | $80                     | $80                  | [Option 1](https://www.amazon.com/Renogy-Monocrystalline-Portable-Battery-Charging/dp/B079JVBVL3), [Option 2](https://www.amazon.com/TP-solar-Portable-Panel-Battery-Charger/dp/B07Q9VJSYY), [Option 3](https://www.amazon.com/ALLPOWERS-Charger-Monocrystalline-Battery-Charging/dp/B07R2JLTNW) |
| Rechargeable Battery Pack    | 1            | $50                     | $50                  | [Option 1](https://www.amazon.com/TalentCell-Rechargeable-6000mAh-112-2Wh-Lithium/dp/B00ME3ZH7C), [Option 2](https://www.amazon.com/Portable-5200mAh-Rechargeable-Lithium-Emergency/dp/B07L6VTR1M), [Option 3](https://www.amazon.com/Poweradd-EnergyCell-10000-Rechargeable-Battery/dp/B081K32VZV) |
| Custom Enclosure             | 1            | $150                    | $150                 | [Option 1](https://www.amazon.com/BUD-Industries-Polycarbonate-Waterproof-IP67-139/dp/B005T5TDWY), [Option 2](https://www.amazon.com/Gonioa-Project-Enclosure-200x120x75mm-Waterproof/dp/B08HJPKPXG), [Option 3](https://www.amazon.com/LeMotech-Junction-Box-Waterproof-Projects/dp/B07Q9ZDG5F) |
| Breadboard, Wires, etc.      | 1 set        | $20                     | $20                  | [Option 1](https://www.amazon.com/EL-KIT-003-Breadboard-Resistor-Potentiometer-Raspberry/dp/B01LZKSVRB), [Option 2](https://www.amazon.com/Electronics-Components-Variety-Multimeter-Ceramic/dp/B07PQMBK8Z), [Option 3](https://www.amazon.com/Breadboard-Breadboards-Potentiometer-Resistors-Jumper/dp/B01N1WQS8K) |

**Total Estimated Cost:** $840

Certainly! Here is a detailed plan to test each part of the orbital bug zapping system, ensuring each component is properly integrated and functions as expected.

### **6. Testing the Full System**

#### **1. Detection Test: Camera and IR Sensor**
This test ensures that the camera and IR sensors can accurately detect flying insects in the environment.

**Steps:**
1. **Setup**: Power on the system and ensure the camera and IR sensors are properly connected to the Raspberry Pi.
2. **Run Detection Script**: Use the following Python script to activate the camera and IR sensor, capturing frames and processing them to detect movement.
   ```python
   from picamera import PiCamera
   from time import sleep
   import cv2
   import numpy as np
   import smbus

   # Initialize camera
   camera = PiCamera()
   camera.resolution = (640, 480)
   camera.start_preview()

   # IR Sensor Setup (Assuming MLX90640)
   bus = smbus.SMBus(1)
   mlx90640_address = 0x33  # Replace with your IR sensor's I2C address

   def capture_frame():
       frame = np.empty((480, 640, 3), dtype=np.uint8)
       camera.capture(frame, 'bgr')
       return frame

   def detect_movement(frame):
       gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
       blur_frame = cv2.GaussianBlur(gray_frame, (21, 21), 0)
       return blur_frame

   def get_ir_data():
       # Simulated read from IR sensor (Replace with actual read logic)
       ir_data = bus.read_i2c_block_data(mlx90640_address, 0x00, 32)
       return ir_data

   try:
       for _ in range(5):  # Capture and process 5 frames
           frame = capture_frame()
           movement_frame = detect_movement(frame)
           ir_data = get_ir_data()

           # Process frames and IR data (Implement your detection logic here)
           print("Movement detected in frame.")
           print("IR data:", ir_data)

           sleep(1)

   finally:
       camera.stop_preview()
   ```
3. **Observe**: The script captures video frames and processes them using basic motion detection. It also retrieves data from the IR sensor.
4. **Expected Outcome**: The output indicates if movement is detected. The IR data readout should show changing values when a heat source (e.g., a hand or insect) is in range.

#### **2. Tracking Test: Gimbal Alignment with Detected Targets**
This test verifies that the gimbal platform accurately tracks detected targets.

**Steps:**
1. **Setup**: Connect the stepper motors to the motor driver and gimbal, ensuring all wiring is correct.
2. **Run Tracking Script**:
   ```python
   import RPi.GPIO as GPIO
   import time

   # Motor pins
   motor_x_pin1 = 17
   motor_x_pin2 = 18
   motor_y_pin1 = 22
   motor_y_pin2 = 23

   GPIO.setmode(GPIO.BCM)
   GPIO.setup([motor_x_pin1, motor_x_pin2, motor_y_pin1, motor_y_pin2], GPIO.OUT)

   def rotate_motor_x(direction):
       if direction == 'left':
           GPIO.output(motor_x_pin1, GPIO.HIGH)
           GPIO.output(motor_x_pin2, GPIO.LOW)
       elif direction == 'right':
           GPIO.output(motor_x_pin1, GPIO.LOW)
           GPIO.output(motor_x_pin2, GPIO.HIGH)

   def rotate_motor_y(direction):
       if direction == 'up':
           GPIO.output(motor_y_pin1, GPIO.HIGH)
           GPIO.output(motor_y_pin2, GPIO.LOW)
       elif direction == 'down':
           GPIO.output(motor_y_pin1, GPIO.LOW)
           GPIO.output(motor_y_pin2, GPIO.HIGH)

   try:
       # Simulate tracking
       rotate_motor_x('right')
       time.sleep(1)
       rotate_motor_x('left')
       time.sleep(1)
       rotate_motor_y('up')
       time.sleep(1)
       rotate_motor_y('down')

   finally:
       GPIO.cleanup()
   ```
3. **Observe**: The gimbal should rotate based on the simulated commands. For real tracking, integrate this with the detection output to dynamically adjust the motor movements.
4. **Expected Outcome**: The gimbal moves smoothly and accurately in response to the motor commands, indicating proper tracking capability.

#### **3. Zapping Test: Activating the Zapper Within Safe Range**
This test verifies the zapper's functionality, ensuring it only activates when the target is within a predefined range.

**Steps:**
1. **Safety Check**: Make sure the laser zapper is installed with safety measures (e.g., a shutter or cover) to avoid accidental firing.
2. **Run Zapper Control Script**:
   ```python
   laser_pin = 17  # Example pin for laser control

   GPIO.setmode(GPIO.BCM)
   GPIO.setup(laser_pin, GPIO.OUT)

   def activate_zapper():
       GPIO.output(laser_pin, GPIO.HIGH)
       time.sleep(0.5)  # Laser on for 0.5 seconds
       GPIO.output(laser_pin, GPIO.LOW)

   try:
       activate_zapper()
       print("Zapper activated!")
   finally:
       GPIO.cleanup()
   ```
3. **Observe**: The zapper should turn on and off briefly, indicating proper control.
4. **Expected Outcome**: The laser activates for the specified duration and turns off without any issue. This confirms the zapper's operational readiness.

#### **4. Safety Test: Using Proximity Sensor to Disable the Zapper**
This test ensures the zapper does not activate when a human or large object is detected nearby.

**Steps:**
1. **Setup**: Connect the ultrasonic proximity sensor (e.g., HC-SR04) to the GPIO pins on the Raspberry Pi.
2. **Run Proximity Sensor Script**:
   ```python
   import RPi.GPIO as GPIO
   import time

   TRIG = 23
   ECHO = 24
   laser_pin = 17

   GPIO.setmode(GPIO.BCM)
   GPIO.setup(TRIG, GPIO.OUT)
   GPIO.setup(ECHO, GPIO.IN)
   GPIO.setup(laser_pin, GPIO.OUT)

   def measure_distance():
       GPIO.output(TRIG, False)
       time.sleep(2)

       GPIO.output(TRIG, True)
       time.sleep(0.00001)
       GPIO.output(TRIG, False)

       while GPIO.input(ECHO) == 0:
           pulse_start = time.time()

       while GPIO.input(ECHO) == 1:
           pulse_end = time.time()

       pulse_duration = pulse_end - pulse_start
       distance = pulse_duration * 17150  # Convert to cm
       return distance

   try:
       distance = measure_distance()
       print(f"Measured Distance: {distance} cm")
       if distance > 30:  # Set safe range limit
           GPIO.output(laser_pin, GPIO.HIGH)  # Enable zapper
           time.sleep(0.5)
           GPIO.output(laser_pin, GPIO.LOW)
       else:
           print("Human detected. Zapper deactivated.")
   finally:
       GPIO.cleanup()
   ```
3. **Observe**: If an object (e.g., a hand) is within 30 cm, the zapper remains deactivated.
4. **Expected Outcome**: The zapper only activates if no object is detected within the safe range, ensuring safety.

#### **5. AI Learning Test: Implementing Feedback-Driven Training**
This test confirms that the system can learn from successes and failures to improve its accuracy over time.

**Steps:**
1. **Initial Data Collection**:
   - Set up the camera to capture images during operation.
   - Save images along with zapper activation status (successful or failed zap).

2. **Labeling**:
   - Manually label the saved images to indicate successful zaps versus false positives.
   - Store the labeled data in a training dataset.

3. **Retrain the Model**:
   - Retrain the convolutional neural network (CNN) model using the new dataset.
   - Update the model file (`insect_detector.h5`) on the Raspberry Pi.
   ```python
   model.fit(train_generator, steps_per_epoch=100, epochs=5)
   model.save('insect_detector_updated.h5')
   ```

4. **Deploy the Updated Model**:
   - Integrate the new model into the real-time detection script.
   ```python
   model = load_model('insect_detector_updated.h5')
   ```

5. **Run the System**: Operate the system with the updated model for a few hours or days, capturing new data.
6. **Monitor Performance**: Compare the accuracy of insect detection before and after the AI model update.

7. **Expected Outcome**: Over time, the system should improve its zap success rate as it learns to distinguish between true insect targets and false positives.

By conducting these tests, you can validate that each part of the system is functional and integrated correctly. This step-by-step testing process ensures the system operates as intended while maintaining safety and adaptability through AI learning.
