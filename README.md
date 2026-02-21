# CollectAccelerometerDatav2

This project is designed to collect accelerometer data from a mobile device using an application, which is then sent to a computer for action prediction based on predefined actions.
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://github.com/sontungkieu/Real-time_detection_of_user-defined_actions)
## Application Repository

The application used for data collection can be found at [CollectAccelerometerDatav2](https://github.com/codemaivanngu/CollectAccelerometerDatav2).

### Current Abilities

The application has the following capabilities:

- **Recognition of Different Actions**: It can recognize various actions such as running, jogging, standing, cycling, etc.
- **Segmentation of Actions in a Routine**: The collected data can be segmented to identify different actions within a routine.

### Future Features

The project aims to implement the following features in the future:

- **Improved Speed and Accuracy of the Model**: Enhancements will be made to optimize the speed and accuracy of the action prediction model.
- **Mobile Device Compatibility**: The goal is to enable the entire process to run seamlessly on a mobile device.

## Usage Guide

To add a new action to the application, follow these steps:

1. **Record Data**: Execute `/recordData.py` to record data from your mobile device. Connect the device and wait for approximately 30 seconds to collect accelerometer data. The recorded data will be saved in `yyyymmddhhmmss.csv`.
   
2. **Clean and Refine Data**: Run `/processData.py` to clean and refine the collected data. You can also label the data at this stage.

3. **Train the Model**: Execute `/model.py` to train the model using the cleaned and labeled data.

Once the model is trained, you can:

- **Run the Application**: Execute `./main.py` to start the application. Connect your device and enjoy the action prediction capabilities!
