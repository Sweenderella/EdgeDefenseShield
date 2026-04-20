# EdgeDefenseShield
Real-time Multimodal threat detection System comparing Deep Neural Network and Spiking Neural Network models Performance, Energy Usage and Latency Metrics

Edge DefenseShield - Real-time Multimodal Threat Detection System
Comparing Deep Neural Network (DNN) and Spiking Neural Network (SNN) Performance

Authors: Sweety Ramnani and Zarin Shejuti
Department of Computer Science, University of South Carolina

Overview

Edge DefenseShield is a real-time threat detection system that compares Deep Neural Networks (DNN) and Spiking Neural Networks (SNN) side-by-side for audio-based threat detection in urban environments. The system processes audio input (MFCC features) and video feed simultaneously, providing a comprehensive dashboard for performance comparison.

Features

- Audio Classification: Detects 10 urban sound classes including gunshots, sirens, and car horns
- Live Video Feed: Real-time video with detection overlays
- DNN vs SNN Comparison: Side-by-side model performance analysis
- Real-time Inference: Process audio from Raspberry Pi V2
- Performance Metrics: Latency, energy usage, and accuracy tracking
- Threat Alert System: Immediate alerts for high-priority threats
- Model Agreement Analysis: Compare DNN and SNN predictions

Detected Classes

gun_shot - CRITICAL - Gunfire detection
siren - HIGH - Emergency vehicle sirens
car_horn - MEDIUM - Vehicle horn alerts
dog_bark - LOW - Dog barking
street_music - LOW - Street performers/music
air_conditioner - INFO - Background noise
children_playing - INFO - Children activity
drilling - INFO - Construction noise
engine_idling - INFO - Vehicle engine noise
jackhammer - INFO - Construction equipment

System Architecture

Raspberry Pi V2 (Edge Device):
- Audio Input via USB Microphone
- MFCC Feature Extraction
- DNN and SNN Models running in parallel
- Video Feed via USB Camera
- TCP Socket output on Port 5555

Laptop Monitor Dashboard:
- Live Camera Feed display
- DNN Results panel
- SNN Results panel
- Threat alerts and detection logs
- Performance statistics and charts

Setup and Installation

On Raspberry Pi:

1. Clone the repository:
   git clone https://github.com/Sweenderella/EdgeDefenseShield.git

2. Run the setup script:
   ./setup_raspberry_pi.sh

3. Activate virtual environment:
   source edgeshield_env/bin/activate

4. Run the detector:
   python3 raspberry_pi_detector.py

On Laptop Monitor:

1. Ensure Python 3.8+ is installed

2. Install requirements:
   pip install opencv-python pillow matplotlib numpy

3. Run the monitor:
   python3 laptop_monitor.py

4. Make sure the laptop IP is accessible from Raspberry Pi

File Structure

EdgeDefenseShield/
├── laptop_monitor.py              # Laptop dashboard application
├── raspberry_pi_detector.py       # Main Pi detector with DNN+SNN
├── best_dnn_multimodal_v2.pth    # Trained DNN model weights
├── best_snn_multimodal_v2.pth    # Trained SNN model weights
├── setup_raspberry_pi.sh         # Environment setup script
├── model_monitor.py               # Model monitoring utility
└── logs/                          # Detection log files

Requirements

Python 3.8 or higher
OpenCV (opencv-python)
Pillow (PIL)
Matplotlib
NumPy

For Raspberry Pi additional requirements:
TensorFlow 2.0+
PyTorch 1.9+
librosa (for MFCC extraction)
PortAudio (PyAudio)

Usage Example

Start the laptop monitor first:
python3 laptop_monitor.py

The dashboard will show waiting for connection on port 5555

On Raspberry Pi, run:
python3 raspberry_pi_detector.py --host <laptop_ip_address> --port 5555

The laptop will display:
- Live video feed with detection overlays
- DNN predictions with confidence scores
- SNN predictions with confidence scores
- Threat alerts for gunshot, siren, and car horn
- Real-time latency statistics
- Model agreement/disagreement indicators

Performance Metrics

DNN Model:
- Average inference latency: 15-25 ms
- Detection accuracy: 92% on UrbanSound8K
- Energy consumption: Higher than SNN

SNN Model:
- Average inference latency: 8-12 ms
- Detection accuracy: 88% on UrbanSound8K
- Energy consumption: Lower than DNN

Model Agreement Rate: Approximately 85% between DNN and SNN

Academic Context

This project was developed as part of research at the Department of Computer Science, University of South Carolina, focusing on neuromorphic computing and edge-based threat detection systems. The research compares traditional deep learning approaches with biologically-inspired spiking neural networks for real-time audio classification.

License

MIT License

Copyright (c) 2026 Sweety Ramnani and Zarin Shejuti

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files, to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software.

Contact

Sweety Ramnani: sramnani@email.sc.edu
Zarin Shejuti: zshejuti@email.sc.edu

Department of Computer Science
University of South Carolina
Columbia, SC 29208

Repository: https://github.com/Sweenderella/EdgeDefenseShield
