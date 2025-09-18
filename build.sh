#!/bin/bash
pip install dlib-bin==19.24.2
pip install --no-deps git+https://github.com/razer-blitz/face_recognition.git face-recognition-models>=0.3.0
pip install --no-cache-dir -r requirements.txt
