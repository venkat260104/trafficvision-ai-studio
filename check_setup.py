import sys

print("Python version:", sys.version)

try:
    import tensorflow as tf
    print("TensorFlow version:", tf.__version__)
    print("Traffic Sign Recognition setup successful!")
except Exception as e:
    print("TensorFlow error:", e)
    print("Failed to set up Traffic Sign Recognition.")