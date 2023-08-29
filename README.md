## TensorFlow Model Training Logs Visualization

This repository contains scripts and tools to execute and visualize TensorFlow model training logs.

### Overview

We're utilizing TensorFlow, an open-source machine learning framework, to train models on specific datasets. This repository helps in monitoring the progress and performance of these models through generated logs.

### Prerequisites

- TensorFlow (Version compatible with the provided scripts, recommended v2.x or later)

### Getting Started

1. **Clone the Repository**

   ```
   git clone https://github.com/marcoacnunes/regression-model-keras
   cd regression-model-keras
   ```

2. **Set up the Environment**

   If you're not already set up with TensorFlow, you can do so by:
   
   ```
   pip install tensorflow
   ```

3. **Run the Script**

   Execute the main script to start the training and generate logs.

   ```
   python main.py
   ```

4. **Visualizing the Logs**

   Upon running the script, you'll see logs similar to:
   
   ```
   10/10 [==============================] - 0s 776us/step
   ...
   ```

   These logs indicate the progress of the model training for each step or epoch. The value `776us` is the duration it took for that particular step.

### Features

- **CPU Feature Guard**: The script notifies if the TensorFlow binary is optimized to use available CPU instructions.
- **Step-by-Step Logging**: Detailed logging of each training step with execution time.
