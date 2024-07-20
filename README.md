# Mask Detection and Social Distancing Project

## Project Structure

- `dataset/`: Contains the `with_mask` and `without_mask` images.
- `models/`: Contains the trained models.
- `src/`: Contains the source code.
  - `dataset_preparation.py`: Script for preparing the dataset (if needed).
  - `model_training.py`: Script for training the mask detection model.
  - `mask_detection.py`: Module for mask detection.
  - `social_distancing.py`: Module for social distancing detection.
  - `utils.py`: Utility functions for logging and other common tasks.
  - `main.py`: Main script to run the application.
- `logs/`: Contains log files.
- `README.md`: Project documentation.
- `requirements.txt`: Required Python packages.

## How to Use

1. **Setup Environment**
   ```bash
   pip install -r requirements.txt
