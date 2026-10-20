# Plant Disease Detection with MobileNetV2

A TensorFlow/Keras image classifier that uses transfer learning with MobileNetV2 to classify plant leaf diseases.

## Run the notebook in Google Colab

1. Open `plant_disease_mobilenetv2.ipynb` in Google Colab (File → Upload notebook, or open it from your GitHub repository).
2. Select **Runtime → Change runtime type → GPU**. GPU availability depends on your Colab account and current capacity.
3. In the notebook's data cell, make sure the Google Drive dataset file is shared as **Anyone with the link → Viewer**.
4. Run the cells from top to bottom. The notebook downloads and extracts the dataset in the Colab runtime.
5. The trained model and metadata are copied to `My Drive/plant_disease_runs/mobilenetv2` at the end.

Colab storage is temporary. The notebook downloads the ZIP again after a runtime reset. Free GPU access and runtime limits vary, so a run may need to be restarted.

## Dataset layout

The ZIP should contain these directories, possibly inside a parent folder:

```text
dataset/
├── train/
│   ├── Apple___Apple_scab/
│   └── ...
├── val/
│   ├── Apple___Apple_scab/
│   └── ...
└── test/
    ├── Apple___Apple_scab/
    └── ...
```

The dataset is not included in this repository because it is large. The notebook contains the Drive file ID used to download it; update that value in the data cell if you use a different copy. The linked file must be accessible to Colab.

## Training setup

- MobileNetV2 ImageNet weights, with a new softmax classification head.
- Two training stages: train the head with the backbone frozen, then fine-tune the top backbone layers at a lower learning rate.
- To shorten training, each epoch uses up to 200 training images and 100 validation images per class. Change `MAX_TRAIN_PER_CLASS` and `MAX_VAL_PER_CLASS` in the data-loading cell to adjust these limits; set a limit to `0` to use the full split.
- Final test evaluation uses the full test split.

The notebook saves `model.keras`, `class_names.json`, and `metrics.json`. Keep the class names file with the model so predicted class indices can be mapped to disease labels.

## Dependencies

Colab includes TensorFlow and common data science packages. For a local Jupyter environment, install the packages in `requirements.txt`. TensorFlow GPU training on Windows requires WSL2; native Windows TensorFlow 2.11 and later does not support NVIDIA GPU training.
