# One class classifier with salience

This repository implements a deep learning-based **one-class classifier** system using **DenseNet-121** as a feature extractor and a custom classifier for spoof detection. The model incorporates **heatmaps** to guide training.

---

## **Features**
- Uses **DenseNet-121** as a feature extractor
- Implements **heatmap-based attention** using CAM
- Uses a **spoof detection classifier** trained with adversarial noise
- Supports **multi-GPU training** and **efficient dataset loading**
- Implements **custom loss functions** including entropy-based attention loss

---

## **Installation**
### **1. Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/face-spoof-detection.git
cd face-spoof-detection
```

### **2. Dependancies**
pytorch, cv2, tdqm

### **3. Dataset structure and training**
The dataset should be structured as follows:

```
/project01/cvrl/datasets/[your dataset]/
│── train/
│   ├── images/
│   ├── heatmaps/
│── test/
│   ├── images/
|── val/
│   ├── images/
│── csvs/
│   ├── [csv name here].csv
```

Modify the dataset path in the script if necessary.
Or use the following run line options for `run_densenet.py`:
 - datasetPath: Path to the dataset (default: /project01/cvrl/datasets/CYBORG_centered_face/)
 - heatmaps: Path to heatmaps (default: /project01/cvrl/datasets/CYBORG_centered_face/train/heatmaps/)
 - csvPath: Path to CSV file (default: /project01/cvrl/datasets/CYBORG_centered_face/csvs/original.csv)
 - network: Backbone network (densenet by default, which is the only one supported)
 - nClasses: Number of output classes (default: 2, also the only one supported)
 - outputPath: Directory to save models

### **4. Testing**

```
python test_droid.py 
```

| Argument         | Description                                       | Default Value  |
|-----------------|---------------------------------------------------|---------------|
| `-imageFolder`  | Path to the folder containing images for processing. | `/project01/cvrl/datasets/CYBORG_centered_face/` |
| `-modelPath`    | Path to the pre-trained model file.               | `./models/current_model.pth` |
| `-csv`          | Path to the CSV file containing dataset metadata.  | `/project01/cvrl/datasets/CYBORG_centered_face/csvs/original.csv` |
| `-output_dir`   | Directory to save output files.                   | `./models/` |
| `-output_filename` | Name of the output CSV file to store results.   | `original_densenet_attempt1_centered_1.csv` |
| `-fakeToken`    | Label used to represent spoofed images.           | `"Spoof"` |
| `-nClasses`     | Number of classes for classification (e.g., real vs. spoof). | `2` |
| `-imageSize`    | Input image size (assumed square).                | `224` |
