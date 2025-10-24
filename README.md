# AI Capstone Project with Deep Learning

## IBM Coursera Course - Agricultural Land Classification

This project implements a comprehensive deep learning solution for agricultural land classification using satellite imagery. The project covers data loading, augmentation, CNN classifiers, and Vision Transformers in both Keras and PyTorch frameworks.

## Project Overview

The goal of this project is to classify satellite images into two categories:
- **Class 0**: Non-Agricultural Land
- **Class 1**: Agricultural Land

## Project Structure

```
AI Capstone Project with Deep Learning/
├── images_dataSAT/
│   ├── class_0_non_agri/          # Non-agricultural land images
│   └── class_1_agri/              # Agricultural land images
├── notebooks/
│   ├── lab1_memory_generator_data_loading.ipynb
│   ├── lab2_keras_data_augmentation.ipynb
│   ├── lab3_pytorch_data_augmentation.ipynb
│   ├── lab4_keras_cnn_classifier.ipynb
│   ├── lab5_pytorch_cnn_classifier.ipynb
│   ├── lab6_comparative_analysis.ipynb
│   ├── lab7_keras_vit.ipynb
│   ├── lab8_pytorch_vit.ipynb
│   └── lab9_integration_evaluation.ipynb
├── models/                        # Saved model files
├── data/                          # Additional data files
├── results/                       # Results and visualizations
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## Labs Overview

### Lab 1: Compare Memory-Based versus Generator-Based Data Loading (10 points)
- **Tasks**: Image shape analysis, data loading comparison
- **Key Concepts**: Memory efficiency vs access speed
- **Techniques**: Custom data generators, batch processing

### Lab 2: Data Loading and Augmentation Using Keras (8 points)
- **Tasks**: Image path management, data augmentation
- **Key Concepts**: ImageDataGenerator, train/validation split
- **Techniques**: Custom data generators, augmentation pipelines

### Lab 3: Data Loading and Augmentation Using PyTorch (10 points)
- **Tasks**: Transform pipelines, DataLoader creation
- **Key Concepts**: torchvision.transforms, ImageFolder
- **Techniques**: Custom transforms, batch processing

### Lab 4: Train and Evaluate a Keras-Based Classifier (12 points)
- **Tasks**: CNN architecture, training loop, evaluation
- **Key Concepts**: Sequential models, callbacks, metrics
- **Techniques**: 4 Conv2D + 5 Dense layers, checkpointing

### Lab 5: Implement and Test a PyTorch-Based Classifier (20 points)
- **Tasks**: Model definition, training loop, evaluation
- **Key Concepts**: nn.Module, optimizers, loss functions
- **Techniques**: Custom CNN class, tqdm progress bars

### Lab 6: Comparative Analysis of Keras and PyTorch Models (10 points)
- **Tasks**: Model comparison, metrics analysis
- **Key Concepts**: F1-score, confusion matrices, evaluation
- **Techniques**: Performance comparison, visualization

### Lab 7: Vision Transformers in Keras (10 points)
- **Tasks**: Pre-trained model loading, hybrid architecture
- **Key Concepts**: Transfer learning, feature extraction
- **Techniques**: CNN-ViT integration, model compilation

### Lab 8: Vision Transformers in PyTorch (12 points)
- **Tasks**: ViT implementation, training configuration
- **Key Concepts**: Transformer architecture, attention mechanisms
- **Techniques**: Custom ViT model, training optimization

### Lab 9: Land Classification: CNN-Transformer Integration Evaluation (8 points)
- **Tasks**: Model evaluation, performance comparison
- **Key Concepts**: Hybrid models, ensemble methods
- **Techniques**: Model integration, comprehensive evaluation

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd "AI Capstone Project with Deep Learning"
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation**:
   ```bash
   python -c "import tensorflow as tf; import torch; print('Installation successful!')"
   ```

## Usage

### Running Individual Labs

1. **Start Jupyter Lab**:
   ```bash
   jupyter lab
   ```

2. **Navigate to notebooks directory**:
   ```bash
   cd notebooks
   ```

3. **Run labs in order**:
   - Lab 1: Data loading fundamentals
   - Lab 2-3: Framework-specific data handling
   - Lab 4-5: CNN implementation and training
   - Lab 6: Model comparison and analysis
   - Lab 7-9: Vision Transformers and integration

### Key Features

- **Dual Framework Support**: Both Keras and PyTorch implementations
- **Comprehensive Data Pipeline**: Loading, augmentation, and preprocessing
- **Advanced Architectures**: CNN and Vision Transformer models
- **Performance Analysis**: Detailed metrics and visualizations
- **Reproducible Results**: Fixed random seeds and consistent evaluation

## Model Architectures

### CNN Model (Keras & PyTorch)
- **Input**: 64x64x3 RGB images
- **Convolutional Layers**: 4 layers with increasing filters (32, 64, 128, 128)
- **Dense Layers**: 5 layers (512, 256, 128, 64, 1/2)
- **Regularization**: Dropout layers to prevent overfitting
- **Activation**: ReLU for hidden layers, Sigmoid/Softmax for output

### Vision Transformer (ViT)
- **Input**: 64x64x3 RGB images
- **Patch Size**: 16x16 patches
- **Embedding Dimension**: 768
- **Attention Heads**: 12
- **Transformer Blocks**: 12 layers
- **Integration**: CNN feature extraction + ViT processing

## Data Augmentation

### Keras Augmentation
- Random rotation (20°)
- Width/height shift (0.1)
- Shear transformation (0.2)
- Zoom (0.2)
- Horizontal flip
- Color jitter

### PyTorch Augmentation
- Random horizontal flip (p=0.5)
- Random vertical flip (p=0.2)
- Random rotation (45°)
- Color jitter
- Random affine transformation

## Evaluation Metrics

- **Accuracy**: Overall classification accuracy
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed error analysis

## Results

The project demonstrates:
- **Data Loading**: Efficient memory-based vs generator-based approaches
- **Model Performance**: Comparable results between Keras and PyTorch
- **Architecture Comparison**: CNN vs Vision Transformer performance
- **Framework Analysis**: Strengths and weaknesses of each framework

## Technical Requirements

- **Python**: 3.8+
- **TensorFlow**: 2.13.0+
- **PyTorch**: 2.0.0+
- **Memory**: 8GB+ RAM recommended
- **GPU**: CUDA-compatible GPU (optional but recommended)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is part of the IBM Coursera AI Capstone Project with Deep Learning course.

## Acknowledgments

- IBM for providing the course materials
- Coursera for the learning platform
- The deep learning community for open-source tools and resources

## Contact

For questions or support, please refer to the course discussion forums or create an issue in the repository.

---

**Note**: This project is for educational purposes as part of the IBM Coursera AI Capstone Project with Deep Learning course. The implementations demonstrate best practices in deep learning for computer vision tasks.






