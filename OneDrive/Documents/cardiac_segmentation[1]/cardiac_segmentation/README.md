# 🫀 Cardiac MRI Segmentation System

A complete end-to-end system for cardiac MRI segmentation using U-Net architecture, featuring real-time web interface, comprehensive evaluation tools, and professional-grade training pipeline.

![Demo Banner](https://img.shields.io/badge/Demo-Live-brightgreen) 
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red) 
![Python](https://img.shields.io/badge/Python-3.8+-blue)
![License](https://img.shields.io/badge/License-MIT-green)

## 🎯 Demo Overview

This demonstration showcases an end-to-end cardiac MRI segmentation system that achieves:
- **Performance Target**: 75-80% Dice coefficient
- **Processing Speed**: 3-5 seconds per image
- **Memory Usage**: <6GB GPU memory during inference
- **Classes**: 4-class segmentation (Background, RV, Myocardium, LV)

### 🏆 Key Features

- **Complete U-Net Implementation** with skip connections and modern training techniques
- **Interactive Web Interface** built with Streamlit for real-time demonstration
- **Comprehensive Evaluation Suite** with detailed metrics and visualizations
- **Professional Training Pipeline** with mixed precision, early stopping, and tensorboard logging
- **ACDC Dataset Support** with automatic preprocessing and augmentation
- **Production-Ready Code** with proper error handling, logging, and configuration management

## 📁 Project Structure

```
cardiac_segmentation/
├── 📊 data/
│   └── acdc_dataset.py          # ACDC dataset handler with NIfTI support
├── 🧠 models/
│   └── unet.py                  # U-Net architecture implementation
├── 🏋️ training/
│   ├── losses.py                # Loss functions (Dice, Focal, Combined)
│   ├── metrics.py               # Evaluation metrics and confusion matrix
│   └── trainer.py               # Complete training pipeline
├── 📈 evaluation/
│   └── evaluator.py             # Model evaluation and analysis
├── 🎨 visualization/
│   └── visualizer.py            # Comprehensive visualization tools
├── 🌐 app.py                    # Streamlit web application
├── ⚙️ config.py                 # Configuration management
├── 🚀 main.py                   # Training script
├── 📋 evaluate.py               # Evaluation script
├── 📦 requirements.txt          # Dependencies
├── 🛠️ setup.py                  # Package installation
└── 📖 README.md                 # This file
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/your-repo/cardiac-segmentation.git
cd cardiac-segmentation

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Or install as package
pip install -e .
```

### 2. Download ACDC Dataset

Download the ACDC dataset from the official challenge website:
- Website: https://www.creatis.insa-lyon.fr/Challenge/acdc/
- Place the training data in: `data/ACDC/training/`

Expected structure:
```
data/ACDC/training/
├── patient001/
│   ├── patient001_frame01.nii.gz    # ED image
│   ├── patient001_frame01_gt.nii.gz # ED ground truth
│   ├── patient001_frame12.nii.gz    # ES image
│   └── patient001_frame12_gt.nii.gz # ES ground truth
├── patient002/
└── ...
```

### 3. Run the Web Demo

```bash
# Start the Streamlit web application
streamlit run app.py

# Open browser to: http://localhost:8501
```

### 4. Train Your Own Model

```bash
# Train with default settings
python main.py --data-path data/ACDC/training

# Train with custom parameters
python main.py \
    --data-path data/ACDC/training \
    --batch-size 16 \
    --epochs 150 \
    --learning-rate 1e-3 \
    --model-size standard
```

### 5. Evaluate Trained Model

```bash
# Evaluate model performance
python evaluate.py \
    --model-path checkpoints/best_model.pth \
    --data-path data/ACDC/training \
    --visualize-samples 10
```

## 🎮 Web Interface Guide

The Streamlit web interface provides an intuitive way to interact with the segmentation system:

### Features:
1. **📁 File Upload**: Support for NIfTI files (.nii, .nii.gz)
2. **⚡ Real-time Processing**: Instant segmentation with timing metrics
3. **🎨 Interactive Visualization**: Adjustable overlay opacity and class toggles
4. **📊 Detailed Metrics**: Class-wise statistics and performance indicators
5. **💾 Export Options**: Download segmentation masks and metrics

### Usage:
1. Upload a cardiac MRI NIfTI file
2. Click "Process Image" to run segmentation
3. View results with interactive visualizations
4. Adjust visualization settings in the sidebar
5. Download results for further analysis

## 🏋️ Training Configuration

### Default Training Settings:
- **Architecture**: Standard U-Net (64 base channels)
- **Loss Function**: Combined (0.3 × CrossEntropy + 0.7 × Dice)
- **Optimizer**: Adam with learning rate 1e-3
- **Batch Size**: 8 (adjustable based on GPU memory)
- **Max Epochs**: 100 (with early stopping)
- **Augmentation**: Horizontal flip, rotation (±10°)

### Advanced Features:
- **Mixed Precision Training** for faster training and reduced memory usage
- **Learning Rate Scheduling** with ReduceLROnPlateau
- **Early Stopping** to prevent overfitting
- **Gradient Clipping** for stable training
- **Tensorboard Logging** for real-time monitoring

## 📊 Model Performance

### Target Performance Metrics:
| Metric | Target | Description |
|--------|--------|-------------|
| Overall Dice | 75-80% | Average across all classes |
| RV Dice | 70-85% | Right ventricle segmentation |
| Myocardium Dice | 75-85% | Heart muscle segmentation |
| LV Dice | 85-95% | Left ventricle segmentation |
| Processing Time | 3-5s | Per image inference time |

### Model Variants:
- **Small**: 32 base channels (~8M parameters)
- **Standard**: 64 base channels (~31M parameters) - **Recommended**
- **Large**: 96 base channels (~69M parameters)

## 🛠️ Development Guide

### Code Structure:
- **Modular Design**: Each component is self-contained and testable
- **Configuration Management**: Centralized config system
- **Error Handling**: Comprehensive error handling and logging
- **Type Hints**: Full type annotation for better code quality

### Adding New Features:
1. **New Loss Functions**: Add to `training/losses.py`
2. **New Metrics**: Extend `training/metrics.py`
3. **New Visualizations**: Add to `visualization/visualizer.py`
4. **New Models**: Create in `models/` directory

### Testing:
```bash
# Run tests
pytest tests/

# Run with coverage
pytest --cov=. tests/
```

## 🔧 Configuration Options

The system uses a centralized configuration system in `config.py`:

### Key Configuration Sections:
- **ACDC_CONFIG**: Dataset-specific settings
- **MODEL_CONFIG**: Model architecture parameters
- **TRAINING_CONFIG**: Training hyperparameters
- **VISUALIZATION_CONFIG**: Plotting and display settings
- **WEBAPP_CONFIG**: Web interface configuration

### Environment Variables:
```bash
export BATCH_SIZE=16
export LEARNING_RATE=1e-3
export ACDC_DATA_PATH=/path/to/data
export MODEL_PATH=/path/to/model.pth
```

## 📈 Evaluation Metrics

The system provides comprehensive evaluation metrics:

### Segmentation Metrics:
- **Dice Coefficient**: Overlap between prediction and ground truth
- **IoU (Jaccard Index)**: Intersection over Union
- **Pixel Accuracy**: Correct pixel classification rate
- **Class-wise Statistics**: Per-class performance breakdown

### Analysis Tools:
- **Confusion Matrix**: Class-wise prediction accuracy
- **Error Analysis**: Visualization of prediction errors
- **Statistical Analysis**: Mean, std, confidence intervals
- **Worst/Best Case Analysis**: Identification of challenging cases

## 🚀 Deployment Options

### Local Deployment:
```bash
# Run web app locally
streamlit run app.py --server.port 8501
```

### Docker Deployment:
```bash
# Build Docker image
docker build -t cardiac-segmentation .

# Run container
docker run -p 8501:8501 cardiac-segmentation
```

### Cloud Deployment:
- **Streamlit Cloud**: Direct GitHub integration
- **Heroku**: Web app deployment
- **AWS/GCP/Azure**: Scalable cloud deployment

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and add tests
4. Ensure code quality: `black . && flake8 .`
5. Submit a pull request

### Development Setup:
```bash
# Install development dependencies
pip install -e ".[dev]"

# Run code formatting
black .

# Run linting
flake8 .

# Run type checking
mypy .
```

## 📚 References and Citations

### Academic References:
1. **U-Net Paper**: Ronneberger, O., Fischer, P., & Brox, T. (2015). U-net: Convolutional networks for biomedical image segmentation.
2. **ACDC Challenge**: Bernard, O., et al. (2018). Deep learning techniques for automatic MRI cardiac multi-structures segmentation.

### Dataset:
```bibtex
@article{bernard2018deep,
  title={Deep learning techniques for automatic MRI cardiac multi-structures segmentation and diagnosis: is the problem solved?},
  author={Bernard, Olivier and Lalande, Alain and Zotti, Clement and others},
  journal={IEEE transactions on medical imaging},
  volume={37},
  number={11},
  pages={2514--2525},
  year={2018}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

### Getting Help:
- **Issues**: Report bugs and request features on GitHub Issues
- **Documentation**: Comprehensive docs in the `docs/` directory
- **Discussions**: Community discussions on GitHub Discussions

### Common Issues:

#### CUDA Out of Memory:
```bash
# Reduce batch size
python main.py --batch-size 4

# Use smaller model
python main.py --model-size small
```

#### Missing Dependencies:
```bash
# Reinstall requirements
pip install -r requirements.txt --force-reinstall
```

#### Dataset Path Issues:
```bash
# Check dataset structure
ls data/ACDC/training/
# Should show patient001/, patient002/, etc.
```

## 🔮 Future Enhancements

### Planned Features:
- [ ] **Attention Mechanisms**: Add attention layers for explainable AI
- [ ] **3D Volume Processing**: Full 3D cardiac volume segmentation
- [ ] **Multi-timepoint Analysis**: Temporal analysis across cardiac cycle
- [ ] **Advanced Augmentation**: More sophisticated data augmentation techniques
- [ ] **Model Ensemble**: Multiple model ensemble for improved accuracy
- [ ] **Clinical Validation**: Integration with clinical workflow tools

### Research Directions:
- [ ] **Uncertainty Quantification**: Bayesian neural networks for uncertainty estimation
- [ ] **Domain Adaptation**: Cross-dataset generalization
- [ ] **Active Learning**: Intelligent sample selection for annotation
- [ ] **Federated Learning**: Multi-institutional collaborative training

---

## 📞 Contact

**Project Maintainers:**
- Email: contact@example.com
- GitHub: [@example](https://github.com/example)

**Acknowledgments:**
- ACDC Challenge organizers for the dataset
- PyTorch team for the deep learning framework
- Streamlit team for the web framework
- Open source community for inspiration and tools

---

*Built with ❤️ for advancing cardiac medical imaging research*