# 🚀 Quick Start Guide - Cardiac MRI Segmentation

Get up and running with the cardiac segmentation demo in minutes!

## 📋 Prerequisites

- Python 3.8+ installed
- At least 8GB RAM (16GB recommended)
- GPU with 6GB+ VRAM (optional but recommended)
- ACDC dataset (for training) or sample NIfTI files (for demo)

## ⚡ 5-Minute Demo Setup

### Step 1: Installation (2 minutes)

```bash
# Clone and enter directory
git clone https://github.com/your-repo/cardiac-segmentation.git
cd cardiac-segmentation

# Quick install
pip install -r requirements.txt
```

### Step 2: Launch Web Demo (30 seconds)

```bash
# Start the web interface
streamlit run app.py
```

🌐 **Open browser to**: http://localhost:8501

### Step 3: Try the Demo (2 minutes)

1. **Upload a NIfTI file** (.nii or .nii.gz)
2. **Click "Process Image"** 
3. **View segmentation results** with interactive controls
4. **Download results** if needed

That's it! Your demo is running! 🎉

## 📁 Sample Data

### Option 1: Use ACDC Dataset
```bash
# Download from: https://www.creatis.insa-lyon.fr/Challenge/acdc/
# Extract to: data/ACDC/training/
```

### Option 2: Create Test Data
```python
# Generate synthetic test data
python -c "
import numpy as np
import nibabel as nib

# Create synthetic cardiac MRI
image = np.random.rand(256, 256, 1) * 1000
mask = np.random.randint(0, 4, (256, 256, 1))

# Save as NIfTI
nib.save(nib.Nifti1Image(image, np.eye(4)), 'test_image.nii.gz')
nib.save(nib.Nifti1Image(mask, np.eye(4)), 'test_mask.nii.gz')
print('Test files created: test_image.nii.gz, test_mask.nii.gz')
"
```

## 🎯 Demo Features Walkthrough

### 1. File Upload
- Supports: `.nii`, `.nii.gz` files
- Automatic preprocessing to 256×256
- File validation and error handling

### 2. Processing
- Real-time segmentation (3-5 seconds)
- Progress indicators
- Performance metrics display

### 3. Results Visualization
- **Original Image**: Grayscale cardiac MRI
- **Segmentation Mask**: Color-coded classes
- **Interactive Overlay**: Adjustable transparency
- **Class Probabilities**: Per-pixel confidence maps

### 4. Metrics & Analysis
- Class distribution percentages
- Processing time statistics
- Downloadable results

## 🏋️ Training Your Own Model

### Quick Training (15 minutes setup + training time)

```bash
# Prepare ACDC data structure
mkdir -p data/ACDC/training
# Copy patient folders here: patient001/, patient002/, etc.

# Start training with defaults
python main.py --data-path data/ACDC/training

# Monitor with tensorboard
tensorboard --logdir logs/
```

### Training Parameters
```bash
# Custom training
python main.py \
    --data-path data/ACDC/training \
    --batch-size 8 \
    --epochs 50 \
    --learning-rate 1e-3 \
    --model-size standard \
    --experiment-name my_cardiac_model
```

## 📊 Evaluation & Analysis

### Evaluate Trained Model
```bash
# Run comprehensive evaluation
python evaluate.py \
    --model-path checkpoints/best_model.pth \
    --data-path data/ACDC/training \
    --visualize-samples 5
```

### Results Generated:
- **Metrics Report**: Detailed performance statistics
- **Visualizations**: Comparison images and error analysis
- **Confusion Matrix**: Class prediction accuracy
- **Statistical Analysis**: Performance distribution

## 🔧 Common Configuration

### Memory-Limited Setup (4GB GPU)
```python
# config.py modifications
TRAINING_CONFIG = {
    "batch_size": 4,          # Reduce batch size
    "mixed_precision": True,  # Enable mixed precision
    "gradient_clip_max_norm": 0.5,
    "num_workers": 2,
}

MODEL_CONFIG = {
    "base_channels": 32,      # Use small model
}
```

### High-Performance Setup (16GB+ GPU)
```python
TRAINING_CONFIG = {
    "batch_size": 16,         # Larger batches
    "num_workers": 8,
    "mixed_precision": True,
}

MODEL_CONFIG = {
    "base_channels": 96,      # Use large model
}
```

## 🐛 Troubleshooting

### Issue: "CUDA out of memory"
**Solution:**
```bash
# Option 1: Reduce batch size
python main.py --batch-size 4

# Option 2: Use CPU only
export CUDA_VISIBLE_DEVICES=""
python main.py

# Option 3: Use smaller model
python main.py --model-size small
```

### Issue: "Module not found"
**Solution:**
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall

# Or install as package
pip install -e .
```

### Issue: "No ACDC data found"
**Solution:**
```bash
# Check directory structure
ls data/ACDC/training/
# Should show: patient001/, patient002/, etc.

# Create symbolic link if data is elsewhere
ln -s /path/to/actual/ACDC/data data/ACDC/training
```

### Issue: Web app won't start
**Solution:**
```bash
# Check port availability
netstat -tulpn | grep :8501

# Use different port
streamlit run app.py --server.port 8502

# Check Streamlit installation
pip install streamlit --upgrade
```

## 📱 Demo Scenarios

### Scenario 1: Research Demo
```bash
# Professional demo with multiple models
python evaluate.py \
    --model-path model1.pth \
    --compare-models model2.pth model3.pth \
    --create-report \
    --visualize-samples 10
```

### Scenario 2: Clinical Validation
```bash
# Evaluate on specific patient subset
python evaluate.py \
    --model-path clinical_model.pth \
    --split val \
    --save-predictions \
    --create-report
```

### Scenario 3: Performance Benchmarking
```bash
# Time and memory benchmarking
python -c "
import time
import torch
from models.unet import create_unet_model

model = create_unet_model('standard')
dummy_input = torch.randn(1, 1, 256, 256)

# Warmup
for _ in range(10):
    _ = model(dummy_input)

# Benchmark
times = []
for _ in range(100):
    start = time.time()
    _ = model(dummy_input)
    times.append(time.time() - start)

print(f'Average inference time: {np.mean(times)*1000:.2f}ms')
print(f'Std: {np.std(times)*1000:.2f}ms')
"
```

## 🎯 Next Steps

### For Researchers:
1. **Modify Architecture**: Edit `models/unet.py` for custom architectures
2. **Add Loss Functions**: Extend `training/losses.py`
3. **Custom Metrics**: Add to `training/metrics.py`
4. **Advanced Training**: Modify `training/trainer.py`

### For Developers:
1. **API Integration**: Create REST API using FastAPI
2. **Database Integration**: Add result storage
3. **User Management**: Add authentication
4. **Scaling**: Deploy with Docker/Kubernetes

### For Clinicians:
1. **Validation Studies**: Use evaluation tools
2. **Custom Datasets**: Adapt data loading
3. **Integration**: Connect to PACS systems
4. **Reporting**: Customize output formats

## 📚 Learning Resources

### Understanding the Code:
- **U-Net Architecture**: `models/unet.py` - Well-documented implementation
- **Training Loop**: `training/trainer.py` - Professional training pipeline
- **Data Handling**: `data/acdc_dataset.py` - Medical image preprocessing

### Key Concepts:
- **Medical Image Segmentation**: Pixel-wise classification
- **Dice Coefficient**: Overlap metric for segmentation
- **Data Augmentation**: Improving model generalization
- **Mixed Precision**: Memory-efficient training

## 🤝 Community & Support

### Getting Help:
- **GitHub Issues**: Bug reports and feature requests
- **Discussions**: Community Q&A
- **Documentation**: Detailed API docs in `docs/`

### Contributing:
- **Code**: Submit pull requests
- **Data**: Share preprocessed datasets
- **Models**: Contribute trained models
- **Documentation**: Improve guides and tutorials

---

## ✅ Checklist: Is Your Demo Ready?

- [ ] Python 3.8+ installed
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Web app starts successfully (`streamlit run app.py`)
- [ ] Sample NIfTI file ready for upload
- [ ] Browser opens to localhost:8501
- [ ] Image uploads and processes without errors
- [ ] Results display correctly
- [ ] Export functionality works

**All checked?** You're ready to demonstrate! 🚀

---

*Need help? Check the full README.md or open an issue on GitHub!*