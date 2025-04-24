# Reinforcement Learning for Interactive Polyp Segmentation in Colonoscopy Images

This repository contains the implementation of a reinforcement learning (RL) approach for interactive polyp segmentation in colonoscopy images, developed as part of the BMI/CS 567: Medical Image Analysis course at the University of Wisconsin-Madison.

## Project Overview

Colorectal cancer is one of the leading causes of cancer-related deaths worldwide, but early detection through colonoscopy significantly improves survival rates. This project explores two approaches to polyp segmentation:

1. **Traditional U-Net Model**: A fully automated one-shot segmentation approach
2. **Interactive RL Agent**: A novel reinforcement learning approach that simulates interactive segmentation through sequential decision-making

The interactive RL approach mimics how human experts might annotate medical images through an iterative refinement process, providing both high accuracy and interpretability.

## Key Findings

Our experiments revealed some interesting patterns:

- U-Net provides efficient segmentation with excellent test set accuracy (Dice: 0.9319, IoU: 0.8734)
- The RL approach offers remarkable performance on validation data (Dice: 0.9877, IoU: 0.9758) 
- The two models show complementary strengths, with U-Net excelling on test data while InteractiveRL shows superior generalization to validation data
- The step-by-step nature of the RL method provides a transparent and interpretable segmentation process

![Performance Comparison](academic_figures/comprehensive_performance.png)

## Dataset

The project uses the CVC-ClinicDB dataset, which includes:

- 612 colonoscopy images acquired from 31 different colonoscopy sequences
- Each image is a 384×288 pixel RGB frame in TIFF format
- Expert-annotated binary segmentation masks indicating polyp regions

The dataset was split using a fixed random seed (42) for reproducibility:
- 70% (428 images) for training
- 15% (92 images) for validation
- 15% (92 images) for testing

## Methodology

### U-Net Implementation
- 4 downsampling blocks in the encoder and 4 upsampling blocks in the decoder
- Trained using a combination of Binary Cross-Entropy and Dice loss functions
- AdamW optimizer with learning rate of 5e-4 and weight decay of 1e-4
- Early stopping triggered at epoch 96 of a maximum 1000 epochs

### InteractiveRL Implementation
- State: Comprises the colonoscopy image, current segmentation mask, pointer location, distance transform, and edge maps
- Actions: Move pointer (up, down, left, right), expand region, shrink region, or confirm segmentation
- Reward: Improvement in Dice coefficient with additional bonuses for high scores and penalties for premature termination
- Architecture: Policy network and value network with convolutional layers and attention mechanism
- Training: 1000 episodes with PPO-style updates, discount factor of 0.99, and entropy coefficient of 0.01

![Interactive RL Process](academic_figures/interactive_rl_process.png)

## Project Structure

```
.
├── code/
│   ├── data_utils.py        # Data loading and preprocessing
│   ├── unet_model.py        # U-Net model implementation
│   ├── simple_rl.py         # Interactive RL implementation
│   └── utils.py             # Utility functions
├── academic_figures/        # Generated visualizations for analysis
├── data/
│   └── raw/                 # Raw dataset files
├── models/                  # Saved models
├── results/                 # Evaluation results
├── generate_academic_plots.py # Script for creating publication-ready figures
├── main.py                  # Main script to run experiments
└── requirements.txt         # Project dependencies
```

## Usage

### Installation
```bash
# Create and activate conda environment
conda create -n polyprl python=3.8 -y
conda activate polyprl

# Install dependencies
pip install -r requirements.txt
```

### Training Models
```bash
# Train U-Net model
python main.py --mode train_unet --unet_epochs 1000 --batch_size 4 --lr 5e-4

# Train InteractiveRL model
python main.py --mode train_simple_rl --simple_rl_episodes 1000 --max_steps 10 --lr 5e-4 --eval_interval 5 --num_eval_episodes 3

# Run both models and compare
python main.py --mode train_and_evaluate
```

### Generating Plots
```bash
# Generate academic-quality plots comparing both approaches
python generate_academic_plots.py
```

## Results

The evaluation compares both approaches using:
- Dice Similarity Coefficient (DSC)
- Intersection over Union (IoU)
- Sample-level performance analysis
- Generalization capability across datasets

Key visualizations include:
- Performance distribution across samples
- Generalization gap analysis
- Sample-wise error analysis
- Step-by-step segmentation progression

![Error Analysis](academic_figures/error_analysis.png)

## Conclusion

This study demonstrates that both U-Net and reinforcement learning offer viable approaches for medical image segmentation, with different strengths depending on the context. The most striking finding is the different generalization patterns exhibited by the two models, suggesting they capture different aspects of the underlying data distribution.

The most promising extension would be a hybrid system that initializes segmentation using U-Net predictions and then refines them using InteractiveRL, potentially combining the efficiency of U-Net with the adaptability and interpretability of reinforcement learning.

## Acknowledgments

- BMI/CS 567: Medical Image Analysis course at the University of Wisconsin-Madison
- CVC-ClinicDB dataset providers
- PyTorch and related libraries 
## Data Availability

The CVC-ClinicDB dataset is not included in this repository due to size constraints. To use this code:

1. Download the dataset from [https://polyp.grand-challenge.org/CVCClinicDB/](https://polyp.grand-challenge.org/CVCClinicDB/)
2. Place the images in `data/raw/Original` and masks in `data/raw/Ground Truth`
