# WGAN CIFAR Image Generator

This project implements a Wasserstein Generative Adversarial Network (WGAN) trained on the CIFAR-10 dataset.

## Features
- Implementation of WGAN based on the research paper
- Training performed on CIFAR-10 dataset
- Generator model saved and deployed locally
- Simple frontend using Gradio to generate images

## Dataset
CIFAR-10 dataset (60,000 images of 10 classes)

## Project Structure

WGAN_CIFAR_Project
│
├── models
│   └── generator.pth
├── model.py
├── app.py
├── requirements.txt
└── README.md

## Run Instructions

1. Create virtual environment

python -m venv venv

2. Activate environment

Windows:
venv\Scripts\activate

3. Install dependencies

pip install -r requirements.txt

4. Run the application

python app.py

The web interface will open at:

http://127.0.0.1:7860

# Reya_ATML_WGAN_CIFAR_Lab5
A simple web interface was built using Gradio to generate synthetic CIFAR images using a trained Wasserstein GAN (WGAN) model. The generator takes random noise as input and produces new images that resemble the CIFAR dataset. For better visualization, the generated 32×32 images are upscaled before displaying them in the interface.

