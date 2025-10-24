#!/usr/bin/env python3
"""
Create clean, properly formatted notebooks
"""

import json
import os

def create_clean_question_2():
    """Create a clean Question 2 notebook"""
    
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Lab 2: Data Loading and Augmentation Using Keras\n",
                    "\n",
                    "## AI Capstone Project with Deep Learning\n",
                    "\n",
                    "This lab focuses on implementing data loading and augmentation techniques using Keras for agricultural land classification.\n",
                    "\n",
                    "### Tasks:\n",
                    "1. Create the list all_image_paths containing paths of files from both folders: class_0_non_agri and class_1_agri\n",
                    "2. Create a temporary list temp by binding image paths and labels. Print 5 random samples\n",
                    "3. Generate a data batch (batch size = 8) using the custom_data_generator function\n",
                    "4. Create validation data using a batch size of 8"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": 1,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Import necessary libraries\n",
                    "import numpy as np\n",
                    "import matplotlib.pyplot as plt\n",
                    "import os\n",
                    "import glob\n",
                    "import random\n",
                    "from PIL import Image\n",
                    "from sklearn.model_selection import train_test_split\n",
                    "from sklearn.utils import shuffle\n",
                    "\n",
                    "print(\"Basic imports successful!\")\n",
                    "print(\"Note: For full TensorFlow functionality, run this in Google Colab\")"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": 2,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Create sample data for demonstration\n",
                    "def create_sample_data():\n",
                    "    # Create directories\n",
                    "    os.makedirs('./images_dataSAT/class_0_non_agri', exist_ok=True)\n",
                    "    os.makedirs('./images_dataSAT/class_1_agri', exist_ok=True)\n",
                    "    \n",
                    "    # Create non-agricultural images (class 0)\n",
                    "    for i in range(20):\n",
                    "        img = np.zeros((64, 64, 3), dtype=np.uint8)\n",
                    "        if i < 10:\n",
                    "            # Urban areas\n",
                    "            img[:, :] = [60, 60, 60]\n",
                    "            for x in range(0, 64, 16):\n",
                    "                for y in range(0, 64, 16):\n",
                    "                    if np.random.random() > 0.3:\n",
                    "                        img[y:y+12, x:x+12] = [80, 80, 80]\n",
                    "            img[30:34, :] = [40, 40, 40]\n",
                    "            img[:, 30:34] = [40, 40, 40]\n",
                    "        else:\n",
                    "            # Forest areas\n",
                    "            img[:, :] = [30, 60, 30]\n",
                    "            for x in range(0, 64, 8):\n",
                    "                for y in range(0, 64, 8):\n",
                    "                    if np.random.random() > 0.4:\n",
                    "                        img[y:y+6, x:x+6] = [20, 80, 20]\n",
                    "        \n",
                    "        noise = np.random.randint(-20, 20, (64, 64, 3))\n",
                    "        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)\n",
                    "        Image.fromarray(img).save(f'./images_dataSAT/class_0_non_agri/non_agri_{i:03d}.png')\n",
                    "    \n",
                    "    # Create agricultural images (class 1)\n",
                    "    for i in range(25):\n",
                    "        img = np.zeros((64, 64, 3), dtype=np.uint8)\n",
                    "        if i < 12:\n",
                    "            img[:, :] = [139, 69, 19]\n",
                    "            for y in range(0, 64, 4):\n",
                    "                if y % 8 < 4:\n",
                    "                    img[y:y+2, :] = [34, 139, 34]\n",
                    "        else:\n",
                    "            base_colors = [[160, 82, 45], [210, 180, 140], [222, 184, 135]]\n",
                    "            img[:, :] = base_colors[i % 3]\n",
                    "            for y in range(0, 64, 6):\n",
                    "                if y % 12 < 6:\n",
                    "                    img[y:y+3, :] = [50, 205, 50]\n",
                    "        \n",
                    "        variation = np.random.randint(-15, 15, (64, 64, 3))\n",
                    "        img = np.clip(img.astype(np.int16) + variation, 0, 255).astype(np.uint8)\n",
                    "        Image.fromarray(img).save(f'./images_dataSAT/class_1_agri/agri_{i:03d}.png')\n",
                    "    \n",
                    "    print(\"Sample data created successfully!\")\n",
                    "\n",
                    "# Create sample data\n",
                    "create_sample_data()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Task 1: Create the list all_image_paths containing paths of files from both folders"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": 3,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Task 1: Create the list all_image_paths containing paths of files from both folders\n",
                    "print(\"Task 1: Create all_image_paths list\")\n",
                    "\n",
                    "# Get paths from both folders\n",
                    "non_agri_paths = glob.glob('./images_dataSAT/class_0_non_agri/*.png')\n",
                    "agri_paths = glob.glob('./images_dataSAT/class_1_agri/*.png')\n",
                    "\n",
                    "# Combine all paths\n",
                    "all_image_paths = non_agri_paths + agri_paths\n",
                    "\n",
                    "print(f\"Non-agricultural images: {len(non_agri_paths)}\")\n",
                    "print(f\"Agricultural images: {len(agri_paths)}\")\n",
                    "print(f\"Total images: {len(all_image_paths)}\")\n",
                    "print(f\"First 3 paths: {all_image_paths[:3]}\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Task 2: Create a temporary list temp by binding image paths and labels"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": 4,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Task 2: Create a temporary list temp by binding image paths and labels. Print 5 random samples\n",
                    "print(\"Task 2: Create temp list with paths and labels\")\n",
                    "\n",
                    "# Create labels for each path\n",
                    "labels = []\n",
                    "for path in all_image_paths:\n",
                    "    if 'class_0_non_agri' in path:\n",
                    "        labels.append(0)  # Non-agricultural\n",
                    "    else:\n",
                    "        labels.append(1)  # Agricultural\n",
                    "\n",
                    "# Create temp list by binding paths and labels\n",
                    "temp = list(zip(all_image_paths, labels))\n",
                    "\n",
                    "# Print 5 random samples\n",
                    "print(\"5 random samples from temp list:\")\n",
                    "random_samples = random.sample(temp, 5)\n",
                    "for i, (path, label) in enumerate(random_samples, 1):\n",
                    "    filename = os.path.basename(path)\n",
                    "    class_name = \"Agricultural\" if label == 1 else \"Non-Agricultural\"\n",
                    "    print(f\"Sample {i}: {filename} -> Class {label} ({class_name})\")\n",
                    "\n",
                    "print(f\"Total samples in temp: {len(temp)}\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Task 3: Generate a data batch using the custom_data_generator function"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": 5,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Task 3: Generate a data batch (batch size = 8) using the custom_data_generator function\n",
                    "print(\"Task 3: Create custom data generator and generate batch\")\n",
                    "\n",
                    "def custom_data_generator(image_paths, labels, batch_size=8):\n",
                    "    \"\"\"Custom data generator for image loading\"\"\"\n",
                    "    while True:\n",
                    "        # Shuffle data\n",
                    "        shuffled_data = list(zip(image_paths, labels))\n",
                    "        random.shuffle(shuffled_data)\n",
                    "        \n",
                    "        # Generate batches\n",
                    "        for i in range(0, len(shuffled_data), batch_size):\n",
                    "            batch_paths, batch_labels = zip(*shuffled_data[i:i+batch_size])\n",
                    "            \n",
                    "            # Load and preprocess images\n",
                    "            batch_images = []\n",
                    "            for path in batch_paths:\n",
                    "                img = Image.open(path)\n",
                    "                img = img.resize((64, 64))\n",
                    "                img_array = np.array(img) / 255.0  # Normalize\n",
                    "                batch_images.append(img_array)\n",
                    "            \n",
                    "            yield np.array(batch_images), np.array(batch_labels)\n",
                    "\n",
                    "# Create generator\n",
                    "data_gen = custom_data_generator(all_image_paths, labels, batch_size=8)\n",
                    "\n",
                    "# Generate one batch\n",
                    "batch_images, batch_labels = next(data_gen)\n",
                    "\n",
                    "print(f\"Batch shape: {batch_images.shape}\")\n",
                    "print(f\"Batch labels: {batch_labels}\")\n",
                    "print(f\"Image data type: {batch_images.dtype}\")\n",
                    "print(f\"Image value range: [{batch_images.min():.3f}, {batch_images.max():.3f}]\")\n",
                    "print(\"Generated Batch (8 images)\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Task 4: Create validation data using a batch size of 8"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": 6,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Task 4: Create validation data using a batch size of 8\n",
                    "print(\"Task 4: Create validation data\")\n",
                    "\n",
                    "# Split data into train and validation\n",
                    "train_paths, val_paths, train_labels, val_labels = train_test_split(\n",
                    "    all_image_paths, labels, test_size=0.2, random_state=42, stratify=labels\n",
                    ")\n",
                    "\n",
                    "# Create validation generator\n",
                    "val_gen = custom_data_generator(val_paths, val_labels, batch_size=8)\n",
                    "\n",
                    "# Generate validation batch\n",
                    "val_batch_images, val_batch_labels = next(val_gen)\n",
                    "\n",
                    "print(f\"Training samples: {len(train_paths)}\")\n",
                    "print(f\"Validation samples: {len(val_paths)}\")\n",
                    "print(f\"Validation batch shape: {val_batch_images.shape}\")\n",
                    "print(f\"Validation batch labels: {val_batch_labels}\")\n",
                    "print(\"Validation data created successfully!\")"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.8.5"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    return notebook

def main():
    """Create clean notebooks"""
    print("Creating clean Question 2 notebook...")
    
    # Create clean Question 2
    clean_notebook = create_clean_question_2()
    
    # Save the clean notebook
    with open('notebooks/Question 2.ipynb', 'w', encoding='utf-8') as f:
        json.dump(clean_notebook, f, indent=1, ensure_ascii=False)
    
    print("Clean Question 2 notebook created successfully!")

if __name__ == "__main__":
    main()
