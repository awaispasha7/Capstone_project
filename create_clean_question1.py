#!/usr/bin/env python3
"""
Create clean, properly formatted Question 1 notebook
"""

import json
import os

def create_clean_question_1():
    """Create a clean Question 1 notebook"""
    
    notebook = {
        "cells": [
            {
                "cell_type": "code",
                "execution_count": 1,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import os\n",
                    "os.chdir('..')"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Lab 1: Compare Memory-Based versus Generator-Based Data Loading\n",
                    "\n",
                    "## AI Capstone Project with Deep Learning\n",
                    "\n",
                    "This lab focuses on comparing memory-based and generator-based data loading approaches for agricultural land classification using satellite imagery.\n",
                    "\n",
                    "### Tasks:\n",
                    "1. Determine the shape (dimensions) of a single image stored in the image_data variable\n",
                    "2. Display the first four images in './images_dataSAT/class_0_non_agri/' directory\n",
                    "3. Create a list named agri_images_paths that contains the full file paths of all images located in the dir_agri directory\n",
                    "4. Determine the number of images of agricultural land that exist in the './images_dataSAT/class_1_agri/' directory"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": 2,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Create necessary directories for the lab\n",
                    "import os\n",
                    "\n",
                    "# Create directories if they don't exist (using absolute path from project root)\n",
                    "os.makedirs('../images_dataSAT/class_0_non_agri', exist_ok=True)\n",
                    "os.makedirs('../images_dataSAT/class_1_agri', exist_ok=True)\n",
                    "\n",
                    "print(\"Directories created successfully!\")\n",
                    "print(\"  - ../images_dataSAT/class_0_non_agri/\")\n",
                    "print(\"  - ../images_dataSAT/class_1_agri/\")"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": 3,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Create realistic satellite images for demonstration\n",
                    "def create_realistic_sample_images():\n",
                    "    # Create non-agricultural land images (class 0) - Urban/Forest areas\n",
                    "    for i in range(20):\n",
                    "        # Base image with urban/forest colors\n",
                    "        img = np.zeros((64, 64, 3), dtype=np.uint8)\n",
                    "        \n",
                    "        # Add urban/forest-like patterns\n",
                    "        if i < 10:  # Urban areas\n",
                    "            # Buildings and roads\n",
                    "            img[:, :] = [60, 60, 60]  # Dark gray base\n",
                    "            # Add some building-like structures\n",
                    "            for x in range(0, 64, 16):\n",
                    "                for y in range(0, 64, 16):\n",
                    "                    if np.random.random() > 0.3:\n",
                    "                        img[y:y+12, x:x+12] = [80, 80, 80]  # Buildings\n",
                    "            # Add roads\n",
                    "            img[30:34, :] = [40, 40, 40]  # Horizontal road\n",
                    "            img[:, 30:34] = [40, 40, 40]  # Vertical road\n",
                    "        else:  # Forest areas\n",
                    "            # Forest green base\n",
                    "            img[:, :] = [30, 60, 30]  # Dark green\n",
                    "            # Add tree-like patterns\n",
                    "            for x in range(0, 64, 8):\n",
                    "                for y in range(0, 64, 8):\n",
                    "                    if np.random.random() > 0.4:\n",
                    "                        img[y:y+6, x:x+6] = [20, 80, 20]  # Trees\n",
                    "        \n",
                    "        # Add some noise for realism\n",
                    "        noise = np.random.randint(-20, 20, (64, 64, 3))\n",
                    "        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)\n",
                    "        \n",
                    "        # Save image\n",
                    "        image_path = f'./images_dataSAT/class_0_non_agri/non_agri_{i:03d}.png'\n",
                    "        Image.fromarray(img).save(image_path)\n",
                    "    \n",
                    "    # Create agricultural land images (class 1) - Farm fields\n",
                    "    for i in range(25):\n",
                    "        # Base image with agricultural colors\n",
                    "        img = np.zeros((64, 64, 3), dtype=np.uint8)\n",
                    "        \n",
                    "        # Agricultural field patterns\n",
                    "        if i < 12:  # Crop fields\n",
                    "            # Brown soil base\n",
                    "            img[:, :] = [139, 69, 19]  # Saddle brown\n",
                    "            # Add crop rows\n",
                    "            for y in range(0, 64, 4):\n",
                    "                if y % 8 < 4:  # Every other row\n",
                    "                    img[y:y+2, :] = [34, 139, 34]  # Green crops\n",
                    "        else:  # Different crop types\n",
                    "            # Different soil colors\n",
                    "            base_colors = [[160, 82, 45],   # Sienna\n",
                    "                          [210, 180, 140],  # Tan\n",
                    "                          [222, 184, 135]]  # Burlywood\n",
                    "            img[:, :] = base_colors[i % 3]\n",
                    "            # Add different crop patterns\n",
                    "            for y in range(0, 64, 6):\n",
                    "                if y % 12 < 6:\n",
                    "                    img[y:y+3, :] = [50, 205, 50]  # Lime green crops\n",
                    "        \n",
                    "        # Add some variation and noise\n",
                    "        variation = np.random.randint(-15, 15, (64, 64, 3))\n",
                    "        img = np.clip(img.astype(np.int16) + variation, 0, 255).astype(np.uint8)\n",
                    "        \n",
                    "        # Save image\n",
                    "        image_path = f'./images_dataSAT/class_1_agri/agri_{i:03d}.png'\n",
                    "        Image.fromarray(img).save(image_path)\n",
                    "    \n",
                    "    print(\"Realistic satellite images created successfully!\")\n",
                    "\n",
                    "# Create realistic sample images\n",
                    "create_realistic_sample_images()"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": 4,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Import necessary libraries\n",
                    "import numpy as np\n",
                    "import matplotlib.pyplot as plt\n",
                    "import os\n",
                    "from PIL import Image\n",
                    "import glob\n",
                    "from pathlib import Path\n",
                    "import random"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": 5,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Create sample satellite images for demonstration\n",
                    "def create_sample_images():\n",
                    "    \"\"\"Create sample satellite images for agricultural land classification\"\"\"\n",
                    "    \n",
                    "    # Create non-agricultural land images (class 0)\n",
                    "    for i in range(20):\n",
                    "        # Create random image with urban/forest-like patterns\n",
                    "        img = np.random.randint(50, 150, (64, 64, 3), dtype=np.uint8)\n",
                    "        # Add some structure to make it look more realistic\n",
                    "        img[20:40, 20:40] = [100, 120, 80]  # Different colored region\n",
                    "        \n",
                    "        # Save image\n",
                    "        image_path = f'./images_dataSAT/class_0_non_agri/non_agri_{i:03d}.png'\n",
                    "        Image.fromarray(img).save(image_path)\n",
                    "    \n",
                    "    # Create agricultural land images (class 1)\n",
                    "    for i in range(25):\n",
                    "        # Create random image with agricultural patterns\n",
                    "        img = np.random.randint(80, 200, (64, 64, 3), dtype=np.uint8)\n",
                    "        # Add agricultural field patterns\n",
                    "        for j in range(0, 64, 8):\n",
                    "            img[j:j+4, :] = [120, 150, 100]  # Field rows\n",
                    "        \n",
                    "        # Save image\n",
                    "        image_path = f'./images_dataSAT/class_1_agri/agri_{i:03d}.png'\n",
                    "        Image.fromarray(img).save(image_path)\n",
                    "    \n",
                    "    print(\"Sample images created successfully!\")\n",
                    "\n",
                    "# Create sample images\n",
                    "create_sample_images()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Task 1: Determine the shape (dimensions) of a single image stored in the image_data variable"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": 6,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Task 1: Determine the shape (dimensions) of a single image stored in the image_data variable\n",
                    "print(\"Task 1: Determine image shape\")\n",
                    "\n",
                    "# Load a sample image to determine its shape\n",
                    "sample_image_path = './images_dataSAT/class_0_non_agri/non_agri_000.png'\n",
                    "image_data = Image.open(sample_image_path)\n",
                    "image_data = np.array(image_data)\n",
                    "\n",
                    "print(f\"Image shape: {image_data.shape}\")\n",
                    "print(f\"Image dimensions: {image_data.shape[0]} x {image_data.shape[1]} x {image_data.shape[2]}\")\n",
                    "print(f\"Data type: {image_data.dtype}\")\n",
                    "print(f\"Value range: [{image_data.min()}, {image_data.max()}]\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Task 2: Display the first four images in './images_dataSAT/class_0_non_agri/' directory"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": 7,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Task 2: Display the first four images in './images_dataSAT/class_0_non_agri/' directory\n",
                    "print(\"Task 2: Display first four non-agricultural images\")\n",
                    "\n",
                    "# Get the first four images from class_0_non_agri\n",
                    "non_agri_dir = './images_dataSAT/class_0_non_agri/'\n",
                    "image_files = sorted(glob.glob(non_agri_dir + '*.png'))[:4]\n",
                    "\n",
                    "# Create subplot to display images\n",
                    "fig, axes = plt.subplots(1, 4, figsize=(12, 3))\n",
                    "fig.suptitle('Non-Agricultural Land Images (Class 0)', fontsize=14)\n",
                    "\n",
                    "for i, image_path in enumerate(image_files):\n",
                    "    # Load and display image\n",
                    "    img = Image.open(image_path)\n",
                    "    axes[i].imshow(img)\n",
                    "    axes[i].set_title(f'Non-Agri {i+1}')\n",
                    "    axes[i].axis('off')\n",
                    "\n",
                    "plt.tight_layout()\n",
                    "plt.show()\n",
                    "\n",
                    "print(f\"Displayed {len(image_files)} non-agricultural images\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Task 3: Create a list named agri_images_paths that contains the full file paths of all images located in the dir_agri directory"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": 8,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Task 3: Create a list named agri_images_paths that contains the full file paths of all images located in the dir_agri directory\n",
                    "print(\"Task 3: Create agri_images_paths list\")\n",
                    "\n",
                    "# Define the agricultural images directory\n",
                    "dir_agri = './images_dataSAT/class_1_agri/'\n",
                    "\n",
                    "# Create list of all agricultural image paths\n",
                    "agri_images_paths = glob.glob(dir_agri + '*.png')\n",
                    "\n",
                    "# Sort the list as requested\n",
                    "agri_images_paths.sort()\n",
                    "\n",
                    "print(f\"Number of agricultural images: {len(agri_images_paths)}\")\n",
                    "print(f\"First 3 paths:\")\n",
                    "for i, path in enumerate(agri_images_paths[:3]):\n",
                    "    print(f\"  {i+1}. {path}\")\n",
                    "\n",
                    "print(f\"Last 3 paths:\")\n",
                    "for i, path in enumerate(agri_images_paths[-3:], len(agri_images_paths)-2):\n",
                    "    print(f\"  {i}. {path}\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Task 4: Determine the number of images of agricultural land that exist in the './images_dataSAT/class_1_agri/' directory"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": 9,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Task 4: Determine the number of images of agricultural land that exist in the './images_dataSAT/class_1_agri/' directory\n",
                    "print(\"Task 4: Count agricultural land images\")\n",
                    "\n",
                    "# Count the number of agricultural images\n",
                    "agri_dir = './images_dataSAT/class_1_agri/'\n",
                    "agri_count = len(glob.glob(agri_dir + '*.png'))\n",
                    "\n",
                    "print(f\"Number of agricultural land images: {agri_count}\")\n",
                    "print(f\"Directory: {agri_dir}\")\n",
                    "\n",
                    "# Also display the first four agricultural images\n",
                    "print(\"\\nFirst four agricultural images:\")\n",
                    "agri_files = sorted(glob.glob(agri_dir + '*.png'))[:4]\n",
                    "\n",
                    "fig, axes = plt.subplots(1, 4, figsize=(12, 3))\n",
                    "fig.suptitle('Agricultural Land Images (Class 1)', fontsize=14)\n",
                    "\n",
                    "for i, image_path in enumerate(agri_files):\n",
                    "    img = Image.open(image_path)\n",
                    "    axes[i].imshow(img)\n",
                    "    axes[i].set_title(f'Agri {i+1}')\n",
                    "    axes[i].axis('off')\n",
                    "\n",
                    "plt.tight_layout()\n",
                    "plt.show()\n",
                    "\n",
                    "print(f\"\\nTask 4 completed: Found {agri_count} agricultural land images\")"
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
    """Create clean Question 1 notebook"""
    print("Creating clean Question 1 notebook...")
    
    # Create clean Question 1
    clean_notebook = create_clean_question_1()
    
    # Save the clean notebook
    with open('notebooks/Question 1.ipynb', 'w', encoding='utf-8') as f:
        json.dump(clean_notebook, f, indent=1, ensure_ascii=False)
    
    print("Clean Question 1 notebook created successfully!")

if __name__ == "__main__":
    main()
