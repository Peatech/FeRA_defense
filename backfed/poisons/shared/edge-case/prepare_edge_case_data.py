#!/usr/bin/env python3
"""
Edge-Case Dataset Preparation Script

This script extracts airplane images (class 0) from CIFAR-10 to create
the edge-case (semantic backdoor) attack dataset.

Usage:
    python backfed/poisons/shared/edge-case/prepare_edge_case_data.py
"""

import torch
import torchvision
import pickle
from pathlib import Path
import sys

def main():
    """Extract airplane images from CIFAR-10 and save as pickle files."""
    
    print("=" * 70)
    print("Edge-Case Dataset Preparation")
    print("=" * 70)
    print()
    
    # Set data directory
    data_dir = './data'
    output_dir = Path('backfed/poisons/shared/edge-case')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Load CIFAR-10 datasets
    print("Loading CIFAR-10 training dataset...")
    trainset = torchvision.datasets.CIFAR10(
        root=data_dir, 
        train=True, 
        download=True
    )
    print(f"  Total training images: {len(trainset)}")
    
    print("Loading CIFAR-10 test dataset...")
    testset = torchvision.datasets.CIFAR10(
        root=data_dir, 
        train=False, 
        download=True
    )
    print(f"  Total test images: {len(testset)}")
    print()
    
    # Filter for airplane images (class 0)
    print("Filtering airplane images (class 0)...")
    airplane_train = []
    airplane_test = []
    
    # Process training set
    for img, label in trainset:
        if label == 0:  # Airplane class
            airplane_train.append(img)
    
    # Process test set
    for img, label in testset:
        if label == 0:  # Airplane class
            airplane_test.append(img)
    
    print(f"  Training airplanes: {len(airplane_train)}")
    print(f"  Test airplanes: {len(airplane_test)}")
    print()
    
    # Verify we have images
    if len(airplane_train) == 0 or len(airplane_test) == 0:
        print("ERROR: No airplane images found!")
        sys.exit(1)
    
    # Save as pickle files
    train_path = output_dir / 'southwest_images_new_train.pkl'
    test_path = output_dir / 'southwest_images_new_test.pkl'
    
    print("Saving pickle files...")
    print(f"  Training: {train_path}")
    with open(train_path, 'wb') as f:
        pickle.dump(airplane_train, f)
    
    print(f"  Test: {test_path}")
    with open(test_path, 'wb') as f:
        pickle.dump(airplane_test, f)
    print()
    
    # Verify saved files
    print("Verifying saved files...")
    try:
        with open(train_path, 'rb') as f:
            loaded_train = pickle.load(f)
        with open(test_path, 'rb') as f:
            loaded_test = pickle.load(f)
        
        print(f"  ✓ Training file loaded: {len(loaded_train)} images")
        print(f"  ✓ Test file loaded: {len(loaded_test)} images")
        print(f"  ✓ Sample image type: {type(loaded_train[0])}")
        print(f"  ✓ Sample image size: {loaded_train[0].size}")
    except Exception as e:
        print(f"  ✗ Verification failed: {e}")
        sys.exit(1)
    
    print()
    print("=" * 70)
    print("✓ Edge-Case Dataset Preparation Complete!")
    print("=" * 70)
    print()
    print("Next steps:")
    print("  1. Run test script: python test_edge_case.py")
    print("  2. Try quick test: python main.py ... atk_config.data_poison_method=edge_case")
    print()

if __name__ == "__main__":
    main()


