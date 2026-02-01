import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm
import logging
import csv
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

import utils
from model import Model

# Define Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_args():
    parser = argparse.ArgumentParser(description='Linear Evaluation on CIFAR-10')
    parser.add_argument('--model_path', type=str, default='/shared/ysh/AUC-CL_mac/cifar/results/auc-cl_lr0.001_bs128_model.pth', help='Path to the pretrained model checkpoint')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for training and testing')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs for linear evaluation')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--feature_dim', type=int, default=128, help='Feature dimension used in pretraining (needed to load model)')
    parser.add_argument('--data_dir', type=str, default='./data', help='Path to dataset')
    parser.add_argument('--output_dir', type=str, default='auc_linear_eval_results', help='Directory to save logs and results')
    return parser.parse_args()

def setup_logger(output_dir):
    log_file = os.path.join(output_dir, 'linear_eval.log')
    
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    # Stream handler (console)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(console_handler)
    
    return logger

def main():
    args = get_args()
    
    # Prepare Output Directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    # Setup Logger
    logger = setup_logger(args.output_dir)
    logger.info(f"Running Linear Evaluation with: {args}")

    # Setup TensorBoard
    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, 'tensorboard'))
    
    # Setup CSV
    csv_file_path = os.path.join(args.output_dir, 'metrics.csv')
    csv_file = open(csv_file_path, mode='w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Epoch', 'Train Loss', 'Train Acc', 'Test Loss', 'Test Acc', 'Precision', 'Recall', 'F1-Score'])

    # 1. Data Preparation
    # Standard augmentation for linear evaluation (SimCLR/MoCo paper style)
    # Usually: RandomResizedCrop, RandomHorizontalFlip. No color jitter.
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ])

    test_transform = transforms.Compose([
        transforms.Resize(32), # Optional, but good to be explicit
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ])

    # Use CIFAR10Single for supervised linear eval
    train_data = utils.CIFAR10Single(root=args.data_dir, train=True, transform=train_transform, download=True)
    test_data = utils.CIFAR10Single(root=args.data_dir, train=False, transform=test_transform, download=True)

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)

    # 2. Load Pretrained Model
    logger.info("Loading pretrained model...")
    # Initialize the model structure matching the training script
    # feature_dim must match the one used during pretraining to load weights into self.g correctly,
    # even though we won't use self.g or self.classifier.
    pretrained_model = Model(feature_dim=args.feature_dim).to(DEVICE)
    
    if os.path.isfile(args.model_path):
        checkpoint = torch.load(args.model_path, map_location=DEVICE)
        # Handle case where checkpoint is state_dict vs full checkpoint dict
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
            
        # If the model was wrapped in DataParallel, keys might have 'module.' prefix
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k.replace("module.", "")
            new_state_dict[name] = v
        
        pretrained_model.load_state_dict(new_state_dict, strict=False) 
        logger.info("Model loaded successfully.")
    else:
        logger.error(f"Error: No checkpoint found at {args.model_path}")
        return

    # 3. Setup Linear Classifier
    # Extract encoder
    encoder = pretrained_model.f
    
    # Freeze encoder
    for param in encoder.parameters():
        param.requires_grad = False
    
    # The output dim of the encoder (ResNet50 modified) is 2048
    enc_dim = 2048
    
    classifier = nn.Linear(enc_dim, 10).to(DEVICE)
    
    # Optimizer
    optimizer = optim.Adam(classifier.parameters(), lr=args.lr, weight_decay=1e-6)
    criterion = nn.CrossEntropyLoss()

    # 4. Training Loop
    logger.info("Starting training...")
    best_acc = 0.0
    
    for epoch in range(1, args.epochs + 1):
        encoder.eval() # Ensure encoder is in eval mode (BatchNorm stats frozen)
        classifier.train()
        
        total_loss = 0.0
        total_correct = 0
        total_num = 0
        
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch}/{args.epochs} [Train]')
        
        for images, labels in train_bar:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            with torch.no_grad():
                # Forward through encoder
                features = encoder(images)
                # Flatten
                features = torch.flatten(features, start_dim=1)
                
            # Forward through classifier
            logits = classifier(features)
            loss = criterion(logits, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * images.size(0)
            _, predicted = torch.max(logits.data, 1)
            total_correct += (predicted == labels).sum().item()
            total_num += images.size(0)
            
            train_bar.set_description(f'Epoch {epoch}/{args.epochs} [Train] Loss: {total_loss/total_num:.4f} Acc: {100.*total_correct/total_num:.2f}%')
        
        avg_train_loss = total_loss / total_num
        avg_train_acc = 100. * total_correct / total_num
        
        # Evaluation
        metrics = test(encoder, classifier, test_loader, criterion)
        
        # Logging
        logger.info(f"Epoch {epoch} | Train Loss: {avg_train_loss:.4f} | Train Acc: {avg_train_acc:.2f}% | "
                    f"Test Loss: {metrics['loss']:.4f} | Test Acc: {metrics['accuracy']:.2f}% | "
                    f"Prec: {metrics['precision']:.4f} | Rec: {metrics['recall']:.4f} | F1: {metrics['f1']:.4f}")
        
        # TensorBoard
        writer.add_scalars('Loss', {'train': avg_train_loss, 'test': metrics['loss']}, epoch)
        writer.add_scalars('Accuracy', {'train': avg_train_acc, 'test': metrics['accuracy']}, epoch)
        writer.add_scalar('Precision/test', metrics['precision'], epoch)
        writer.add_scalar('Recall/test', metrics['recall'], epoch)
        writer.add_scalar('F1-Score/test', metrics['f1'], epoch)
        
        # CSV
        csv_writer.writerow([epoch, avg_train_loss, avg_train_acc, metrics['loss'], metrics['accuracy'], metrics['precision'], metrics['recall'], metrics['f1']])
        csv_file.flush() # Ensure data is written
        
        if metrics['accuracy'] > best_acc:
            best_acc = metrics['accuracy']
            # Optionally save the best linear classifier
            torch.save(classifier.state_dict(), os.path.join(args.output_dir, 'best_linear_classifier.pth'))
    
    logger.info(f"Best Test Accuracy: {best_acc:.2f}%")
    writer.close()
    csv_file.close()

def test(encoder, classifier, loader, criterion):
    encoder.eval()
    classifier.eval()
    
    all_preds = []
    all_targets = []
    total_loss = 0.0
    total_num = 0
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc='[Test]'):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            features = encoder(images)
            features = torch.flatten(features, start_dim=1)
            outputs = classifier(features)
            
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            total_num += images.size(0)
            
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
            
    # Calculate Metrics
    accuracy = 100 * np.mean(np.array(all_preds) == np.array(all_targets))
    precision = precision_score(all_targets, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_targets, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    avg_loss = total_loss / total_num
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

if __name__ == '__main__':
    main()