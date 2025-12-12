# Training Report

## Architecture Selection

Several YOLO architecture variants were evaluated for this fire detection task, specifically YOLOv8, YOLOv9, and YOLOv11. Due to hardware constraints (NVIDIA GeForce RTX 3050 Laptop GPU with 3768MiB VRAM), only the small model variants were considered.

**Key Observations:**

- YOLOv8s demonstrated competitive performance during preliminary testing
- YOLOv11s was selected as the final architecture for the following reasons:
  - Latest generation in the YOLO family with architectural improvements
  - Qualitative evaluation through visual inspection of detection results showed no significant performance advantage from older architectures. 

## Hyperparameter Configuration

Due to the extensive search space and computational requirements, formal hyperparameter tuning was not performed. Training with exhaustive grid or random search would have required several days of GPU time. The batch size was limited to 12 to prevent GPU memory overflow. 

**Configuration Decisions:**

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Learning Rate Scheduler | Cosine (`cos_lr: True`) | Provided measurable performance improvement over constant learning rate |
| Early Stopping | Enabled (`patience: 10`) | Prevents overfitting by monitoring validation metrics |
| Other Parameters | Default | Ultralytics defaults are well-tuned for general object detection tasks |


## Results Summary

### Training Performance

The final model selected was YOLOv11s (small variant). Training was configured for 100 epochs with early stopping enabled.

- **Total Epochs Trained**: 51 (early stopping triggered)
- **Best Checkpoint**: Epoch 41

**Best Training Metrics (Epoch 41):**

| Metric | Value |
|--------|-------|
| GPU Memory | 3.05 GB |
| Box Loss | 1.481 |
| Classification Loss | 1.267 |
| DFL Loss | 1.313 |
| Precision (P) | 0.554 |
| Recall (R) | 0.508 |
| mAP@50 | 0.517 |
| mAP@50-95 | 0.241 |

### Validation Performance

Final evaluation on the validation set (IoU threshold: 0.7):

| Metric | Value |
|--------|-------|
| Images | 150 |
| Instances | 256 |
| Precision (P) | 0.528 |
| Recall (R) | 0.434 |
| mAP@50 | 0.437 |
| mAP@50-95 | 0.206 |

### Analysis

The model achieves moderate precision (52.8%) and recall (43.4%) on the validation set (not perfect but plausible). Evaluation was performed with a stricter IoU threshold of 0.7, meaning predictions must overlap at least 70% with ground truth to be considered correct. The mAP@50 of 0.437 indicates reasonable detection capability at a 50% IoU threshold, while the mAP@50-95 of 0.206 reflects the challenge of precise bounding box localisation across stricter IoU thresholds.

**Potential Improvements:**

- Increase training data volume and diversity
- Experiment with larger model variants if hardware permits
- Implement formal hyperparameter optimisation