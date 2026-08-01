## Error Analysis Findings — faster_rcnn

### Summary
- **Total GT objects**: 1263
- **Total predictions**: 2109
- **Correct detections (TP)**: 664 (52.6% of GT)
- **False Positives**: 1445
- **False Negatives**: 599
- **Precision**: 0.3148
- **Recall**: 0.5257
- **F1 Score**: 0.3938

### Most Common Errors
1. **Background Fp** (1165 instances, 57.0% of errors)
2. **Missed** (599 instances, 29.3%)
3. **Localization** (280 instances, 13.7%)

### Size-Based Analysis
- **Small objects** (<1% of image): 47.4% miss rate (599/1263)
- **Medium objects** (1-10%): 0.0% miss rate (0/0)
- **Large objects** (>10%): 0.0% miss rate (0/0)
- **Worst category**: small objects (47.4% miss rate)

### Key Findings
- **Critical**: Small object detection is severely impaired — models struggle with objects <1% of image area
- **Good**: Large objects are detected reliably (>90% recall)
- **Localization**: 280 predictions had IoU in [0.1, 0.5) — boxes are roughly correct but imprecise
- **Background FPs**: 1165 predictions with IoU < 0.1 — likely false alarms on similar-looking terrain

### Recommendations
1. **Small-object augmentation**: Apply mosaic, random crop+resize, or Copy-Paste augmentation to improve small object detection
2. **Confidence threshold tuning**: Consider raising conf threshold if background FP rate is high
3. **Data imbalance**: If large objects dominate recall but small objects fail, add more small-object training examples
4. **Model ensemble**: Combine Faster R-CNN (better localization) with YOLO (better recall) for complementary strengths
