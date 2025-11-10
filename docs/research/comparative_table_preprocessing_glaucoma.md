# Comparative Table: Preprocessing Techniques for Glaucoma Detection in Fundus and OCT Images

## Executive Summary
This table compares preprocessing techniques from two research papers on glaucoma detection using fundus images. The studies focus on deep learning approaches for mobile and code-free applications.

## Paper Information

**Paper 1:** Esengönül, M., & Cunha, A. (2023). Glaucoma Detection using Convolutional Neural Networks for Mobile Use. *Procedia Computer Science, 219*, 1153-1160.

**Paper 2:** Milad, D., et al. (2025). Code-Free Deep Learning Glaucoma Detection on Color Fundus Images. *Ophthalmology Science, 5*, 100721.

---

## Comparative Table of Preprocessing Techniques

| Technique | Paper | Image Type | Methodology | Advantages | Disadvantages | Performance Impact | Application Suitability |
|-----------|-------|------------|-------------|------------|---------------|-------------------|----------------------|
| **1. Scaling to 224×224 pixels** | Paper 1 | Fundus | Resize images to 224×224 pixels to match MobileNet input requirements | • Standardizes image dimensions<br>• Required for transfer learning<br>• Memory efficient<br>• Enables batch processing | • May lose important details in downsizing<br>• Requires interpolation<br>• Aspect ratio may be distorted | Achieved 0.967 accuracy with AIROGS dataset | Mobile applications, deep learning pipelines |
| **2. CLAHE (Contrast Limited Adaptive Histogram Equalization)** | Paper 1 | Fundus | Apply CLAHE after cropping to enhance local contrast | • Improves contrast in fundus images<br>• Preserves image details<br>• Reduces over-amplification<br>• Handles lighting variations | • Computationally expensive<br>• May produce artifacts<br>• Requires parameter tuning | Enhanced discrimination for optic disc features | Medical imaging, low-quality images |
| **3. Grayscale Conversion** | Paper 1 | Fundus | Convert RGB images to grayscale | • Reduces computational complexity<br>• Decreases memory usage<br>• Speeds up processing<br>• Sufficient for certain features | • Loses color information<br>• May reduce diagnostic accuracy<br>• Less informative for vascular structures | Faster processing on mobile devices | Resource-constrained devices |
| **4. Cropping** | Paper 1 | Fundus | Crop images to focus on relevant regions | • Reduces irrelevant background<br>• Focuses on optic nerve<br>• Reduces computational load<br>• Improves feature extraction | • May remove important context<br>• Requires accurate region detection<br>• Risk of losing peripheral features | Improved training efficiency | Automated screening systems |
| **5. Data Augmentation (Zoom & Rotation)** | Paper 1 | Fundus | Apply zoom factor of 0.035 and rotation range of 0.025 | • Prevents overfitting<br>• Increases dataset variability<br>• Improves generalization<br>• Increases robustness | • Requires parameter tuning<br>• Slight transformation only<br>• Does not include flipping | Maintained comparable fundus appearance | Limited dataset scenarios |
| **6. Random Undersampling** | Paper 2 | Fundus | Balance classes by undersampling majority class (NRG) to achieve 1:2 ratio (RG:NRG) | • Addresses class imbalance<br>• Prevents bias toward majority class<br>• Improves model sensitivity<br>• Better training balance | • Loss of data from majority class<br>• May reduce overall dataset size<br>• Potential information loss | Changed prevalence from 6.5% to 33.33% | Imbalanced datasets |
| **7. No Pre-Processing Augmentation** | Paper 2 | Fundus | Upload images directly to Google Cloud Vertex AI without augmentation | • Simplifies preprocessing pipeline<br>• Faster implementation<br>• Relies on model robustness<br>• Reduces complexity | • Relies heavily on model capability<br>• No manual optimization<br>• May require larger datasets<br>• Less control over input quality | Achieved 0.988 AuPRC | Code-free deep learning platforms |
| **8. Class Balancing (Case:Control Ratio)** | Paper 2 | Fundus | Achieve 1:2 RG:NRG ratio to avoid perpetuating class imbalance | • Better representation of both classes<br>• Improved sensitivity<br>• More reliable metrics<br>• Statistical validity | • Reduces total training samples<br>• May underrepresent minority class<br>• Manual threshold selection | Improved sensitivity at 95% specificity | Research studies requiring balanced classes |

---

## Selection of Three Best Preprocessing Techniques

Based on the analysis of both papers, we recommend the following three preprocessing techniques as optimal for glaucoma detection:

### 1. **CLAHE (Contrast Limited Adaptive Histogram Equalization)** ⭐⭐⭐⭐⭐
**Reasoning:**
- Significantly improves contrast in fundus images, which is crucial for detecting optic nerve head changes
- Handles varying illumination conditions common in fundus photography
- Preserves important anatomical details while enhancing visibility
- Critical for accurate feature extraction in glaucoma detection
- Part of preprocessing pipeline in Paper 1, which achieved 0.967 accuracy

**Implementation:** Apply after initial cropping and before model input

### 2. **Scaling to Standard Dimensions (224×224 or appropriate size)** ⭐⭐⭐⭐⭐
**Reasoning:**
- Essential for transfer learning with pre-trained models (MobileNet, ResNet, etc.)
- Enables batch processing and memory efficiency
- Standard requirement for most deep learning frameworks
- Maintains compatibility across different architectures
- Used by both papers with successful results

**Implementation:** Resize all images to 224×224 pixels or model-specific input size

### 3. **Smart Class Balancing (Random Undersampling with 1:2 ratio)** ⭐⭐⭐⭐⭐
**Reasoning:**
- Addresses severe class imbalance in glaucoma screening (low prevalence)
- Prevents model bias toward majority class
- Improves sensitivity metrics crucial for medical screening
- Paper 2 achieved 95% SE@95SP using this technique
- More effective than naive training on imbalanced data

**Implementation:** Use random undersampling to achieve 1:2 or 1:3 RG:NRG ratio

---

## Recommended Preprocessing Pipeline

For glaucoma detection in fundus images, use this sequential pipeline:

1. **Image Acquisition** → Raw fundus images
2. **Scaling** → Resize to 224×224 pixels (or model-specific size)
3. **Cropping** → Focus on optic nerve region (if automated cropping is available)
4. **Grayscale Conversion** → Reduce complexity (optional, for mobile applications)
5. **CLAHE** → Enhance local contrast
6. **Data Augmentation** → Apply subtle zoom (0.035) and rotation (0.025)
7. **Class Balancing** → Apply random undersampling if needed

---

## Key Insights

- **Paper 1** achieved best results with AIROGS dataset (96.7% accuracy) using comprehensive preprocessing
- **Paper 2** achieved excellent results (AuPRC 0.988) with minimal preprocessing but sophisticated class balancing
- Both approaches are valid but target different use cases:
  - Paper 1: Mobile applications requiring local processing
  - Paper 2: Cloud-based solutions with online training

---

## Performance Comparison

| Metric | Paper 1 (Best) | Paper 2 (Best) |
|--------|----------------|----------------|
| **Accuracy** | 96.7% (AIROGS) | 91% |
| **AUC** | 0.967 | 0.988 |
| **Sensitivity @ 95% Specificity** | - | 95% |
| **Dataset Size** | 7,214 images | 9,810 images (after balancing) |

---

## Conclusion

The three selected preprocessing techniques (CLAHE, Scaling, and Class Balancing) provide the optimal foundation for glaucoma detection systems. These techniques address the core challenges in fundus image analysis: contrast enhancement, computational efficiency, and class imbalance.

**For Mobile Applications:** CLAHE + Scaling + Minimal Augmentation
**For Cloud/AI Platforms:** Scaling + Class Balancing + Automatic Processing

Both approaches demonstrate that appropriate preprocessing is crucial for achieving high-performance glaucoma detection systems suitable for clinical screening.





