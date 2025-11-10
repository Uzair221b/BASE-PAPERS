# How to Classify Fundus Images (Positive/Negative)

## Quick Start

### Step 1: Place Your Images in a Folder

Create a folder and put all your fundus images inside:
```
my_images/
├── image1.jpg
├── image2.png
├── image3.jpg
└── ...
```

### Step 2: Run Classification

```bash
cd preprocessing
python classify_images.py --folder path/to/your/images/folder/
```

**Example:**
```bash
python classify_images.py --folder ../my_fundus_images/ --output my_results.csv
```

### Step 3: Check Results

The script creates **two CSV files**:

1. **`classifications.csv`** - Full results with:
   - Image_Name
   - Image_Path
   - Label (1=Positive, 0=Negative)
   - Label_Text (Positive/Negative)
   - Confidence

2. **`classifications_simple.csv`** - Simple format with:
   - Image_Name
   - Label (1 or 0)

## CSV Format

### Full CSV (`classifications.csv`)
```csv
Image_Name,Image_Path,Label,Label_Text,Confidence
image1.jpg,path/to/image1.jpg,1,Positive,0.6234
image2.jpg,path/to/image2.jpg,0,Negative,0.3421
```

### Simple CSV (`classifications_simple.csv`)
```csv
Image_Name,Label
image1.jpg,1
image2.jpg,0
```

**Label meanings:**
- `1` = **Positive** (Glaucoma detected)
- `0` = **Negative** (Normal, no glaucoma)

## Usage Examples

### Basic Usage
```bash
python classify_images.py --folder my_images/
```
Output: `classifications.csv` and `classifications_simple.csv`

### Custom Output File
```bash
python classify_images.py --folder my_images/ --output results.csv
```
Output: `results.csv` and `results_simple.csv`

### Using a Trained Model (When Available)
```bash
python classify_images.py --folder my_images/ --model trained_model.h5
```

## Important Notes

### Current Status

⚠️ **Important:** The script currently uses a **placeholder classification** method based on simple image characteristics. For accurate classification, you need a **trained deep learning model**.

### What's Working Now:
✅ Image preprocessing (all 5 techniques applied)
✅ Image loading from folder
✅ CSV file generation with labels
✅ Classification structure ready

### What You Need:
- A trained model file (`.h5` or `.keras`) OR
- Labeled training data to train a model

### Next Steps to Get Real Classification:

**Option 1: Train Your Own Model**
- I can create a training script
- You provide labeled dataset (images with known labels)
- Train the model
- Use trained model for classification

**Option 2: Use Pre-trained Model**
- Download a pre-trained glaucoma detection model
- Load it using `--model` flag

**Option 3: Improve Placeholder**
- The current placeholder uses basic heuristics
- This is NOT accurate for real diagnosis
- Should only be used for testing the pipeline

## Output Summary

After running, you'll see:
```
Classification Summary:
  Positive (Glaucoma): X
  Negative (Normal): Y
  Errors: Z
```

The CSV files are saved in the current directory.

## Troubleshooting

**No images found?**
- Check folder path is correct
- Ensure images are `.jpg`, `.png`, or `.jpeg`

**Classification seems inaccurate?**
- Current method is placeholder only
- Need trained model for accurate results

**CSV file not created?**
- Check write permissions in directory
- Check for errors in console output

## Ready to Use!

Just provide the folder path with your images and run the command. The script will:
1. ✅ Load all images from folder
2. ✅ Preprocess each image (scaling, cropping, normalization, CLAHE)
3. ✅ Classify as Positive (1) or Negative (0)
4. ✅ Create CSV files with results

**Example command:**
```bash
python preprocessing/classify_images.py --folder "C:/path/to/your/images/"
```





