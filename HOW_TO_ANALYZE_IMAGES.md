# How to Analyze Fundus Images for Glaucoma Detection

## Quick Start

### Option 1: Process a Single Image

```bash
cd preprocessing
python analyze_images.py --image path/to/your/fundus_image.jpg
```

**Example:**
```bash
python analyze_images.py --image glaucoma.png --output my_processed_images/
```

This will:
- Load your image
- Apply all preprocessing steps (scaling, cropping, normalization, CLAHE)
- Save the preprocessed image to the output directory

### Option 2: Process a Directory of Images

```bash
python analyze_images.py --dir path/to/your/images/folder/
```

**Example:**
```bash
python analyze_images.py --dir ../data/fundus_images/ --output processed/
```

This processes all images (.jpg, .png, .jpeg) in the specified directory.

### Option 3: Visualize Preprocessing Steps

See each step of preprocessing applied to your image:

```bash
python analyze_images.py --image path/to/image.jpg --visualize
```

This creates a visualization showing:
- Original image
- After scaling
- After cropping
- After normalization
- After CLAHE
- Final processed image

## Ways to Provide Images

### Method 1: Place Images in This Directory

1. Copy your fundus image(s) to the workspace directory
2. Run:
   ```bash
   python preprocessing/analyze_images.py --image your_image.jpg
   ```

### Method 2: Use Full Path

```bash
python preprocessing/analyze_images.py --image "C:/Users/YourName/Desktop/fundus_image.jpg"
```

### Method 3: Create a Folder with All Images

1. Create a folder: `my_fundus_images/`
2. Put all your images in that folder
3. Run:
   ```bash
   python preprocessing/analyze_images.py --dir my_fundus_images/
   ```

## Supported Image Formats

- `.jpg` / `.jpeg`
- `.png`
- Any format supported by OpenCV

## What Gets Processed?

The pipeline applies these 5 techniques:
1. ✅ **Scaling** to 224×224 pixels
2. ✅ **Cropping** to center optic disc region
3. ✅ **Color normalization** (z-score)
4. ✅ **CLAHE enhancement** for better contrast
5. ✅ Ready for classification (classification model can be added later)

## Output

Processed images are saved to:
- Default: `processed_output/` folder
- Custom: Use `--output your_folder/` flag

## Important Note: Classification

**Currently implemented:** Preprocessing pipeline ✅  
**To be added:** Classification model to determine positive/negative

The preprocessing prepares your images for classification. To actually classify images as:
- **Positive (Glaucoma)** 
- **Negative (Normal)**

We need to either:
1. Train a classification model (requires labeled dataset)
2. Use a pre-trained model
3. Integrate with an existing model

Would you like me to:
- **A)** Create a classification module that we can train?
- **B)** Integrate with a pre-trained model?
- **C)** Show you how to use the preprocessed images with an existing model?

## Example Usage Workflow

```bash
# Step 1: Process a single image
python preprocessing/analyze_images.py --image growthma.png

# Step 2: Process multiple images
python preprocessing/analyze_images.py --dir fundus_dataset/ --output processed/

# Step 3: See preprocessing visualization
python preprocessing/analyze_images.py --image glaucoma.png --visualize
```

## Troubleshooting

**Image not found?**
- Use full path: `--image "C:/full/path/to/image.jpg"`
- Check file extension: `.jpg`, `.png`, `.jpeg`

**Permission errors?**
- Make sure you have write permissions for the output directory

**Images not processing correctly?**
- Ensure images are fundus images (retinal photographs)
- Check that images are valid image files

## Next Steps

After preprocessing, you can:
1. Inspect the processed images in the output folder
2. Use them to train a classification model
3. Use them with an existing model for glaucoma detection




