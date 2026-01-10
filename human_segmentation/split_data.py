import os
import shutil
import random

def split_data():
    random.seed(23)
    
    src_images = 'human segmentation/tmp/images'
    src_masks = 'human segmentation/tmp/masks'
    
    images = [f for f in os.listdir(src_images) if f.endswith(('.jpg', '.png', '.jpeg'))]
    random.shuffle(images)
    
    split_idx = int(len(images) * 0.9)
    train_imgs = images[:split_idx]
    val_imgs = images[split_idx:]
    
    os.makedirs('human segmentation/dataset/train/images', exist_ok=True)
    os.makedirs('human segmentation/dataset/train/masks', exist_ok=True)
    os.makedirs('human segmentation/dataset/val/images', exist_ok=True)
    os.makedirs('human segmentation/dataset/val/masks', exist_ok=True)
    
    for img in train_imgs:
        shutil.copy(
            os.path.join(src_images, img), 
            os.path.join('human segmentation/dataset/train/images', img)
        )
        
        name_no_ext = os.path.splitext(img)[0]
        
        for ext in ['.jpg', '.png', '.jpeg', '.JPG', '.PNG', '.JPEG']:
            mask_path = os.path.join(src_masks, name_no_ext + ext)
            if os.path.exists(mask_path):
                shutil.copy(
                    mask_path, 
                    os.path.join('human segmentation/dataset/train/masks', name_no_ext + '.jpg')
                )
                break
    
    for img in val_imgs:
        shutil.copy(
            os.path.join(src_images, img), 
            os.path.join('human segmentation/dataset/val/images', img)
        )
        
        name_no_ext = os.path.splitext(img)[0]
        for ext in ['.jpg', '.png', '.jpeg', '.JPG', '.PNG', '.JPEG']:
            mask_path = os.path.join(src_masks, name_no_ext + ext)
            if os.path.exists(mask_path):
                shutil.copy(
                    mask_path, 
                    os.path.join('human segmentation/dataset/val/masks', name_no_ext + '.jpg')
                )
                break
    
    print(f"train={len(train_imgs)}, val={len(val_imgs)}")

if __name__ == "__main__":
    split_data()