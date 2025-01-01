import argparse
import os
import shutil

def prepare_validation_data(data_dir, labels_file):
    """
    Organize validation images into class folders based on labels file
    
    Args:
        data_dir: Directory containing the validation images
        labels_file: File containing image name to class label mapping
    """
    processed_classes = set()

    with open(labels_file, "r") as file:
        # skip header
        next(file)
        for line in file:
            img_name, labels = line.split(",")
            class_name = labels.split(" ")[0]
            
            # create a dir for this classname
            if class_name not in processed_classes:
                dir_path = os.path.join(data_dir, class_name)
                if not os.path.exists(dir_path):
                    os.makedirs(dir_path)
                processed_classes.add(class_name)
                
            # Move image to class directory    
            src = os.path.join(data_dir, f"{img_name}.JPEG")
            dst = os.path.join(data_dir, class_name, f"{img_name}.JPEG")
            shutil.move(src, dst)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dir", help="dir with the images", required=True)
    parser.add_argument("-l", "--labels", help="file with image name to class label mapping", required=True)

    args = parser.parse_args()
    prepare_validation_data(args.dir, args.labels)

if __name__ == "__main__":
    main() 