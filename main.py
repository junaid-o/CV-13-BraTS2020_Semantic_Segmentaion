import os
import sys
import argparse
import subprocess



sys.path.append(os.path.abspath('.'))

def main():
    parser = argparse.ArgumentParser(description='Run various scripts for BraTS Segmentation project.')
    parser.add_argument('--train', action='store_true', help='Run the trainer script')
    parser.add_argument('--prepare-data', action='store_true', help='Run the data preparation script')
    args = parser.parse_args()

    if args.train:
        trainer_script_path = os.path.abspath(os.path.join('src', 'BraTS_Segmentation','components', 'trainer.py'))
        subprocess.call(['python', trainer_script_path])

    if args.prepare_data:
        data_preparation_script_path = os.path.abspath(os.path.join('src', 'BraTS_Segmentation', 'components', 'data_preparation.py'))
        subprocess.call(['python', data_preparation_script_path])

if __name__ == "__main__":
    main()
