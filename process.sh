#!/bin/bash
folder="/media/user/File/SF_3/yes/"
#target_foloer="/media/user/File/SF_32/"
all_folder=$(ls $folder)
for file in ${all_folder}
do
  python3 predict_dir.py --input $folder$file
done