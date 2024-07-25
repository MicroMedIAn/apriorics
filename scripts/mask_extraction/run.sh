#!/bin/bash

set -e
data_path=$(yq .base.data_path params.yaml)
ihc_type=$(yq .base.ihc_type params.yaml)
he_slide_folder=$data_path/$(yq .local.main_slide_path params.yaml)/$ihc_type/HE
ihc_slide_folder=$data_path/$(yq .local.main_slide_path params.yaml)/$ihc_type/IHC
slide_folder=$data_path/$(yq .local.slide_path params.yaml)
outfolder=$data_path/$(yq .local.mask_path params.yaml)/$ihc_type/HE
mask_extension=$(yq .base.mask_extension params.yaml)
eval "$(conda shell.bash hook)"
conda activate apriorics
for file_path in $he_slide_folder/*.svs; do
  filename=$(basename $file_path)
  filestem=${filename%.*}
  outfile=$outfolder/$filestem$mask_extension
  if [ -f $outfile ]; then
    continue
  fi
  echo $filestem
  mkdir -p $slide_folder/$ihc_type/HE
  mkdir -p $slide_folder/$ihc_type/IHC
  cp $file_path $slide_folder/$ihc_type/HE
  cp $ihc_slide_folder/$filename $slide_folder/$ihc_type/IHC
  dvc repro -f -s mask_extraction >> $HOME/mask_extraction_logs 2>&1 
  if [ ! $? -eq 0 ]; then
    dvc commit -f mask_extraction
  fi
  rm $slide_folder/$ihc_type/HE/$filename
  rm $slide_folder/$ihc_type/IHC/$filename
done
