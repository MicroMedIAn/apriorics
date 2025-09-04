#!/bin/bash

set -e
data_path=$(yq .base.data_path params.yaml)
ihc_type=$(yq .base.ihc_type params.yaml)
he_slide_folder=$data_path/$(yq .local.main_slide_path params.yaml)/HE
ihc_slide_folder=$data_path/$(yq .local.main_slide_path params.yaml)/IHC
slide_folder=$data_path/$(yq .local.slide_path params.yaml)
outfolder=$data_path/$(yq .local.mask_path params.yaml)
mask_extension=$(yq .base.mask_extension params.yaml)
for file_path in $he_slide_folder/*.svs; do
  filename=$(basename $file_path)
  filestem=${filename%.*}
  outfile=$outfolder/$filestem$mask_extension
  if [ -f $outfile ]; then
    continue
  fi
  echo $filestem
  mkdir -p $slide_folder/HE
  mkdir -p $slide_folder/IHC
  ln -f $file_path $slide_folder/HE/$filename
  ln -f $ihc_slide_folder/$filename $slide_folder/IHC/$filename
  uv run dvc repro -f -s mask_extraction >> $HOME/mask_extraction_logs 2>&1 
  if [ ! $? -eq 0 ]; then
    uv run dvc commit -f mask_extraction
  fi
  rm -f $slide_folder/HE/$filename
  rm -f $slide_folder/IHC/$filename
done
