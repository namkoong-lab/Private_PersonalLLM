#!/bin/sh

set -e # fail if any command fails

rm -rf ./arxiv-submission/

mkdir ./arxiv-submission

# Copy main tex file and other necessary files
cp -v \
    temp/arxiv/main.tex \
    temp/arxiv/sections/*.tex \
    arxiv-submission/

# Copy macros
mkdir -p arxiv-submission/macros
cp -v temp/arxiv/macros/*.sty arxiv-submission/macros/

# Copy figures
mkdir -p arxiv-submission/figures
cp -v temp/arxiv/figures/* arxiv-submission/figures/

# Copy bibliography file (if it exists)
if [ -f temp/arxiv/main.bbl ]; then
    cp -v temp/arxiv/main.bbl arxiv-submission/
fi

# Remove comments from tex files
sed -i 's/^%.*$//' arxiv-submission/*.tex
sed -i 's/^%.*$//' arxiv-submission/macros/*.sty

# Remove comments from tex files
# find arxiv-submission -name "*.tex" -type f -print0 | xargs -0 sed -i 's/^%.*$//'
# find arxiv-submission/macros -name "*.sty" -type f -print0 | xargs -0 sed -i 's/^%.*$//'

# Add arXiv compiler instruction at the end of main.tex
echo "\n\typeout{get arXiv to do 4 passes: Label(s) may have changed. Rerun}" >> arxiv-submission/main.tex

# Create tarball
cd arxiv-submission
tar -cvvf ../arxiv-submission.tar *
cd ..

# Cleanup
# rm -rf arxiv-submission