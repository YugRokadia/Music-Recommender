#!/usr/bin/env bash
# Last.fm 1K dataset (~640 MB). Hosted on Zenodo: https://zenodo.org/record/6090214
set -e
mkdir -p data && cd data
curl -L -o lastfm-dataset-1K.tar.gz "https://zenodo.org/records/6090214/files/lastfm-dataset-1K.tar.gz?download=1"
tar -xzf lastfm-dataset-1K.tar.gz
