#!/usr/bin/env sh
set -e

cat neg_48.txt | head -n 300000 > label_list.txt
cat pos_48.txt | head -n 100000 >> label_list.txt
cat part_48.txt | head -n 100000 >> label_list.txt
cat landmark.txt | head -n 200000 >> label_list.txt
