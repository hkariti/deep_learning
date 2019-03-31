#!/bin/bash

if [ "$1" = -h -o "$1" = '--help' ]; then
    echo "Usage: $0 [FILE]"
    echo Parse loss output and output a csv
fi

file=$1

cat $file | sed -n '/train/ h; /val/ { H; x; s/\n/ /; p; }' | awk 'BEGIN { OFS=","; print "epoch", "train_loss", "validation_loss" } {print NR-1, $3, $11}'
