#!/bin/bash
dir=$1
if [ -z "$dir" ]; then
    echo "Usage: $0 DIR_TO_GROUP"
    exit 1
fi

cd "$dir"
count=`ls | wc -l`
i=0
for i in *.jpg; do
    i=$((i+1))
    name=${i/.jpg/}
    class=`grep "^$name" ../labels.csv| cut -d, -f2`
    echo -ne "\rMoving $i/$count"
    mv $i $class; done
