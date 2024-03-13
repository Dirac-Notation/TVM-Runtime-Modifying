#!/bin/bash

echo "Build Start"
cd ./tvm-unity/build
if cmake .. && cmake --build . --parallel $(nproc); then
    echo "Build Complete"

    cd ../..

    echo "=============== Git ==============="
    git add -u
    git commit -m "$(date +'%y-%m-%d')"
    echo "=============== Git ==============="

    echo
    python3 a.py
else
    cd ../..
    
    echo "Build Failed"
fi