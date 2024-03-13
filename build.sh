#!/bin/bash

echo "Build Start"
cd ./tvm-unity/build
if cmake .. && cmake --build . --parallel $(nproc); then
    echo "Build Complete"

    echo "=============== Git ==============="
    git add -u
    git commit -m "$(date +'%y-%m-%d')"
    echo "=============== Git ==============="

    echo
    python3 a.py
else
    echo "Build Failed"
fi
cd ../..