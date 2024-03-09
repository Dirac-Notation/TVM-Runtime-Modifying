#!/bin/bash

echo "Build Start"
if cmake ./tvm-unity && cmake --build ./tvm-unity/build --parallel $(nproc); then
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