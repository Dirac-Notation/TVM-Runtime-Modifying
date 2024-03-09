환경
CMAKE >= 3.24
LLVM >= 15
CUDA >= 1.8
Git

빌드
git clone --recursive https://github.com/mlc-ai/relax.git tvm-unity && cd tvm-unity
rm -rf build && mkdir build && cd build
cp ../cmake/config.cmake .

echo "set(CMAKE_BUILD_TYPE RelWithDebInfo)" >> config.cmake

echo "set(USE_LLVM \"llvm-config --ignore-libllvm --link-static\")" >> config.cmake
echo "set(HIDE_PRIVATE_SYMBOLS ON)" >> config.cmake

echo "set(USE_CUDA   ON)" >> config.cmake
echo "set(USE_METAL  OFF)" >> config.cmake
echo "set(USE_VULKAN OFF)" >> config.cmake
echo "set(USE_OPENCL OFF)" >> config.cmake

cmake .. && cmake --build . --parallel $(nproc)

파이썬 연결
export PYTHONPATH=/path-to-tvm-unity/python:$PYTHONPATH