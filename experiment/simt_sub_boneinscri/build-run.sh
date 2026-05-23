mkdir -p build && cd build;   # 创建并进入build目录
cmake -DCMAKE_ASC_ARCHITECTURES=dav-3510 ..; make -j;   # 编译工程
./demo                        # 执行样例