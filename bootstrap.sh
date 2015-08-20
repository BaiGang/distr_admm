#!/bin/bash -x -e

git submodule init
git submodule update

PROJECT_PATH=`pwd`/`dirname $0`
ROOT_PATH=$PROJECT_PATH/third_party/root

cd $PROJECT_PATH/third_party/liblbfgs
./autogen.sh && ./configure --prefix=$ROOT_PATH --disable-shared --enable-static --enable-sse2
make && make install

cd $PROJECT_PATH/third_party/rabit
make all && cp -r include/* $ROOT_PATH/include \
  && rm $ROOT_PATH/include/README.md \
  && rm $ROOT_PATH/include/dmlc/README.md \
  && cp -r lib/lib*.a $ROOT_PATH/lib

cd $PROJECT_PATH/third_party/dmlc-core
make all USE_HDFS=1 \
         HDFS_INC_PATH=$PROJECT_PATH/include \
         HDFS_LIB_PATH=$PROJECT_PATH/lib \
  && cp -r include/dmlc $ROOT_PATH/include \
  && cp libdmlc.a $ROOT_PATH/lib

