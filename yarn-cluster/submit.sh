#!/bin/bash -x

num_nodes=10
num_thread=1
input="hdfs://nn1/data/to"

cd `dirname $0`

hadoop fs -rmr -skipTrash temp

export DMLC_CPU_VCORES=1
export DMLC_MEMORY_MB=4096
export DMLC_NUM_WORKER=20
export DMLC_NUM_SERVER=1
export DMLC_HDFS_OPTS="--Xmx128m"

java -cp `hadoop classpath`:dmlc-yarn.jar \
  org.apache.hadoop.yarn.dmlc.Client \
  -file admm \
  -file libhdfs.so.0.0.0 \
  -file libjvm.so \
  -file run_hdfs_prog.py \
  -file dmlc-yarn.jar \
  -jobname admm_lr_training \
  -tempdir ns1/user/baigang/temp \
  -queue root.megatron.baigang \
  ./run_hdfs_prog.py ./admm --help 

