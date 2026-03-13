# Ascend相关环境变量
export ASCEND_HOME_PATH=/home/dndx/Ascend/ascend-toolkit/latest
export ASCEND_AICPU_PATH=$ASCEND_HOME_PATH
export ASCEND_OPP_PATH=$ASCEND_HOME_PATH/opp
export TOOLCHAIN_HOME=$ASCEND_HOME_PATH/toolkit
export ASDOPS_HOME_PATH=/usr/local/Ascend/nnal/atb/latest/atb/cxx_abi_1
export ATB_HOME_PATH=/usr/local/Ascend/nnal/atb/latest/atb/cxx_abi_1
export PYTHONNOUSERSITE=1

# 更新PATH，避免重复
export PATH=$ASCEND_HOME_PATH/bin:$ASCEND_HOME_PATH/compiler/ccec_compiler/bin:$ASCEND_HOME_PATH/tools/ccec_compiler/bin:/home/dndx/miniconda3/condabin:$PATH

# 更新LD_LIBRARY_PATH，去除重复路径
export LD_LIBRARY_PATH=$ASCEND_HOME_PATH/tools/aml/lib64:$ASCEND_HOME_PATH/lib64:$ASCEND_HOME_PATH/opp/built-in/op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64:$ASCEND_HOME_PATH/lib64/plugin/opskernel:$ASCEND_HOME_PATH/lib64/plugin/nnengine:/usr/local/Ascend/driver/lib64:$LD_LIBRARY_PATH

## 更新PYTHONPATH，去除重复路径
export PYTHONPATH=$ASCEND_HOME_PATH/python/site-packages:$ASCEND_HOME_PATH/opp/built-in/op_impl/ai_core/tbe

#source /usr/local/Ascend/ascend-toolkit/set_env.sh

