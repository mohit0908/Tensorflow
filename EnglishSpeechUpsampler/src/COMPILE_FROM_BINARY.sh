CPP=g++
ARGS=""
OS=$(uname)
if [ "$OS" = "Darwin" ]; then
  CPP=clang++
  ARGS="-undefined dynamic_lookup"
fi


TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )
$CPP -std=c++11 -shared shuffle_op.cc -o shuffle_op.so -fPIC ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O2
