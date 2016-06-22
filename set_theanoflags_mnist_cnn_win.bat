set THEANO_FLAGS=floatX=float32,device=cpu
python mnist_cnn.py
set THEANO_FLAGS=floatX=float32,device=cpu,blas.ldflags=-LC:\\openblas -lopenblas
python mnist_cnn.py
set THEANO_FLAGS=floatX=float32,device=gpu
python mnist_cnn.py
set THEANO_FLAGS=floatX=float32,device=gpu,optimizer_including=cudnn
python mnist_cnn.py
set THEANO_FLAGS=floatX=float32,device=gpu,optimizer_including=cudnn,lib.cnmem=0.8
python mnist_cnn.py
