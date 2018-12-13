CUDA_VISIBLE_DEVICES=0 python main.py $PATH_TO_CIFAR10$ --dataset cifar10 --arch resnet110 --save_path ./110epoch500D8 --epochs 500 --schedule 250 375 --gammas 0.1 0.1 --learning_rate 0.1 --decay 0.0001 --batch_size 128 --Ddim 8

pre-trained model:
./pretrained_model
