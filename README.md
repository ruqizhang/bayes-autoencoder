under construction - so far all implemented models have been tested for running, but not for accuracy

Example train_semisup.py call:
  python train_semisup.py --dataset MNIST --data_path /scratch/datasets/ --epochs 3 --model SSBAE --dir exps/test --optimizer SGHMC --optimizer_options lr=1e-7 --save-epochs 1

Example train_unsup.py call:
  python train_unsup.py --dataset MNIST --data_path /scratch/datasets/ --epochs 3 --model SSBAE --dir exps/test --optimizer SGHMC --optimizer_options lr=1e-7 --save-epochs 1



