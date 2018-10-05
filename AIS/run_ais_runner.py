import subprocess,time, os
import argparse

parser = argparse.ArgumentParser(description='AIS with bayesian auto-encoder for text data only')
parser.add_argument('--file', type=str, default ='', help = 'file to run test loading on')
parser.add_argument('--model', type=str, default='BAE', help = 'class of model to load')
parser.add_argument('--num-steps', type=int, default = 500, help = 'number of steps to run AIS for')
args = parser.parse_args()

all_commands = []
for i, seed in enumerate(range(14, 16)):
    time.sleep(5)
    print('Now running seed ', seed, ' with ', args.num_steps, ' steps')
    command_str = 'export CUDA_VISIBLE_DEVICES='+str(i)+'; python run_ais.py' + \
                                ' --dataset ptb --data_path /scratch/datasets/ptb' + \
                                ' --num-steps ' + str(args.num_steps) + ' --num-samples 1 ' + \
                                ' --seed ' + str(seed) + \
                                ' --file ' + args.file + ' --model ' + args.model + '> ' + \
                                args.model+'_output_'+str(seed)+'.out'
    command = subprocess.Popen(command_str,stdout=subprocess.PIPE, shell=True)

#now wait 30 minutes
time.sleep(60 * 30)

all_outuput = [command.stdout for command in all_commands]
print(all_outuput)
