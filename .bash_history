sbatch network.sh 
nano network.sh 
sbatch network.sh 
python 20200921_clean_code.py -v -n clean_code -t 20000 -b 8 -l 70000 -ss
cd exps/sparks/
python 20200921_clean_code.py -v -n clean_code -t 20000 -b 8 -l 70000 -ss
python 20200921_clean_code.py -v -n clean_code -t 20000 -b 8 -l 70000
python 20200921_clean_code.py -v -n clean_code -t 20000 -b 8 -l 65000 -ss
python 20200921_clean_code.py -v -n clean_code -t 20000 -b 8 -l 70000 -ss
cd exps/sparks/
python 20200921_clean_code.py -v -n clean_code -t 20000 -b 8 -l 70000 -ss
squeue -u pd19x428
sbatch network.sh 
srun --mem-per-cpu=60G --cpus-per-task=1 --partition=gpu --gres=gpu:gtx1080ti:1 --time=01:00:00
srun --mem-per-cpu=60G --cpus-per-task=1 --partition=gpu --gres=gpu:gtx1080ti:1 --time=01:00:00 --pty bash
srun --mem-per-cpu=60G --cpus-per-task=1 --partition=gpu --gres=gpu:gtx1080ti:2 --time=01:00:00 --pty bash
sbatch network.sh 
squeue -u pd19x428
nano network.sh 
sbatch network.sh 
squeue -u pd19x428
nvidia-smi
cd exps/sparks/
python 20201007_new_masks_spark_centres.py -v 
srun --mem-per-cpu=60G --cpus-per-task=1 --partition=gpu --gres=gpu:gtx1080ti:2 --time=01:00:00 --pty bash
srun --mem-per-cpu=60G --cpus-per-task=1 --partition=gpu --gres=gpu:gtx1080ti:1 --time=01:00:00 --pty bash
cd exps/sparks/
python 20201007_new_masks_spark_centres.py -v 
sbatch network.sh 
squeue -u pd19x428
nvidia-smi
scancel 35843332
nvidia-smi
squeue -u pd19x428
sbatch network.sh 
squeue -u pd19x428
srun --mem-per-cpu=60G --cpus-per-task=1 --partition=gpu --gres=gpu:gtx1080ti:1 --time=01:00:00 --pty bash
srun --mem-per-cpu=60G --cpus-per-task=1 --partition=gpu --gres=gpu:gtx1080ti:1 --time=00:30:00 --pty bash
srun --mem-per-cpu=60G --cpus-per-task=1 --partition=gpu --gres=gpu:gtx1080ti:1 --time=01:00:00 --pty bash
cd exps/sparks/
python 20201007_new_masks_spark_centres.py -v -ss
sbatch network.sh 
sinfo
sbatch network2.sh 
cd exps/sparks/
python 20201013_new_masks_spark_centres_new_data.py -v
python 20201013_new_masks_spark_centres_new_data.py -v -b 4
srun --mem-per-cpu=60G --cpus-per-task=1 --partition=gpu --gres=gpu:gtx1080ti:2 --time=01:00:00 --pty bash
sbatch network2.sh 
squeue -u pd19x428
sbatch network.sh 
squeue -u pd19x428
sbatch network2.sh 
sbatch network.sh 
sbatch network2.sh 
sbatch network.sh 
sbatch network2.sh 
squeue -u pd19x428
scancel 36108366
squeue -u pd19x428
sbatch network.sh 
squeue -u pd19x428
cd exps/sparks/
python save_single_chunk_predictions.py 
srun --mem-per-cpu=60G --cpus-per-task=1 --partition=gpu --gres=gpu:gtx1080ti:2 --time=01:00:00 --pty bash
squeue -u pd19x428
scancel 36273593
squeue -u pd19x428
scancel 36273596
squeue -u pd19x428
cd
sbatch save_chunks.sh 
sbatch network.sh 
squeue -u pd19x428
scancel 36274884
sbatch save_chunks.sh 
scancel 36274884
squeue -u pd19x428
scancel 36274875
squeue -u pd19x428
srun --mem-per-cpu=60G --cpus-per-task=1 --partition=gpu --gres=gpu:gtx1080ti:2 --time=01:00:00 --pty bash
sbatch network.sh 
srun --mem-per-cpu=60G --cpus-per-task=1 --partition=gpu --gres=gpu:gtx1080ti:2 --time=02:00:00 --pty bash
pwd
cd 
pwd
cd ..
ls
cd shared/
ls
pwd
o+r -R forTill/data
chmod o+r -R forTill/data
chmod o+r -R forTill/dataded
