/home/lei/miniconda3/envs/CS/bin/python /mnt/md0/lei/projects/CellSegmentsation/CellSegmentation/minibatch/NEW_pre.py

echo "第一个脚本已完成运行"

/home/lei/miniconda3/envs/CS/bin/python /mnt/md0/lei/projects/CellSegmentsation/CellSegmentation/minibatch/NEW_trans.py

echo "第二个脚本已完成运行"



cd /mnt/md0/lei/projects/CellSegmentsation

git add .
git commit -m "auto commit $(date +"%Y%m%d_%H%M%S")"
git push

