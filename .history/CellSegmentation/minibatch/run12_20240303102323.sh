./run.sh > log/output_$(date +"%Y%m%d_%H%M%S").log 2>&1 &

cd /mnt/md0/lei/projects/CellSegmentsation

git add .
git commit -m "auto commit $(date +"%Y%m%d_%H%M%S")"
git push
