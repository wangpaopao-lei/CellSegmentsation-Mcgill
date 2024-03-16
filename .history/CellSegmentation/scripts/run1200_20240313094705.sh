/home/lei/miniconda3/envs/CS/bin/python -u /mnt/md0/lei/projects/CellSegmentsation/CellSegmentation/src/1_preprocess.py -p 0 0 1200 1200

echo "preprocess done!"

/home/lei/miniconda3/envs/CS/bin/python -u /mnt/md0/lei/projects/CellSegmentsation/CellSegmentation/src/2_transformer.py

echo "transformer done!"

/home/lei/miniconda3/envs/CS/bin/python -u /mnt/md0/lei/projects/CellSegmentsation/CellSegmentation/minibatch/align.py

echo "align done!"

/home/lei/miniconda3/envs/CS/bin/python -u /mnt/md0/lei/projects/CellSegmentsation/CellSegmentation/src/4_fieldseg.py

echo "all done!"