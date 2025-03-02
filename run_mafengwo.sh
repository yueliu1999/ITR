# for Mafengwo dataset
python -u main.py --dataset=Mafengwo --predictor=MLP --loss_type=BPR --learning_rate=0.0001 --device=cuda:0 --num_negatives=8 --layers=3 --epoch=2000
