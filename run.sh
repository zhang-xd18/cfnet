python ./main.py \
  --cpu \
  --evaluate \
  --name 'CRNet' \
  --data-dir '/home/3GPP/' \
  --batch-size 100 \
  --workers 0 \
  --cr 4 \
  --scenarios 'CDA' \
  --pretrained './checkpoint/CDA_MT_4.pth' \
  2>&1 | tee log.out