TIMESTAMP=$(date '+%Y-%m-%d_%H-%M-%S')
LOGFILE="./save/logs/UniCon_LRL_CER_Dataset__${TIMESTAMP}.log"

python trainer_unicon_multi.py \
--batch_size 64 \
--model efficientnet \
--mode none \
--model_name UniCon_ETD_LRL_Multi \
--indir data/CER_ETD/preprocessed \
--method UniCon \
--n_classes 7 \
--epoch_start_classifier 0 \
--trial ETD_EFF_MULTI \
> "$LOGFILE" 2>&1

echo "✅ Log saved to $LOGFILE"