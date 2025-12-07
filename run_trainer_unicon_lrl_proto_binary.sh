TIMESTAMP=$(date '+%Y-%m-%d_%H-%M-%S')
LOGFILE="./save/logs/UniCon_LRL_CER_Dataset__${TIMESTAMP}.log"

python trainer_unicon_lrl_prototype_binary.py \
--batch_size 64 \
--model efficientnet \
--mode none \
--model_name UniCon_v2_ETD_LRL_Binary_2 \
--indir data/CER_ETD/prepared_etd_weekly_3 \
--method UniCon \
--n_classes 2 \
--epoch_start_classifier 0 \
--trial ETD_EFF_BINARY_PROTO \
> "$LOGFILE" 2>&1

echo "✅ Log saved to $LOGFILE"