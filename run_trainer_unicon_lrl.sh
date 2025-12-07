TIMESTAMP=$(date '+%Y-%m-%d_%H-%M-%S')
LOGFILE="./save/logs/UniCon_LRL_ETD__${TIMESTAMP}.log"

python trainer_unicon_prototype.py \
--batch_size 64 \
--model efficientnet \
--mode none \
--model_name UniCon_ETD_LRL \
--indir data/ksm_transformer_best_result \
--method UniCon \
--epoch_start_classifier 0 \
--trial ETD_EFF_PROTOTYPE \
> "$LOGFILE" 2>&1

echo "✅ Log saved to $LOGFILE"
