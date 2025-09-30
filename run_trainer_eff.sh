TIMESTAMP=$(date '+%Y-%m-%d_%H-%M-%S')
LOGFILE="./save/logs/Efficient_net_ETD__${TIMESTAMP}.log"

python trainer_eff.py \
--batch_size 64 \
--model efficientnet \
--mode none \
--model_name UniCon_ETD \
--method UniCon \
--epoch_start_classifier 0 \
--trial ETD_EFF \
> "$LOGFILE" 2>&1

echo "✅ Log saved to $LOGFILE"
