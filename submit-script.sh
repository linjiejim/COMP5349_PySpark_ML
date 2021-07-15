spark-submit \
    --master yarn \
    --deploy-mode client \
    --num-executors 3 \
    workload.py