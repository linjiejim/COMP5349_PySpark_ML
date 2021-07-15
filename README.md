# COMP5349_PySpark_ML
COMP5349 assignment. Performance comparison of PySpark on local machine and ERM Cluster. 


# Run on EMR cluster
```bash
# Select EMR
# Software: Pick emr-6.2.0 (spark, hadoop)
# Hardware: 1 Master & 4 Core nodes (m5.xlarge)
# Select ssh key
# Assign Security Group: Allow ssh (port 22)
# Wait for provisioning (5-6 mins)

# Inside the work directory
# Give execution priviledge
chmod +x submit-script.sh

# Push the file to hdfs
hdfs dfs -put tweets.json tweets.json
hdfs dfs -ls

# Execute the job
. submit-script.sh
```


# Run locally on Jupyter Notebook
```bash
# 1. Install
docker pull jupyter/pyspark-notebook


# 2. Create folders & files
.
├── conf
│   └── spark-defaults.conf
├── env.txt
├── logs
└── spark-history

# Inside: env.txt
SPARK_CONF_DIR=/home/jovyan/work/conf
SPARK_LOG_DIR=/home/jovyan/work/logs

# Inside: spark-defaults.conf
spark.eventLog.dir                      file:///home/jovyan/work/spark-history
spark.eventLog.enabled                  true
spark.history.fs.logDirectory           file:///home/jovyan/work/spark-history
spark.yarn.historyServer.address        http://localhost:18080


# 3. Run pyspark & history server
# - Mount current pwd into working directory
# - notebook port:  10000(loc) -> 8888(container)
# - history server port: 18080(loc) -> 18080(container)
# - specify config file
docker run --rm -p 10000:8888 \
    -p 18080:18080 \
    -v "$PWD":/home/jovyan/work \
    --env-file env.txt \
    --name comp5349 \
    jupyter/pyspark-notebook


# 4. Start the history server
docker exec comp5349 /usr/local/spark/sbin/start-history-server.sh


# 5. Access notebook 
http://127.0.0.1:10000/?token=<TOKEN_STRING>


# 6. Access history server
http://127.0.0.1:18080


# 7. Open ipynb
workload.ipynb
```
