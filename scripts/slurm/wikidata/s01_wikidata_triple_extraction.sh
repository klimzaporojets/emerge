#!/bin/bash
#SBATCH --job-name=wd01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=48:00:00
#SBATCH --mem=32G
#SBATCH --partition=rome
#SBATCH --output=logs/s01_wdata_triple_extraction_%A_%a.out

INPUT_7Z_DIR=$1
OUTPUT_TRIPLES_DIR=$2
OUTPUT_ENTITIES_DIR=$3
NR_THREADS=${4:-32}
LOG_FILE=${5:-logs/s01_wdata_triple_extraction.log}

cd src/dataset/wikidata
mvn clean compile assembly:single

srun java -cp target/wikidata-1.0-SNAPSHOT-jar-with-dependencies.jar \
    wikidata.MainConcurrentWikidata \
    $INPUT_7Z_DIR \
    $OUTPUT_TRIPLES_DIR \
    $OUTPUT_ENTITIES_DIR \
    $NR_THREADS 2>&1 | tee $LOG_FILE
