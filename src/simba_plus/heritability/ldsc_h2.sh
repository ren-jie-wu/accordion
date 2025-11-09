#!/bin/bash
annot_id=$1
sumstats_path=$2
output_prefix=$3
rerun=$4
# Running LDSC
filedir=/data/pinello/PROJECTS/2023_09_JF_SIMBAvariant/pre_ldsc_analysis/sldsc_analysis/ldsc_files
ldscdir=/data/pinello/PROJECTS/2022_12_LDSC/ldsc

if [ $# -gt 0 ]; then
    SUMSTATSFILE=$sumstats_path
fi
WEIGHTS=$filedir/GRCh38/weights/weights.hm3_noMHC.
FRQFILE=$filedir/FRQFILES/1000G.EUR.QC.
REFMODEL=$filedir/GRCh38/baselineLD_v2.2/baselineLD.,$annot_id.
#REFMODEL=$annot_id.
#REFMODEL=$filedir/GRCh38/baseline_v1.2/baseline.,$annot_id.
#REFMODEL=$workdir/$annot_id.
sumstat_basename=$(basename -s .gz $SUMSTATSFILE)
sumstat_basename=$(basename -s .sumstats $sumstat_basename)
OUTFILE=${output_prefix}$sumstat_basename

if [[ $rerun = "True" || ! -f $OUTFILE.results ]]; then
    python $ldscdir/ldsc.py \
        --h2 $SUMSTATSFILE \
        --w-ld-chr $WEIGHTS \
        --frqfile-chr $FRQFILE \
        --ref-ld-chr $REFMODEL \
        --overlap-annot \
        --thin-annot \
        --print-coefficients \
        --print-delete-vals \
        --out $OUTFILE
fi