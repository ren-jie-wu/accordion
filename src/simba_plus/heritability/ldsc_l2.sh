#!/bin/bash
annot_id=$1
output_prefix=$2
rerun=$3
# Running LDSC
filedir=/data/pinello/PROJECTS/2023_09_JF_SIMBAvariant/pre_ldsc_analysis/sldsc_analysis/ldsc_files
pwd -P
ldscdir=/data/pinello/PROJECTS/2022_12_LDSC/ldsc
BFILE=$filedir/GRCh38/plink_files/1000G.EUR.hg38
SNPLIST=$filedir/hm3_no_MHC.list.txt
ANNOTFILE=$annot_id.
OUTFILE=$annot_id

cd $workdir
for CHR in $(seq 1 22); do
    if [[ $rerun = "True" ]] || [[ ! -f $output_prefix.$CHR.l2.ldscore.gz ]] ; then
        $ldscdir/ldsc.py \
        --l2 \
        --bfile $BFILE.$CHR \
        --ld-wind-cm 1 \
        --print-snps $SNPLIST \
        --annot ${ANNOTFILE}$CHR.annot.npz \
        --thin-annot \
        --chunk-size 10000 --no-print-condnum \
        --out $output_prefix.$CHR 
    fi
done