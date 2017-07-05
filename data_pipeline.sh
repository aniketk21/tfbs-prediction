#!/bin/bash
# script to create a dataset that is compatible with the model's input

python create_bp_windows.py gm_chr22.dat 16050000 51234702 51244566 
python mod_chrom.py gm_chr22.bed gm_chr22.dat gm_chr22_mod_chrom.dat
python mod_motif.py gm_chr22_mod_chrom.dat gm_chr22.gff gm_chr22_mod_motif.dat
python mod_dnase_score.py gm_chr22.bedgraph gm_chr22_mod_motif.dat gm_chr22_mod_dnase.dat
python mod_peak.py gm_chr22_e2f1.tsv gm_chr22_mod_dnase.dat gm_chr22_mod_peak.dat
python dat2csv.py gm_chr22_mod_peak.dat
python uniq.py gm_chr22_mod_peak.dat.csv
