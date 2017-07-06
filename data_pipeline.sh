#!/bin/bash
# script to create a dataset that is compatible with the model's input

python create_bp_windows.py gm_chr22.dat 17639300 51021850 51022050  
python mod_chrom.py gm_chr22.bed gm_chr22.dat gm_chr22_mod_chrom.dat
python mod_motif.py gm_chr22_mod_chrom.dat chr1.gff gm_chr22_mod_motif.dat
python mod_dnase_score.py gm_chr22.bedgraph gm_chr22_mod_motif.dat gm_chr22_mod_dnase.dat
python mod_peak.py gm_chr22_e2f1.tsv gm_chr22_mod_dnase.dat gm_chr22_mod_peak.dat
python dat2csv.py gm_chr22_mod_peak.dat
python uniq.py gm_chr22_mod_peak.dat.csv
mv gm_chr22_mod_peak.dat.csv_uniq.csv gm_chr22_mod_peak_uniq.csv
