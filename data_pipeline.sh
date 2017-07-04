#!/bin/bash
# script to create a dataset that is compatible with the model's input

python create_bp_windows.py k_chr8.dat 600 146363400 146363600 
python mod_chrom.py k_chr8.bed k_chr8.dat k_chr8_mod_chrom.dat
python mod_dnase_score.py k_chr8.bedgraph k_chr8_mod_chrom.dat k_chr8_mod_dnase.dat
python mod_motif.py k_chr8_mod_dnase.dat chr8.gff k_chr8_mod_motif.dat
python mod_peak.py k_chr8.tsv k_chr8_mod_motif.dat k_chr8_mod_peak_no_dnase.dat
