#!/bin/bash
# script to create a dataset that is compatible with the model's input

python create_bp_windows.py gm12878_chr17.dat 0 81195107 81195200
python mod_chrom.py gm12878_chr17.bed gm12878_chr17.dat gm12878_chr17_mod_chrom.dat
python mod_motif.py gm12878_chr17_mod_chrom.dat chr17.gff gm12878_chr17_mod_motif.dat
python mod_peak.py gm12878_chr17.narrowPeak gm12878_chr17_mod_motif.dat gm12878_chr17_mod_peak.dat
