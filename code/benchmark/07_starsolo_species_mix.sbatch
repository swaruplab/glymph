#!/bin/bash
#SBATCH --job-name=glymph_starsolo
#SBATCH --account=vswarup_lab
#SBATCH --partition=standard
#SBATCH --cpus-per-task=24
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# Unfiltered count matrix from reads, for species mixtures that have no public raw matrix.
#   sbatch --export=ALL,PLATFORM=rhapsody,NAME=<name>,PAIRS="<fastq prefix> [<prefix> ...]",OUT=<dir> 07_starsolo_species_mix.sbatch
# Each prefix names a pair <prefix>_1.fastq.gz and <prefix>_2.fastq.gz. Which mate holds the
# barcodes is read off the first reads (public deposits differ), and every file is checked
# against the archive's md5 in config/read_files.tsv before anything is aligned.
#
# PLATFORM=rhapsody  BD Rhapsody beads with the short linkers. Barcode read: 0 to 3 diversity bases, cell label 1
#                    (9), GTGA, cell label 2 (9), GACA, cell label 3 (9), UMI (8). Anchored on the
#                    two linkers, the settings of the RoCK/ROI authors (imallona/rock_roi_method).
# PLATFORM=parse     Parse Evercode v1/v2. Barcode read: UMI (10), round 3 barcode (8), 30-base
#                    linker, round 2 barcode (8), 22-base linker, round 1 barcode (8).
#                    Set PARSE_BC1 to the round 1 list of the kit (n24_v4 = WT Mini, n99_v5 = WT v2,
#                    n198_v5 = WT Mega). Round 1 polyT and random-hexamer barcodes of one well are
#                    merged at ingestion, as split-pipe does. cDNA reads may start with the
#                    template-switch oligo, so the minimum aligned share of a read is lowered.
# Output: $OUT/Solo.out/{Gene,GeneFull}/raw (every barcode) and /filtered (EmptyDrops_CR calls).
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
: "${PLATFORM:?}" "${NAME:?}" "${PAIRS:?}" "${OUT:?}"
DATA="${DATA:-/dfs7/swaruplab/vswarup/compute/glymph/data}"
case "$PLATFORM" in rhapsody) PATTERN='GTGA.........GACA' ;; parse) PATTERN='ATCCACGTGCTTGAGA' ;; *) echo "unknown PLATFORM $PLATFORM"; exit 2 ;; esac
CDNA=""; CBUMI=""
for pre in $PAIRS; do
  for m in 1 2; do
    f="${pre}_$m.fastq.gz"
    want=$(awk -F'\t' -v p="${f#"$DATA"/}" '$1 == p {print $3}' config/read_files.tsv)
    got=$(md5sum "$f" | cut -d' ' -f1)
    if [ -z "$want" ] || [ "$want" != "$got" ]; then echo "md5 mismatch for $f: archive ${want:-none}, file $got"; exit 3; fi
  done
  h1=$(zcat "${pre}_1.fastq.gz" | head -80000 | awk 'NR % 4 == 2' | grep -c -E "$PATTERN" || true)
  h2=$(zcat "${pre}_2.fastq.gz" | head -80000 | awk 'NR % 4 == 2' | grep -c -E "$PATTERN" || true)
  echo "$pre: barcode pattern in $h1 of 20000 first mates and $h2 of 20000 second mates"
  if [ "$h1" -ge 1000 ] && [ "$h1" -gt $((5 * h2)) ]; then bc=1; cd_=2
  elif [ "$h2" -ge 1000 ] && [ "$h2" -gt $((5 * h1)) ]; then bc=2; cd_=1
  else echo "cannot tell which mate holds the barcodes for $pre"; exit 4; fi
  CDNA="${CDNA:+$CDNA,}${pre}_${cd_}.fastq.gz"; CBUMI="${CBUMI:+$CBUMI,}${pre}_${bc}.fastq.gz"
done
echo "cDNA: $CDNA"; echo "barcodes: $CBUMI"
IDX="${IDX:-/dfs7/swaruplab/vswarup/compute/glymph/ref/star_GRCh38_mm10_2020A_2710a_sparse2}"
module load star/2.7.10a
mkdir -p "$OUT/whitelist"
case "$PLATFORM" in
  rhapsody)
    # BD_WL: 96x3 (97 labels per position) or 384x3. A run with the wrong list matches nothing
    # (GSE265962 with 384x3: 108 million reads, zero valid barcodes), so the share of valid
    # barcodes is checked below.
    cp config/bd_rhapsody_whitelist_${BD_WL:-96x3}/BD_CLS{1,2,3}.txt "$OUT/whitelist/"
    SOLO=(--soloType CB_UMI_Complex --soloAdapterSequence GTGANNNNNNNNNGACA
          --soloCBposition 2_-9_2_-1 2_4_2_12 2_17_2_25 --soloUMIposition 3_10_3_17
          --soloCBwhitelist "$OUT/whitelist/BD_CLS1.txt" "$OUT/whitelist/BD_CLS2.txt" "$OUT/whitelist/BD_CLS3.txt")
    ;;
  parse)
    BC=/dfs7/swaruplab/vswarup/bin/ParseBiosciences-Pipeline.1.2.1/splitpipe/barcodes
    for pair in "bc1:${PARSE_BC1:-n24_v4}" "bc2:${PARSE_BC2:-v1}" "bc3:${PARSE_BC3:-v1}"; do
      tail -n +2 "$BC/bc_data_${pair#*:}.csv" | cut -d, -f2 | sort -u > "$OUT/whitelist/${pair%%:*}.txt"
    done
    cp "$BC/bc_data_${PARSE_BC1:-n24_v4}.csv" "$OUT/whitelist/round1_wells.csv"
    SOLO=(--soloType CB_UMI_Complex --soloCBposition 0_10_0_17 0_48_0_55 0_78_0_85 --soloUMIposition 0_0_0_9
          --soloCBwhitelist "$OUT/whitelist/bc3.txt" "$OUT/whitelist/bc2.txt" "$OUT/whitelist/bc1.txt"
          --outFilterScoreMinOverLread 0.3 --outFilterMatchNminOverLread 0.3)
    ;;
  *) echo "unknown PLATFORM $PLATFORM"; exit 2 ;;
esac
bash code/benchmark/profile.sh "logs/starsolo_${NAME}" STAR --runThreadN "$SLURM_CPUS_PER_TASK" \
  --genomeDir "$IDX" --readFilesIn "$CDNA" "$CBUMI" --readFilesCommand zcat \
  "${SOLO[@]}" --soloCBmatchWLtype 1MM --soloBarcodeReadLength 0 \
  --soloFeatures Gene GeneFull --soloUMIdedup 1MM_All --soloCellFilter EmptyDrops_CR \
  --soloCellReadStats Standard --outSAMtype None --outFileNamePrefix "$OUT/"
cat "$OUT/Log.final.out" | grep -E "input reads|Uniquely mapped reads %|multiple loci %|too short %"
cat "$OUT/Solo.out/Barcodes.stats"
cat "$OUT/Solo.out/Gene/Summary.csv"
valid=$(awk -F, '$1 == "Reads With Valid Barcodes" {print $2}' "$OUT/Solo.out/Gene/Summary.csv")
awk -v v="$valid" 'BEGIN { exit !(v + 0 >= 0.3) }' || { echo "only $valid of reads carry a valid barcode: wrong barcode list or read layout"; exit 5; }
gzip -f "$OUT"/Solo.out/*/raw/matrix.mtx "$OUT"/Solo.out/*/raw/barcodes.tsv "$OUT"/Solo.out/*/raw/features.tsv \
        "$OUT"/Solo.out/*/filtered/matrix.mtx "$OUT"/Solo.out/*/filtered/barcodes.tsv "$OUT"/Solo.out/*/filtered/features.tsv
echo STARSOLO_DONE "$NAME"
