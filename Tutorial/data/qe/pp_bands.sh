PREFIX=$(pwd)
TMP_DIR=$PREFIX/tmp
PSEUDO_DIR=$PSEUDOS

for DIR in "$TMP_DIR" "$PREFIX/results_pp_bands"; do
    if test ! -d $DIR; then
        mkdir $DIR
    fi
done

rm -r results_pp_bands/*
cd $PREFIX/results_pp_bands

cat >$NAME.pp.bands.pwi <<EOF

&BANDS
  prefix='$NAME',
  outdir='$TMP_DIR',
  filband='$NAME.bands.dat'
 /
EOF

echo "processing the bands calculation"
mpiexec -np $NPROCS $QE_PATH/bands.x <$NAME.pp.bands.pwi >$NAME.pp.bands.pwo
rm input_tmp.in
echo "done"
