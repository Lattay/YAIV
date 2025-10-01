PREFIX=$(pwd)
TMP_DIR=$PREFIX/tmp

for DIR in "$TMP_DIR" "$PREFIX/results_ph"; do
    if test ! -d $DIR; then
        mkdir $DIR
    fi
done
rm -r results_ph/*
cd $PREFIX/results_ph

cat >$NAME.ph.pwi <<EOF

&INPUTPH
  prefix='$NAME',
  recover=.true.
  outdir='$TMP_DIR/',
  fildyn='$NAME.dyn',
  ldisp=.true.,
  tr2_ph=1e-5
  alpha_mix=0.5,
  verbosity='high'
  nq1=2, nq2=2, nq3=2,
 /
EOF

echo "running the phonons calculation"
mpiexec -np $NPROCS $QE_PATH/ph.x <$NAME.ph.pwi >$NAME.ph.pwo
rm input_tmp.in
echo "done"
