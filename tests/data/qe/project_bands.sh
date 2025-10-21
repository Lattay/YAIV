PREFIX=$(pwd)
TMP_DIR=$PREFIX/tmp
PSEUDO_DIR=$PSEUDOS

for DIR in "$TMP_DIR" "$PREFIX/results_proj"; do
    if test ! -d $DIR; then
        mkdir $DIR
    fi
done

rm -r results_proj/*
cd $PREFIX/results_proj

cat >$NAME.proj.pwi <<EOF
&PROJWFC
  prefix='$NAME',
  outdir='$TMP_DIR',
  ngauss=0, degauss=0.001
  kresolveddos=.true.
  filpdos='pdos.dat'
  filproj='proj.dat'
/
EOF

echo "running the projection calculation"
mpiexec -np $NPROCS $QE_PATH/projwfc.x <$NAME.proj.pwi >$NAME.proj.pwo
rm pdos.data*
rm proj.dat*
rm input_tmp.in
echo "done"
