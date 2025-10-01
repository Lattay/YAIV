PREFIX=$(pwd)
TMP_DIR=$PREFIX/tmp
PSEUDO_DIR=$PSEUDOS

for DIR in "$TMP_DIR" "$PREFIX/results_bands"; do
    if test ! -d $DIR; then
        mkdir $DIR
    fi
done

rm -r results_bands/*
cd $PREFIX/results_bands

cat >$NAME.bands.pwi <<EOF
&CONTROL
  calculation='bands'
  restart_mode='from_scratch',
  prefix='$NAME',
  pseudo_dir = '$PSEUDO_DIR',
  outdir='$TMP_DIR',
  verbosity='high'
 /
&SYSTEM
  noncolin=.true.
  lspinorb=.true.
  ibrav=0,
  nat=$ATM_NUM,
  ntyp=$ATM_TYPES,
  ecutwfc=$CUTOFF,
  ecutrho=$ECUTRHO,
  occupations='smearing',
  smearing='mp',
  degauss=$SMEAR,
 /
&ELECTRONS
  conv_thr =  1d-5
  mixing_beta = 0.7
 /
ATOMIC_SPECIES
$ATOMIC_SPECIES
ATOMIC_POSITIONS {crystal}
$ATOMIC_CRYST_POSITIONS
K_POINTS { crystal_b }
 $QE_CRYST_PATH
CELL_PARAMETERS {angstrom}
$LATTICE
EOF

echo "running the bands calculation"
mpiexec -np $NPROCS $QE_PATH/pw.x <$NAME.bands.pwi >$NAME.bands.pwo
cp ../tmp/$NAME.save/data-file-schema.xml ./bands.xml
echo "done"
