#!/bin/bash


export INDATADIR="/work/yw18581/data"
export OUTDIR="/work/yw18581/readmaroc"

export OUTDATADIR=$OUTDIR/test_5sigma_q
if [ ! -d $OUTDATADIR ]; then
    mkdir -p $OUTDATADIR;
fi;

export SUBSDIR=$OUTDIR/subs_5sigma
if [ ! -d $SUBSDIR ]; then
    mkdir -p $SUBSDIR;
fi;


#for i in $(seq 83442 83790);
for i in $(seq 133053 137647);
do
echo "#!/bin/bash" > $SUBSDIR/submitJob_$i.sh
#
echo "#PBS -j oe" >> $SUBSDIR/submitJob_$i.sh
echo "#PBS -l select=1:ncpus=1:mem=1G" >> $SUBSDIR/submitJob_$i.sh
echo "#PBS -l walltime=00:10:00">> $SUBSDIR/submitJob_$i.sh
# MODULES ------------------------------------------------------------

echo source /home/yw18581/.bash_profile>> $SUBSDIR/submitJob_$i.sh

#echo . /etc/profile.d/modules.sh>> $SUBSDIR/submitJob_$i.sh

#echo module add languages/gcc-5.3>> $SUBSDIR/submitJob_$i.sh

export RUNDIR="/work/yw18581/readmaroc"
echo export "INDATADIR"=$INDATADIR >> $SUBSDIR/submitJob_$i.sh
echo export "RUNDIR"=$RUNDIR >> $SUBSDIR/submitJob_$i.sh
#echo cd "/newhome/yw18581/chance/marocfindhits-new">> $SUBSDIR/submitJob_$i.sh
echo export "OUTDIR"=$OUTDIR >> $SUBSDIR/submitJob_$i.sh
echo export "OUTDATADIR"=$OUTDATADIR >> $SUBSDIR/submitJob_$i.sh

    echo     "python ${RUNDIR}/plot_and_count.py ${INDATADIR}/Run000$i.dat 5 3 ${OUTDATADIR}"  >> $SUBSDIR/submitJob_$i.sh


#echo done>> submitJob_$i.sh

chmod +x $SUBSDIR/submitJob_$i.sh
cd $SUBSDIR
qsub submitJob_$i.sh
cd $RUNDIR

done
