#!/bin/bash


export INDATADIR="/user/work/yw18581/data_1110"
export OUTDIR="/user/work/yw18581/output"

export OUTDATADIR=$OUTDIR/test_4sigma_1110_q
if [ ! -d $OUTDATADIR ]; then
    mkdir -p $OUTDATADIR;
fi;

export SUBSDIR=$OUTDIR/subs_4sigma_1110
if [ ! -d $SUBSDIR ]; then
    mkdir -p $SUBSDIR;
fi;


#for i in $(seq 83442 83790);
for i in $(seq 137649 142654);
do
echo "#!/bin/bash" > $SUBSDIR/submitJob_$i.sh
#
echo "#SBATCH --output=$SUBSDIR/submitJob_${i}_o" >> $SUBSDIR/submitJob_$i.sh
echo "#SBATCH --nodes=1 --ntasks-per-node=1 --mem=1G" >> $SUBSDIR/submitJob_$i.sh
echo "#SBATCH --time=0-00:10:00">> $SUBSDIR/submitJob_$i.sh
# MODULES ------------------------------------------------------------

echo source /user/home/yw18581/.bash_profile>> $SUBSDIR/submitJob_$i.sh

#echo . /etc/profile.d/modules.sh>> $SUBSDIR/submitJob_$i.sh

#echo module add languages/gcc-5.3>> $SUBSDIR/submitJob_$i.sh

export RUNDIR="/user/work/yw18581/readmaroc"
echo export "INDATADIR"=$INDATADIR >> $SUBSDIR/submitJob_$i.sh
echo export "RUNDIR"=$RUNDIR >> $SUBSDIR/submitJob_$i.sh
#echo cd "/newhome/yw18581/chance/marocfindhits-new">> $SUBSDIR/submitJob_$i.sh
echo export "OUTDIR"=$OUTDIR >> $SUBSDIR/submitJob_$i.sh
echo export "OUTDATADIR"=$OUTDATADIR >> $SUBSDIR/submitJob_$i.sh

    echo     "python ${RUNDIR}/plot_and_count.py ${INDATADIR}/Run000$i.dat 4 3 ${OUTDATADIR} True"  >> $SUBSDIR/submitJob_$i.sh


#echo done>> submitJob_$i.sh

chmod +x $SUBSDIR/submitJob_$i.sh
cd $SUBSDIR
sbatch submitJob_$i.sh
cd $RUNDIR

done
