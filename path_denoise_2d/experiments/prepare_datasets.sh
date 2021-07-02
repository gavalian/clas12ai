#!/bin/bash

declare -a train_sets
path="https://userweb.jlab.org/~gavalian/ML/2021/Denoise/"
path_luminosity="https://userweb.jlab.org/~gavalian/ML/2021/Denoise/luminocity_fixed/"
train_sets=( "dc_denoise_one_track_1.lsvm" "dc_denoise_two_track_1.lsvm" )
test_sets=( "dc_denoise_one_track_2.lsvm" "dc_denoise_two_track_2.lsvm" )


declare -a na
na=( 5 45 50 55 90 100 110 )
if [ ! -d datasets ]; then
    mkdir datasets
fi

cd datasets

echo -n "" > train_set.lsvm
for t in ${train_sets[@]};
do 
    if [ ! -f $t ]; then
        wget ${path}$t
    fi

    cat $t >> train_set.lsvm
done

echo -n "" > test_set.lsvm

for t in ${test_sets[@]};
do 
    if [ ! -f $t ]; then
        wget ${path}$t
    fi
    cat $t >> test_set.lsvm
done

# cd ..
if [ ! -d luminosity ]; then
    mkdir luminosity
fi

cd luminosity

for n in ${na[@]};
do
    echo -n "" > luminosity_${n}nA.lsvm
    
    na_prefix1="${path_luminosity}/dc_denoise_one_track_fixed_"${n}"nA.lsvm"
    na_prefix2="${path_luminosity}/dc_denoise_two_track_fixed_"${n}"nA.lsvm"
    if [ ! -f "dc_denoise_one_track_fixed_"${n}"nA.lsvm" ]; then
        wget $na_prefix1
    fi
    if [ ! -f "dc_denoise_two_track_fixed_"${n}"nA.lsvm" ]; then
        wget $na_prefix2
    fi
    
    if [ ! -d fixed_data ]; then
        mkdir fixed_data
    fi
        if [ $n -ne 5 ]; then
            cat dc_denoise_one_track_fixed_"${n}"nA.lsvm >> luminosity_${n}nA.lsvm
            cat dc_denoise_two_track_fixed_"${n}"nA.lsvm >> luminosity_${n}nA.lsvm
        fi
    #fi
done

cd ..
cd ..
