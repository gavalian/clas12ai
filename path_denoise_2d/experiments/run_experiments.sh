#!/bin/bash

if [ ! -f ml-cli.py ]; then
    ln -s ../ml-cli.py
    echo "Created symbolic link to mc-cli.py..."
fi

echo "Downloading and preparing datasets..."
./prepare_datasets.sh
echo "Done!"

declare -a models
models=( 0 0a 0b 0c 0d 0e 0f 0g 1 2 )

declare -a na
na=( 5 45 50 55 90 100 110 )

declare -a thresholds
thresholds=( 0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40 0.45 0.5 )

run_model_studies() {
 #   if [ ! -d models_studies ]; then
 #       mkdir models_studies
 #   fi
    cd models_studies
    for m in ${models[@]};
    do
        if [ ! -d $m ]; then
            mkdir $m
        fi
        if [ ! -f ${m}/train/cnn_autoenc_full.h5 ]; then
            echo "Training model ${m}..."
            python3 ../ml-cli.py train -t ../datasets/train_set.lsvm -v ../datasets/test_set.lsvm -r ${m}/train  -n $m
        fi

        # if [ ! -d ${m}/test ]; then
        echo "Testing model ${m}..."
        python3 ../ml-cli.py test -v ../datasets/test_set.lsvm -m ${m}/train/cnn_autoenc_full.h5 -r ${m}/test
        # fi
    done 
    cd ..
}

run_luminosity_studies() {

#    if [ ! -d luminosity_studies ]; then
#        mkdir luminosity_studies
#    fi


    cd luminosity_studies

    for n in ${na[@]};
    do
        if [ ! -d $n ]; then
            mkdir $n
        fi
        echo "Testing on ${n}nA luminosity..."
        python3 ../ml-cli.py test -v ../datasets/luminosity/luminosity_${n}nA.lsvm -m ../models_studies/0b/train/cnn_autoenc_full.h5 -r ${n}

    done 

    cd ..
}

run_threshold_studies() {

#    if [ ! -d threshold_studies ]; then
#        mkdir threshold_studies
#    fi


    cd threshold_studies
    for n in ${na[@]};
    do
        if [ ! -d $n ]; then
            mkdir $n
        fi

        for t in ${thresholds[@]};
        do
            if [ ! -d ${n}/$t ]; then
                mkdir ${n}/$t
            fi

            echo "Testing with threshold ${t} on ${n}nA luminosity..."
            python3 ../ml-cli.py test -v ../datasets/luminosity/luminosity_${n}nA.lsvm -m ../models_studies/0b/train/cnn_autoenc_full.h5 --threshold ${t} -r ${n}/${t}/

        done
    done
    cd ..
}

run_model_studies
run_luminosity_studies
run_threshold_studies


echo "Collecting Results for Model Studies"
cd models_studies
python3 collect_all_results.py
cd ..

echo "Collecting Results for Luminosity Studies"
cd luminosity_studies
python3 collect_all_results.py
python3 plot_segments.py
cd ..

echo "Collecting Results for Threshold Studies"
cd threshold_studies
python3 collect_all_results.py
cd ..
