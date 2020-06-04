#!/bin/sh
$JAWHOME/bin/hipoutils.sh -filter -o nn_run_$1 -s false -b "*::adc,*::tdc,RUN::*,nn::*" $1
$JAWHOME/bin/hipoutils.sh -filter -o rg_run_$1 -s false -b "*::adc,*::tdc,RUN::*" $1
