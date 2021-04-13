#!/bin/sh
#****************************************************************
# SCRIPT : run prediction class from distribution.
#****************************************************************
SCRIPTDIR=`dirname $0`
java -cp "$SCRIPTDIR/target/denoise2d-1.0-jar-with-dependencies.jar" org.crtcjlab.denoise2d.Denoise2d $*
