#!/bin/sh
rm -rf ../../bin
mkdir ../../bin
javac -classpath "../../lib/*:../../lib/commons-math3-3.6.1.jar:../../lib/jFuzzyLogic_v3.0.jar:../../lib/weka.jar:../../lib/colt.jar" -sourcepath ../../src ../../src/edu/boun/edgecloudsim/applications/uarc/Application.java -d ../../bin
