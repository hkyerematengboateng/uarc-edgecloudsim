#!/bin/sh
rm -rf ../../bin
mkdir ../../bin
javac -classpath "../../lib/*.jar" -sourcepath ../../src ../../src/edu/boun/edgecloudsim/applications/uarc/Application.java -d ../../bin
