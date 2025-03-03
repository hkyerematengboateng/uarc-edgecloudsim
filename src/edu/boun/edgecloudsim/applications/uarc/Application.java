package edu.boun.edgecloudsim.applications.uarc;

import edu.boun.edgecloudsim.utils.SimLogger;
import org.cloudbus.cloudsim.Log;

/**
 *
 */
public class Application {

    public static void main(String[] args){
        Log.disable();

        SimLogger.enablePrintLog();
        int iterationNumber = 1;
        String configFile = "";
        String outputFolder = "";
        String edgeDevicesFile = "";
        String applicationsFile = "";
        if (args.length == 5){
            configFile = args[0];
            edgeDevicesFile = args[1];
            applicationsFile = args[2];
            outputFolder = args[3];
            iterationNumber = Integer.parseInt(args[4]);
        }
        else{
            SimLogger.printLine("Simulation setting file, output folder and iteration number are not provided! Using default ones...");
            configFile = "scripts/sample_app4/config/default_config.properties";
            applicationsFile = "scripts/sample_app4/config/applications.xml";
            edgeDevicesFile = "scripts/sample_app4/config/edge_devices.xml";
            outputFolder = "sim_results/ite" + iterationNumber;
        }
    }
}
