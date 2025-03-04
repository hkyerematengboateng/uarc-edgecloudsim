package edu.boun.edgecloudsim.applications.uarc;

import edu.boun.edgecloudsim.core.SimSettings;
import edu.boun.edgecloudsim.utils.SimLogger;
import edu.boun.edgecloudsim.utils.SimUtils;
import org.cloudbus.cloudsim.Log;

import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.Calendar;
import java.util.Date;

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
        SimSettings SS = SimSettings.getInstance();

        if(SS.initialize(configFile, edgeDevicesFile, applicationsFile) == false){
            SimLogger.printLine("cannot initialize simulation settings!");
            System.exit(0);
        }

        if(SS.getFileLoggingEnabled()){
            SimLogger.enableFileLog();
            SimUtils.cleanOutputFolder(outputFolder);
        }

        DateFormat df = new SimpleDateFormat("dd/MM/yyyy HH:mm:ss");
        Date SimulationStartDate = Calendar.getInstance().getTime();
        String now = df.format(SimulationStartDate);
        SimLogger.printLine("Simulation started at " + now);
        SimLogger.printLine("----------------------------------------------------------------------");

    }
}
