package edu.boun.edgecloudsim.applications.uarc;

import edu.boun.edgecloudsim.applications.sample_app4.FCL_definition;
import edu.boun.edgecloudsim.cloud_server.CloudVM;
import edu.boun.edgecloudsim.core.SimManager;
import edu.boun.edgecloudsim.core.SimSettings;
import edu.boun.edgecloudsim.edge_client.CpuUtilizationModel_Custom;
import edu.boun.edgecloudsim.edge_client.Task;
import edu.boun.edgecloudsim.edge_orchestrator.EdgeOrchestrator;
import edu.boun.edgecloudsim.edge_server.EdgeHost;
import edu.boun.edgecloudsim.edge_server.EdgeVM;
import edu.boun.edgecloudsim.utils.SimLogger;
import net.sourceforge.jFuzzyLogic.FIS;
import org.antlr.runtime.RecognitionException;
import org.cloudbus.cloudsim.Host;
import org.cloudbus.cloudsim.UtilizationModelFull;
import org.cloudbus.cloudsim.Vm;
import org.cloudbus.cloudsim.core.CloudSim;
import org.cloudbus.cloudsim.core.SimEvent;

import java.util.List;

public class AdaptiveEdgeOrchestrator extends EdgeOrchestrator {
    public static final double MAX_DATA_SIZE=2500;

    private int numberOfHost; //used by load balancer
    private FIS fis1 = null;
    private FIS fis2 = null;
    private FIS fis3 = null;
    public AdaptiveEdgeOrchestrator(String orchestratorPolicy, String simScenario) {
        super(orchestratorPolicy, simScenario);
    }
    @Override
    public void initialize() {
        try {
            fis1 = FIS.createFromString(FCL_definition.fclDefinition1, false);
            fis2 = FIS.createFromString(FCL_definition.fclDefinition2, false);
            fis3 = FIS.createFromString(FCL_definition.fclDefinition3, false);
        } catch (RecognitionException e) {
            SimLogger.printLine("Cannot generate FIS! Terminating simulation...");
            e.printStackTrace();
            System.exit(0);
        }
    }
    /*
     * (non-Javadoc)
     * @see edu.boun.edgecloudsim.edge_orchestrator.EdgeOrchestrator#getDeviceToOffload(edu.boun.edgecloudsim.edge_client.Task)
     *
     * It is assumed that the edge orchestrator app is running on the edge devices in a distributed manner
     */
    @Override
    public int getDeviceToOffload(Task task) {
        int result = 0;

        //RODO: return proper host ID

        if(simScenario.equals("SINGLE_TIER")){
            result = SimSettings.GENERIC_EDGE_DEVICE_ID;
        }
        else if(simScenario.equals("TWO_TIER_WITH_EO")){
            int bestRemoteEdgeHostIndex = 0;
            int nearestEdgeHostIndex = 0;
            double nearestEdgeUtilization = 0;

            //dummy task to simulate a task with 1 Mbit file size to upload and download
            Task dummyTask = new Task(0, 0, 0, 0, 128, 128, new UtilizationModelFull(), new UtilizationModelFull(), new UtilizationModelFull());

            double wanDelay = SimManager.getInstance().getNetworkModel().getUploadDelay(task.getMobileDeviceId(),
                    SimSettings.CLOUD_DATACENTER_ID, dummyTask /* 1 Mbit */);
            double wanBW = (wanDelay == 0) ? 0 : (1 / wanDelay); /* Mbps */

            double manDelay = SimManager.getInstance().getNetworkModel().getUploadDelay(SimSettings.GENERIC_EDGE_DEVICE_ID,
                    SimSettings.GENERIC_EDGE_DEVICE_ID, dummyTask /* 1 Mbit */);

            double edgeUtilization = SimManager.getInstance().getEdgeServerManager().getAvgUtilization();

            //finding least loaded neighbor edge host
            double bestRemoteEdgeUtilization = 100; //start with max value
            for(int hostIndex=0; hostIndex<numberOfHost; hostIndex++){
                List<EdgeVM> vmArray = SimManager.getInstance().getEdgeServerManager().getVmList(hostIndex);

                double totalUtilization=0;
                for(int vmIndex=0; vmIndex<vmArray.size(); vmIndex++){
                    totalUtilization += vmArray.get(vmIndex).getCloudletScheduler().getTotalUtilizationOfCpu(CloudSim.clock());
                }

                double avgUtilization = (totalUtilization / (double)(vmArray.size()));

                EdgeHost host = (EdgeHost)(vmArray.get(0).getHost()); //all VMs have the same host
                if(host.getLocation().getServingWlanId() == task.getSubmittedLocation().getServingWlanId()){
                    nearestEdgeUtilization = totalUtilization / (double)(vmArray.size());
                    nearestEdgeHostIndex = hostIndex;
                }
                else if(avgUtilization < bestRemoteEdgeUtilization){
                    bestRemoteEdgeHostIndex = hostIndex;
                    bestRemoteEdgeUtilization = avgUtilization;
                }
            }

            switch (policy) {
                case "FUZZY_BASED" -> {
                    int bestHostIndex = nearestEdgeHostIndex;
                    double bestHostUtilization = nearestEdgeUtilization;

                    // Set inputs
                    fis2.setVariable("man_delay", manDelay);
                    fis2.setVariable("nearest_edge_uitl", nearestEdgeUtilization);
                    fis2.setVariable("best_remote_edge_uitl", bestRemoteEdgeUtilization);

                    // Evaluate
                    fis2.evaluate();
                    if (fis2.getVariable("offload_decision").getValue() > 50) {
                        bestHostIndex = bestRemoteEdgeHostIndex;
                        bestHostUtilization = bestRemoteEdgeUtilization;
                    }

                    double delay_sensitivity = SimSettings.getInstance().getTaskLookUpTable()[task.getTaskType()][12];

                    // Set inputs
                    fis1.setVariable("wan_bw", wanBW);
                    fis1.setVariable("task_size", task.getCloudletLength());
                    fis1.setVariable("delay_sensitivity", delay_sensitivity);
                    fis1.setVariable("avg_edge_util", bestHostUtilization);

                    // Evaluate
                    fis1.evaluate();
                    if (fis1.getVariable("offload_decision").getValue() > 50) {
                        result = SimSettings.CLOUD_DATACENTER_ID;
                    } else {
                        result = bestHostIndex;
                    }
                }
                case "FUZZY_COMPETITOR" -> {
                    double cpuSpeed = (double) 100 - edgeUtilization;
                    double videoExecution = SimSettings.getInstance().getTaskLookUpTable()[task.getTaskType()][12];
                    double dataSize = task.getCloudletFileSize() + task.getCloudletOutputSize();
                    double normalizedDataSize = Math.min(MAX_DATA_SIZE, dataSize) / MAX_DATA_SIZE;

                    // Set inputs
                    fis3.setVariable("wan_bw", wanBW);
                    fis3.setVariable("cpu_speed", cpuSpeed);
                    fis3.setVariable("video_execution", videoExecution);
                    fis3.setVariable("data_size", normalizedDataSize);

                    // Evaluate
                    fis3.evaluate();

                    if (fis3.getVariable("offload_decision").getValue() > 50)
                        result = SimSettings.CLOUD_DATACENTER_ID;
                    else
                        result = SimSettings.GENERIC_EDGE_DEVICE_ID;
                }
                case "NETWORK_BASED" -> {
                    if (wanBW > 6)
                        result = SimSettings.CLOUD_DATACENTER_ID;
                    else
                        result = SimSettings.GENERIC_EDGE_DEVICE_ID;
                }
                case "UTILIZATION_BASED" -> {
                    if (edgeUtilization > 80)
                        result = SimSettings.CLOUD_DATACENTER_ID;
                    else
                        result = SimSettings.GENERIC_EDGE_DEVICE_ID;
                }
                case "HYBRID" -> {
                    if (wanBW > 6 && edgeUtilization > 80)
                        result = SimSettings.CLOUD_DATACENTER_ID;
                    else
                        result = SimSettings.GENERIC_EDGE_DEVICE_ID;
                }
                default -> {
                    SimLogger.printLine("Unknown edge orchestrator policy! Terminating simulation...");
                    System.exit(0);
                }
            }
        }
        else {
            SimLogger.printLine("Unknown simulation scenario! Terminating simulation...");
            System.exit(0);
        }
        return result;
    }

    @Override
    public Vm getVmToOffload(Task task, int deviceId) {
        Vm selectedVM = null;

        if(deviceId == SimSettings.CLOUD_DATACENTER_ID){
            //Select VM on cloud devices via Least Loaded algorithm!
            double selectedVmCapacity = 0; //start with min value
            List<Host> list = SimManager.getInstance().getCloudServerManager().getDatacenter().getHostList();
            for (int hostIndex=0; hostIndex < list.size(); hostIndex++) {
                List<CloudVM> vmArray = SimManager.getInstance().getCloudServerManager().getVmList(hostIndex);
                for(int vmIndex=0; vmIndex<vmArray.size(); vmIndex++){
                    double requiredCapacity = ((CpuUtilizationModel_Custom)task.getUtilizationModelCpu()).predictUtilization(vmArray.get(vmIndex).getVmType());
                    double targetVmCapacity = (double)100 - vmArray.get(vmIndex).getCloudletScheduler().getTotalUtilizationOfCpu(CloudSim.clock());
                    if(requiredCapacity <= targetVmCapacity && targetVmCapacity > selectedVmCapacity){
                        selectedVM = vmArray.get(vmIndex);
                        selectedVmCapacity = targetVmCapacity;
                    }
                }
            }
        }
        else if(deviceId == SimSettings.GENERIC_EDGE_DEVICE_ID){
            //Select VM on edge devices via Least Loaded algorithm!
            double selectedVmCapacity = 0; //start with min value
            for(int hostIndex=0; hostIndex<numberOfHost; hostIndex++){
                List<EdgeVM> vmArray = SimManager.getInstance().getEdgeServerManager().getVmList(hostIndex);
                for(int vmIndex=0; vmIndex<vmArray.size(); vmIndex++){
                    double requiredCapacity = ((CpuUtilizationModel_Custom)task.getUtilizationModelCpu()).predictUtilization(vmArray.get(vmIndex).getVmType());
                    double targetVmCapacity = (double)100 - vmArray.get(vmIndex).getCloudletScheduler().getTotalUtilizationOfCpu(CloudSim.clock());
                    if(requiredCapacity <= targetVmCapacity && targetVmCapacity > selectedVmCapacity){
                        selectedVM = vmArray.get(vmIndex);
                        selectedVmCapacity = targetVmCapacity;
                    }
                }
            }
        }
        else{
            //if the host is specifically defined!
            List<EdgeVM> vmArray = SimManager.getInstance().getEdgeServerManager().getVmList(deviceId);

            //Select VM on edge devices via Least Loaded algorithm!
            double selectedVmCapacity = 0; //start with min value
            for(int vmIndex=0; vmIndex<vmArray.size(); vmIndex++){
                double requiredCapacity = ((CpuUtilizationModel_Custom)task.getUtilizationModelCpu()).predictUtilization(vmArray.get(vmIndex).getVmType());
                double targetVmCapacity = (double)100 - vmArray.get(vmIndex).getCloudletScheduler().getTotalUtilizationOfCpu(CloudSim.clock());
                if(requiredCapacity <= targetVmCapacity && targetVmCapacity > selectedVmCapacity){
                    selectedVM = vmArray.get(vmIndex);
                    selectedVmCapacity = targetVmCapacity;
                }
            }
        }
        return selectedVM;
    }

    @Override
    public void startEntity() {

    }

    @Override
    public void processEvent(SimEvent simEvent) {

    }

    @Override
    public void shutdownEntity() {

    }
}
