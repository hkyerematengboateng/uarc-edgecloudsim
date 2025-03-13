package edu.boun.edgecloudsim.applications.uarc;

import edu.boun.edgecloudsim.applications.sample_app4.FCL_definition;
import edu.boun.edgecloudsim.cloud_server.CloudVM;
import edu.boun.edgecloudsim.core.SimManager;
import edu.boun.edgecloudsim.core.SimSettings;
import edu.boun.edgecloudsim.edge_client.CpuUtilizationModel_Custom;
import edu.boun.edgecloudsim.edge_client.Task;
import edu.boun.edgecloudsim.edge_client.mobile_processing_unit.MobileVM;
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

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;

import static org.cloudbus.cloudsim.core.CloudSim.getEntityId;

public class AdaptiveMLEdgeOrchestrator extends EdgeOrchestrator {
    private static final int BASE = 100000;
    // Constants
    public static final double MAX_DATA_SIZE = 2500;
    public static final int METRIC_UPDATE_INTERVAL = 1; // seconds
    public static final int LEARNING_INTERVAL = 30; // seconds
    public static final int MODEL_UPDATE_INTERVAL = 300; // seconds
    public static final int REWARD_LATENCY_WEIGHT = 70; // percentage weight for latency in reward calculation
    public static final int REWARD_ENERGY_WEIGHT = 15; // percentage weight for energy in reward calculation
    public static final int REWARD_RELIABILITY_WEIGHT = 15; // percentage weight for reliability in reward calculation

    public static final int TASK_COMPLETED = BASE+5;
    public static final int TASK_FAILED = BASE+6;
    // Metrics
    private double activeManTaskCount = 0;
    private double activeWanTaskCount = 0;
    private double totalSizeOfActiveManTasks = 0;
    private int numberOfHost; // used by load balancer

    // Real-time monitoring
    private Map<Integer, DeviceStats> deviceStats = new ConcurrentHashMap<>();
    private Map<Integer, List<TaskExecution>> taskHistory = new ConcurrentHashMap<>();
    private Map<String, Double> featureImportance = new HashMap<>();

    // ML model parameters
    private Map<String, Map<String, Double>> qTable = new HashMap<>(); // Q-learning table
    private double learningRate = 0.1;
    private double discountFactor = 0.9;
    private double explorationRate = 0.2;
    private Map<Integer, List<Double>> lastNTaskCompletionTimes = new HashMap<>(); // Device -> completion times
    private Map<Integer, Double> deviceReliability = new HashMap<>();
    private List<PredictionModel> predictionModels = new ArrayList<>();

    // Feature extraction
    private Map<Integer, TaskFeatures> taskFeatureStore = new HashMap<>();
    public AdaptiveMLEdgeOrchestrator(String orchestratorPolicy, String simScenario) {
        super(orchestratorPolicy, simScenario);
    }
    @Override
    public void initialize() {
        // Initialize device statistics
        initializeDeviceStats();

        // Initialize Q-learning parameters
        initializeQTable();

        // Initialize feature importance
        initializeFeatureImportance();

        // Schedule periodic events for real-time adaptation
//        scheduleEvents();

        // Initialize prediction models
        initializePredictionModels();

        SimLogger.printLine("AdaptiveMLEdgeOrchestrator initialized with ML-based orchestration");
    }
    /**
     * Initialize Q-learning table
     */
    private void initializeQTable() {
        // States: combination of task type and network conditions
        List<String> states = new ArrayList<>();

        // For each task type
        for (int taskType = 0; taskType < SimSettings.getInstance().getTaskLookUpTable().length; taskType++) {
            // For different network conditions (low, medium, high bandwidth)
            for (String networkCondition : Arrays.asList("low_bw", "medium_bw", "high_bw")) {
                // For different load conditions (low, medium, high)
                for (String loadCondition : Arrays.asList("low_load", "medium_load", "high_load")) {
                    states.add("task_" + taskType + "_" + networkCondition + "_" + loadCondition);
                }
            }
        }

        // Actions: offload to cloud, edge, or mobile
        List<String> actions = Arrays.asList("cloud", "edge", "mobile");

        // Initialize Q-table with zeros
        for (String state : states) {
            Map<String, Double> actionValues = new HashMap<>();
            for (String action : actions) {
                actionValues.put(action, 0.0);
            }
            qTable.put(state, actionValues);
        }
    }
    private void initializeFeatureImportance() {
        featureImportance.put("taskType", 0.1);
        featureImportance.put("taskSize", 0.2);
        featureImportance.put("inputDataSize", 0.2);
        featureImportance.put("outputDataSize", 0.1);
        featureImportance.put("delayRequirement", 0.2);
        featureImportance.put("wanBandwidth", 0.1);
        featureImportance.put("edgeUtilization", 0.1);
    }
//    private void scheduleEvents() {
//        // Schedule metric updates
//        CloudSim.send(getEntityId(), getEntityId(), METRIC_UPDATE_INTERVAL,
//                SimSettings.PERIODIC_METRIC_UPDATE, null);
//
//        // Schedule learning updates
//        CloudSim.send(getEntityId(), getEntityId(), LEARNING_INTERVAL,
//                SimSettings.LEARNING_UPDATE, null);
//
//        // Schedule model updates
//        CloudSim.send(getEntityId(), getEntityId(), MODEL_UPDATE_INTERVAL,
//                SimSettings.MODEL_UPDATE, null);
//    }

    private void initializePredictionModels() {
        // Create models for predicting task completion time at different devices
        predictionModels.add(new PredictionModel(PredictionTarget.CLOUD_COMPLETION_TIME));
        predictionModels.add(new PredictionModel(PredictionTarget.EDGE_COMPLETION_TIME));
        predictionModels.add(new PredictionModel(PredictionTarget.MOBILE_COMPLETION_TIME));

        // Create models for predicting energy consumption
        predictionModels.add(new PredictionModel(PredictionTarget.CLOUD_ENERGY));
        predictionModels.add(new PredictionModel(PredictionTarget.EDGE_ENERGY));
        predictionModels.add(new PredictionModel(PredictionTarget.MOBILE_ENERGY));
    }

    private void initializeDeviceStats() {
        // Get the number of edge hosts
        numberOfHost = SimSettings.getInstance().getNumOfEdgeHosts();
        System.out.println("Number of edge host is  "+numberOfHost);
        // Initialize stats for all edge devices
        for (int i = 0; i < numberOfHost; i++) {
            deviceStats.put(SimSettings.GENERIC_EDGE_DEVICE_ID, new DeviceStats());
            taskHistory.put(i, new ArrayList<>());
            lastNTaskCompletionTimes.put(i, new ArrayList<>());
            deviceReliability.put(i, 0.95); // Initial reliability estimate
        }
        deviceStats.put(SimSettings.GENERIC_EDGE_DEVICE_ID, new DeviceStats());
        taskHistory.put(SimSettings.GENERIC_EDGE_DEVICE_ID, new ArrayList<>());
        lastNTaskCompletionTimes.put(SimSettings.GENERIC_EDGE_DEVICE_ID, new ArrayList<>());
        deviceReliability.put(SimSettings.GENERIC_EDGE_DEVICE_ID, 0.95); // Initial reliability estimate
        // Initialize stats for cloud
        deviceStats.put(SimSettings.CLOUD_DATACENTER_ID, new DeviceStats());
        taskHistory.put(SimSettings.CLOUD_DATACENTER_ID, new ArrayList<>());
        lastNTaskCompletionTimes.put(SimSettings.CLOUD_DATACENTER_ID, new ArrayList<>());
        deviceReliability.put(SimSettings.CLOUD_DATACENTER_ID, 0.99); // Initial reliability estimate
    }
    /*
     * (non-Javadoc)
     * @see edu.boun.edgecloudsim.edge_orchestrator.EdgeOrchestrator#getDeviceToOffload(edu.boun.edgecloudsim.edge_client.Task)
     *
     * It is assumed that the edge orchestrator app is running on the edge devices in a distributed manner
     */
//    @Override
//    public int getDeviceToOffload(Task task) {
//        int result = 0;
//
//        //RODO: return proper host ID
//
//        if(simScenario.equals("SINGLE_TIER")){
//            result = SimSettings.GENERIC_EDGE_DEVICE_ID;
//        }
//        else if(simScenario.equals("TWO_TIER_WITH_EO")){
//            int bestRemoteEdgeHostIndex = 0;
//            int nearestEdgeHostIndex = 0;
//            double nearestEdgeUtilization = 0;
//
//            //dummy task to simulate a task with 1 Mbit file size to upload and download
//            Task dummyTask = new Task(0, 0, 0, 0, 128, 128, new UtilizationModelFull(), new UtilizationModelFull(), new UtilizationModelFull());
//
//            double wanDelay = SimManager.getInstance().getNetworkModel().getUploadDelay(task.getMobileDeviceId(),
//                    SimSettings.CLOUD_DATACENTER_ID, dummyTask /* 1 Mbit */);
//            double wanBW = (wanDelay == 0) ? 0 : (1 / wanDelay); /* Mbps */
//
//            double manDelay = SimManager.getInstance().getNetworkModel().getUploadDelay(SimSettings.GENERIC_EDGE_DEVICE_ID,
//                    SimSettings.GENERIC_EDGE_DEVICE_ID, dummyTask /* 1 Mbit */);
//
//            double edgeUtilization = SimManager.getInstance().getEdgeServerManager().getAvgUtilization();
//
//            //finding least loaded neighbor edge host
//            double bestRemoteEdgeUtilization = 100; //start with max value
//            for(int hostIndex=0; hostIndex<numberOfHost; hostIndex++){
//                List<EdgeVM> vmArray = SimManager.getInstance().getEdgeServerManager().getVmList(hostIndex);
//
//                double totalUtilization=0;
//                for(int vmIndex=0; vmIndex<vmArray.size(); vmIndex++){
//                    totalUtilization += vmArray.get(vmIndex).getCloudletScheduler().getTotalUtilizationOfCpu(CloudSim.clock());
//                }
//
//                double avgUtilization = (totalUtilization / (double)(vmArray.size()));
//
//                EdgeHost host = (EdgeHost)(vmArray.get(0).getHost()); //all VMs have the same host
//                if(host.getLocation().getServingWlanId() == task.getSubmittedLocation().getServingWlanId()){
//                    nearestEdgeUtilization = totalUtilization / (double)(vmArray.size());
//                    nearestEdgeHostIndex = hostIndex;
//                }
//                else if(avgUtilization < bestRemoteEdgeUtilization){
//                    bestRemoteEdgeHostIndex = hostIndex;
//                    bestRemoteEdgeUtilization = avgUtilization;
//                }
//            }
//
//            switch (policy) {
//                case "UARC" -> {
//                    List<MobileVM> mobileVMList = SimManager.getInstance().getMobileServerManager().getVmList(task.getMobileDeviceId());
//                }
//                default -> {
//                    SimLogger.printLine("Unknown edge orchestrator policy! Terminating simulation...");
//                    System.exit(0);
//                }
//            }
//        }
//        else {
//            SimLogger.printLine("Unknown simulation scenario! Terminating simulation...");
//            System.exit(0);
//        }
//        return result;
//    }
    /**
     * Main method to decide where to offload a task
     */
    @Override
    public int getDeviceToOffload(Task task) {
        int result = 0;

        if (simScenario.equals("SINGLE_TIER")) {
            result = SimSettings.GENERIC_EDGE_DEVICE_ID;
        }
        else if (simScenario.equals("TWO_TIER_WITH_EO")) {
            int bestRemoteEdgeHostIndex = 0;
            int nearestEdgeHostIndex = 0;
            double nearestEdgeUtilization = 0;

            // Get network metrics for decision making
            NetworkMetrics networkMetrics = getNetworkMetrics(task);



            double wanBW = networkMetrics.wanBandwidth;
            double manDelay = networkMetrics.manDelay;
            double edgeUtilization = SimManager.getInstance().getEdgeServerManager().getAvgUtilization();
//            //finding least loaded neighbor edge host
//            double bestRemoteEdgeUtilization = 100; //start with max value
//            for(int hostIndex=0; hostIndex<numberOfHost; hostIndex++){
//                List<EdgeVM> vmArray = SimManager.getInstance().getEdgeServerManager().getVmList(hostIndex);
//
//                double totalUtilization=0;
//                for(int vmIndex=0; vmIndex<vmArray.size(); vmIndex++){
//                    totalUtilization += vmArray.get(vmIndex).getCloudletScheduler().getTotalUtilizationOfCpu(CloudSim.clock());
//                }
//
//                double avgUtilization = (totalUtilization / (double)(vmArray.size()));
//
//                EdgeHost host = (EdgeHost)(vmArray.getFirst().getHost()); //all VMs have the same host
//                if(host.getLocation().getServingWlanId() == task.getSubmittedLocation().getServingWlanId()){
//                    nearestEdgeUtilization = totalUtilization / (double)(vmArray.size());
//                    nearestEdgeHostIndex = hostIndex;
//                }
//                else if(avgUtilization < bestRemoteEdgeUtilization){
//                    bestRemoteEdgeHostIndex = hostIndex;
//                    bestRemoteEdgeUtilization = avgUtilization;
//                }
//            }

            // Get nearest and best remote edge hosts
            // Get nearest and best remote edge hosts
            int[] hostIndices = findBestEdgeHosts(task);
            nearestEdgeHostIndex = hostIndices[0];
            nearestEdgeUtilization = hostIndices[1];
            bestRemoteEdgeHostIndex = hostIndices[2];
            // Make decision based on policy
            switch (policy) {
                case "UARC" -> {
                    // Get mobile VM resources
                    List<MobileVM> mobileVMList = SimManager.getInstance().getMobileServerManager().getVmList(task.getMobileDeviceId());

                    // UARC implementation with ML integration
                    result = mlBasedOffloadingDecision(task, mobileVMList, networkMetrics, nearestEdgeHostIndex, edgeUtilization);
                }
                case "REINFORCEMENT_LEARNING" -> {
                    // Select offloading target using Q-learning
                    result = reinforcementLearningDecision(task, networkMetrics, nearestEdgeHostIndex, edgeUtilization);
                }
                case "PREDICTION_BASED" -> {
                    // Use trained prediction models to select best target
                    result = predictionBasedDecision(task, networkMetrics, nearestEdgeHostIndex, edgeUtilization);
                }
                case "ENSEMBLE_ML" -> {
                    // Combine multiple ML approaches for better decision making
                    result = ensembleMLDecision(task, networkMetrics, nearestEdgeHostIndex, edgeUtilization);
                }
                case "ADAPTIVE_RL" -> {
                    // Reinforcement learning with adaptive exploration rate
                    result = adaptiveRLDecision(task, networkMetrics, nearestEdgeHostIndex, edgeUtilization);
                }
                default -> {
                    SimLogger.printLine("Unknown edge orchestrator policy! Using default ML-based policy.");
                    result = mlBasedOffloadingDecision(task, null, networkMetrics, nearestEdgeHostIndex, edgeUtilization);
                }
            }

            // Extract and store task features for learning
            storeTaskFeatures(task, networkMetrics, edgeUtilization, result);
        }
        else {
            SimLogger.printLine("Unknown simulation scenario! Terminating simulation...");
            System.exit(0);
        }

        return result;
    }

    /**
     * Store task features for learning
     */
    private void storeTaskFeatures(Task task, NetworkMetrics networkMetrics, double edgeUtilization, int selectedDevice) {
        TaskFeatures features = new TaskFeatures(
                task.getTaskType(),
                task.getCloudletLength(),
                task.getCloudletFileSize(),
                task.getCloudletOutputSize(),
                SimSettings.getInstance().getTaskLookUpTable()[task.getTaskType()][12], // delay requirement
                networkMetrics.wanBandwidth,
                networkMetrics.manDelay,
                edgeUtilization,
                selectedDevice,
                CloudSim.clock()
        );

        taskFeatureStore.put(task.getCloudletId(), features);
    }

    /**
     * ML-based offloading decision using feature-based approach
     */
    private int mlBasedOffloadingDecision(Task task, List<MobileVM> mobileVMList, NetworkMetrics networkMetrics,
                                          int nearestEdgeHostIndex, double edgeUtilization) {
        // Extract features for ML decision making
        double taskSize = task.getCloudletLength();
        double inputDataSize = task.getCloudletFileSize();
        double outputDataSize = task.getCloudletOutputSize();
        double delayRequirement = SimSettings.getInstance().getTaskLookUpTable()[task.getTaskType()][12];
        double wanBW = networkMetrics.wanBandwidth;
        double manDelay = networkMetrics.manDelay;

        // Normalized feature values (0-1 range)
        Map<String, Double> features = new HashMap<>();
        features.put("taskType", (double)task.getTaskType() / SimSettings.getInstance().getTaskLookUpTable().length);
        features.put("taskSize", Math.min(taskSize / 20000.0, 1.0)); // Assuming max task size is 20000 MI
        features.put("inputDataSize", Math.min(inputDataSize / 1500.0, 1.0)); // Assuming max input size is 1500 KB
        features.put("outputDataSize", Math.min(outputDataSize / 1500.0, 1.0)); // Assuming max output size is 1500 KB
        features.put("delayRequirement", delayRequirement > 0 ? Math.min(10.0 / delayRequirement, 1.0) : 0.5); // Inverse of delay req
        features.put("wanBandwidth", Math.min(wanBW / 10.0, 1.0)); // Assuming max bandwidth is 10 Mbps
        features.put("edgeUtilization", edgeUtilization / 100.0); // Already in 0-100 range

        // Predict completion time for each option
        double cloudCompletionTime = predictCompletionTime(PredictionTarget.CLOUD_COMPLETION_TIME, features);
        double edgeCompletionTime = predictCompletionTime(PredictionTarget.EDGE_COMPLETION_TIME, features);
        double mobileCompletionTime = predictCompletionTime(PredictionTarget.MOBILE_COMPLETION_TIME, features);

        // Predict energy consumption for each option
        double cloudEnergy = predictEnergy(PredictionTarget.CLOUD_ENERGY, features);
        double edgeEnergy = predictEnergy(PredictionTarget.EDGE_ENERGY, features);
        double mobileEnergy = predictEnergy(PredictionTarget.MOBILE_ENERGY, features);

        // Device reliability estimates
        double cloudReliability = deviceReliability.get(SimSettings.CLOUD_DATACENTER_ID);
        double edgeReliability = deviceReliability.get(nearestEdgeHostIndex);
        double mobileReliability = 0.9; // Assuming mobile devices are less reliable

        // Calculate utility scores (lower is better)
        double timeWeight = delayRequirement > 0 ?
                (REWARD_LATENCY_WEIGHT / 100.0) : 0.5; // If no delay requirement, use default weight

        double cloudScore = calculateUtilityScore(cloudCompletionTime, cloudEnergy, cloudReliability,
                timeWeight, delayRequirement);
        double edgeScore = calculateUtilityScore(edgeCompletionTime, edgeEnergy, edgeReliability,
                timeWeight, delayRequirement);
        double mobileScore = calculateUtilityScore(mobileCompletionTime, mobileEnergy, mobileReliability,
                timeWeight, delayRequirement);

        // Select device with lowest score (best utility)
        if (mobileVMList != null && !mobileVMList.isEmpty() && mobileScore <= edgeScore && mobileScore <= cloudScore) {
            // Execute on mobile device
            return task.getMobileDeviceId();
        } else if (edgeScore <= cloudScore) {
            // Offload to edge server
            return nearestEdgeHostIndex;
        } else {
            // Offload to cloud
            return SimSettings.CLOUD_DATACENTER_ID;
        }
    }

    /**
     * Calculate utility score for device selection (lower is better)
     */
    private double calculateUtilityScore(double completionTime, double energyConsumption, double reliability,
                                         double timeWeight, double delayRequirement) {
        // Normalize completion time relative to delay requirement
        double normalizedTime = delayRequirement > 0 ?
                Math.min(completionTime / delayRequirement, 2.0) : completionTime / 5.0;

        // Normalize energy consumption (assuming max energy is 10 units)
        double normalizedEnergy = Math.min(energyConsumption / 10.0, 1.0);

        // Calculate weighted score (lower is better)
        double timeScore = normalizedTime * timeWeight;
        double energyScore = normalizedEnergy * (REWARD_ENERGY_WEIGHT / 100.0);
        double reliabilityScore = (1.0 - reliability) * (REWARD_RELIABILITY_WEIGHT / 100.0);

        return timeScore + energyScore + reliabilityScore;
    }

    /**
     * Reinforcement learning based decision
     */
    private int reinforcementLearningDecision(Task task, NetworkMetrics networkMetrics,
                                              int nearestEdgeHostIndex, double edgeUtilization) {
        // Determine current state based on task and environment
        String state = getStateRepresentation(task, networkMetrics, edgeUtilization);

        // Epsilon-greedy strategy for exploration vs exploitation
        if (Math.random() < explorationRate) {
            if (edgeUtilization > 70) {
                // If edge is heavily loaded or showing high failure rate, prefer cloud
                return SimSettings.CLOUD_DATACENTER_ID;
            }
            if (!hasEdgeCapacity(task, nearestEdgeHostIndex)) {
                return SimSettings.CLOUD_DATACENTER_ID;
            }
            // Explore: choose a random action
            double rand = Math.random();
            if (rand < 0.33) {
                return SimSettings.CLOUD_DATACENTER_ID;
            } else if (rand < 0.66) {
                return nearestEdgeHostIndex;
            } else {
                return task.getMobileDeviceId();
            }
        } else {
            // Exploit: choose best action from Q-table
            Map<String, Double> actionValues = qTable.get(state);
            if (actionValues == null) {
                // If state not found, create it
                actionValues = new HashMap<>();
                actionValues.put("cloud", 0.0);
                actionValues.put("edge", 0.0);
                actionValues.put("mobile", 0.0);
                qTable.put(state, actionValues);
            }
            if (edgeUtilization > 70) {
                // If edge is heavily loaded or showing high failure rate, prefer cloud
                return SimSettings.CLOUD_DATACENTER_ID;
            }
            if (!hasEdgeCapacity(task, nearestEdgeHostIndex)) {
                return SimSettings.CLOUD_DATACENTER_ID;
            }
            // Find action with highest Q-value
            String bestAction = Collections.max(actionValues.entrySet(), Map.Entry.comparingByValue()).getKey();

            // Convert action to device ID
            switch (bestAction) {
                case "cloud":
                    return SimSettings.CLOUD_DATACENTER_ID;
                case "edge":
                    return nearestEdgeHostIndex;
                case "mobile":
                    return task.getMobileDeviceId();
                default:
                    return nearestEdgeHostIndex; // Default to edge if something goes wrong
            }
        }
    }
    // Add this helper method
    private boolean hasEdgeCapacity(Task task, int hostIndex) {
        List<EdgeVM> vmArray = SimManager.getInstance().getEdgeServerManager().getVmList(hostIndex);
        double requiredCapacity = ((CpuUtilizationModel_Custom)task.getUtilizationModelCpu()).predictUtilization(SimSettings.VM_TYPES.EDGE_VM);

        // Check if any VM has capacity
        for (EdgeVM vm : vmArray) {
            double availableCapacity = 100.0 - vm.getCloudletScheduler().getTotalUtilizationOfCpu(CloudSim.clock());
            if (requiredCapacity <= availableCapacity * 0.9) { // 10% safety margin
                return true;
            }
        }
        return false;
    }
    /**
     * Prediction-based decision using ML models
     */
    private int predictionBasedDecision(Task task, NetworkMetrics networkMetrics,
                                        int nearestEdgeHostIndex, double edgeUtilization) {
        // Extract features
        Map<String, Double> features = extractFeatures(task, networkMetrics, edgeUtilization);

        // Predict metrics for each option
        double cloudCompletion = predictCompletionTime(PredictionTarget.CLOUD_COMPLETION_TIME, features);
        double edgeCompletion = predictCompletionTime(PredictionTarget.EDGE_COMPLETION_TIME, features);
        double mobileCompletion = predictCompletionTime(PredictionTarget.MOBILE_COMPLETION_TIME, features);

        double cloudEnergy = predictEnergy(PredictionTarget.CLOUD_ENERGY, features);
        double edgeEnergy = predictEnergy(PredictionTarget.EDGE_ENERGY, features);
        double mobileEnergy = predictEnergy(PredictionTarget.MOBILE_ENERGY, features);

        // Get delay requirement
        double delayRequirement = SimSettings.getInstance().getTaskLookUpTable()[task.getTaskType()][12];

        // If strict delay requirement, prioritize meeting it
        if (delayRequirement > 0) {
            // Check if any option can meet the requirement
            boolean cloudMeetsDelay = cloudCompletion <= delayRequirement;
            boolean edgeMeetsDelay = edgeCompletion <= delayRequirement;
            boolean mobileMeetsDelay = mobileCompletion <= delayRequirement;

            if (mobileMeetsDelay && (!cloudMeetsDelay || mobileEnergy <= cloudEnergy) &&
                    (!edgeMeetsDelay || mobileEnergy <= edgeEnergy)) {
                return task.getMobileDeviceId();
            } else if (edgeMeetsDelay && (!cloudMeetsDelay || edgeEnergy <= cloudEnergy)) {
                return nearestEdgeHostIndex;
            } else if (cloudMeetsDelay) {
                return SimSettings.CLOUD_DATACENTER_ID;
            }
        }

        // If no strict requirement or none can meet it, balance completion time and energy
        double cloudScore = (cloudCompletion * 0.7) + (cloudEnergy * 0.3);
        double edgeScore = (edgeCompletion * 0.7) + (edgeEnergy * 0.3);
        double mobileScore = (mobileCompletion * 0.7) + (mobileEnergy * 0.3);

        // Choose option with lowest score
        if (mobileScore <= edgeScore && mobileScore <= cloudScore) {
            return task.getMobileDeviceId();
        } else if (edgeScore <= cloudScore) {
            return nearestEdgeHostIndex;
        } else {
            return SimSettings.CLOUD_DATACENTER_ID;
        }
    }

    /**
     * Ensemble ML decision combining multiple approaches
     */
    private int ensembleMLDecision(Task task, NetworkMetrics networkMetrics,
                                   int nearestEdgeHostIndex, double edgeUtilization) {
        // Use multiple ML models and combine their outputs
        int rlDecision = reinforcementLearningDecision(task, networkMetrics, nearestEdgeHostIndex, edgeUtilization);
        int predictionDecision = predictionBasedDecision(task, networkMetrics, nearestEdgeHostIndex, edgeUtilization);
        int mlBasedDecision = mlBasedOffloadingDecision(task, null, networkMetrics, nearestEdgeHostIndex, edgeUtilization);

        // Count votes for each target
        Map<Integer, Integer> votes = new HashMap<>();
        votes.put(rlDecision, votes.getOrDefault(rlDecision, 0) + 1);
        votes.put(predictionDecision, votes.getOrDefault(predictionDecision, 0) + 1);
        votes.put(mlBasedDecision, votes.getOrDefault(mlBasedDecision, 0) + 1);

        // Find majority decision
        int maxVotes = 0;
        int bestDecision = nearestEdgeHostIndex; // Default to edge

        for (Map.Entry<Integer, Integer> entry : votes.entrySet()) {
            if (entry.getValue() > maxVotes) {
                maxVotes = entry.getValue();
                bestDecision = entry.getKey();
            }
        }

        return bestDecision;
    }

    /**
     * Adaptive reinforcement learning with dynamic exploration rate
     */
    private int adaptiveRLDecision(Task task, NetworkMetrics networkMetrics,
                                   int nearestEdgeHostIndex, double edgeUtilization) {
        // Adjust exploration rate based on system stability
        adjustExplorationRate();

        // Use the adjusted RL algorithm
        return reinforcementLearningDecision(task, networkMetrics, nearestEdgeHostIndex, edgeUtilization);
    }

    /**
     * Get success rates for each action from history
     */
    private Map<String, Double> getActionSuccessRate() {
        Map<String, Double> successRates = new HashMap<>();
        Map<String, Integer> successCount = new HashMap<>();
        Map<String, Integer> totalCount = new HashMap<>();

        // Count successes for each action
        for (List<TaskExecution> executions : taskHistory.values()) {
            for (TaskExecution execution : executions) {
                String action;
                if (execution.deviceId == SimSettings.CLOUD_DATACENTER_ID) {
                    action = "cloud";
                } else if (execution.deviceId < numberOfHost) {
                    action = "edge";
                } else {
                    action = "mobile";
                }

                totalCount.put(action, totalCount.getOrDefault(action, 0) + 1);
                if (execution.successful) {
                    successCount.put(action, successCount.getOrDefault(action, 0) + 1);
                }
            }
        }

        // Calculate success rates
        for (String action : Arrays.asList("cloud", "edge", "mobile")) {
            int total = totalCount.getOrDefault(action, 0);
            if (total > 0) {
                double rate = (double) successCount.getOrDefault(action, 0) / total;
                successRates.put(action, rate);
            } else {
                successRates.put(action, 0.5); // Default if no data
            }
        }

        return successRates;
    }

    /**
     * Adjust exploration rate based on system stability
     */
    private void adjustExplorationRate() {
        // Calculate variance in task completion times as a measure of stability
        double totalVariance = 0;
        int count = 0;

        for (Map.Entry<Integer, List<Double>> entry : lastNTaskCompletionTimes.entrySet()) {
            List<Double> times = entry.getValue();
            if (times.size() < 2) continue;

            // Calculate variance
            double mean = times.stream().mapToDouble(Double::doubleValue).average().orElse(0);
            double variance = times.stream()
                    .mapToDouble(time -> Math.pow(time - mean, 2))
                    .average().orElse(0);

            totalVariance += variance;
            count++;
        }

        double avgVariance = count > 0 ? totalVariance / count : 0;

        // Adjust exploration rate based on stability
        // High variance -> more exploration needed
        // Low variance -> more exploitation
        double normalizedVariance = Math.min(avgVariance / 10.0, 1.0); // Normalize to 0-1

        // New exploration rate between 0.05 (stable) and 0.4 (unstable)
        explorationRate = 0.05 + (normalizedVariance * 0.35);
    }

    /**
     * Extract features for ML models
     */
    private Map<String, Double> extractFeatures(Task task, NetworkMetrics networkMetrics, double edgeUtilization) {
        Map<String, Double> features = new HashMap<>();

        // Task features
        features.put("taskType", (double)task.getTaskType());
//        features.put("taskSize", task.getCloudletLength());
//        features.put("inputDataSize", task.getCloudletFileSize());
//        features.put("outputDataSize", task.getCloudletOutputSize());
        features.put("delayRequirement", SimSettings.getInstance().getTaskLookUpTable()[task.getTaskType()][12]);

        // Network features
        features.put("wanBandwidth", networkMetrics.wanBandwidth);
        features.put("manDelay", networkMetrics.manDelay);

        // System features
        features.put("edgeUtilization", edgeUtilization);
        features.put("time", CloudSim.clock());

        return features;
    }

    /**
     * Get state representation for RL
     */
    private String getStateRepresentation(Task task, NetworkMetrics networkMetrics, double edgeUtilization) {
        int taskType = task.getTaskType();

        // Discretize network bandwidth
        String networkCondition;
        if (networkMetrics.wanBandwidth < 2.0) {
            networkCondition = "low_bw";
        } else if (networkMetrics.wanBandwidth < 6.0) {
            networkCondition = "medium_bw";
        } else {
            networkCondition = "high_bw";
        }

        // Discretize system load
        String loadCondition;
        if (edgeUtilization < 30.0) {
            loadCondition = "low_load";
        } else if (edgeUtilization < 70.0) {
            loadCondition = "medium_load";
        } else {
            loadCondition = "high_load";
        }

        return "task_" + taskType + "_" + networkCondition + "_" + loadCondition;
    }

    /**
     * Predict task completion time
     */
    private double predictCompletionTime(PredictionTarget target, Map<String, Double> features) {
        // Find the right prediction model
        for (PredictionModel model : predictionModels) {
            if (model.getTarget() == target) {
                return model.predict(features);
            }
        }

        // Fallback if model not found - simple estimation
        double taskSize = features.get("taskSize");
        double inputSize = features.get("inputDataSize");
        double outputSize = features.get("outputDataSize");
        double wanBW = features.get("wanBandwidth");
        double edgeUtil = features.get("edgeUtilization");

        switch (target) {
            case CLOUD_COMPLETION_TIME:
                // Estimate: network transfer + cloud processing
                return ((inputSize + outputSize) / wanBW) + (taskSize / 4000); // Assuming cloud is 4x faster
            case EDGE_COMPLETION_TIME:
                // Estimate: edge processing (affected by utilization)
                double edgeCapacity = Math.max(5.0, 100.0 - edgeUtil); // 5-100% capacity
                return (taskSize / 1000) * (100.0 / edgeCapacity);
            case MOBILE_COMPLETION_TIME:
                // Estimate: mobile processing (slower)
                return taskSize / 500; // Assuming mobile is 2x slower than edge
            default:
                return 1.0; // Default fallback
        }
    }

    /**
     * Predict energy consumption
     */
    private double predictEnergy(PredictionTarget target, Map<String, Double> features) {
        // Find the right prediction model
        for (PredictionModel model : predictionModels) {
            if (model.getTarget() == target) {
                return model.predict(features);
            }
        }

        // Fallback if model not found - simple estimation
        double taskSize = features.get("taskSize");
        double inputSize = features.get("inputDataSize");
        double outputSize = features.get("outputDataSize");

        switch (target) {
            case CLOUD_ENERGY:
                // Cloud: mostly network energy
                return 0.1 + ((inputSize + outputSize) * 0.001);
            case EDGE_ENERGY:
                // Edge: moderate energy
                return 0.2 + (taskSize * 0.0002);
            case MOBILE_ENERGY:
                // Mobile: highest energy impact (battery powered)
                return 0.5 + (taskSize * 0.0005);
            default:
                return 1.0; // Default fallback
        }
    }

    /**
     * Get VM to offload the task to
     */
    @Override
    public Vm getVmToOffload(Task task, int deviceId) {
        Vm selectedVM = null;

        if (deviceId == SimSettings.CLOUD_DATACENTER_ID) {
            // Cloud VM selection with ML optimization
            selectedVM = selectCloudVM(task);
        } else if (deviceId == SimSettings.GENERIC_EDGE_DEVICE_ID || deviceId < numberOfHost) {
            // Edge VM selection with ML optimization
            selectedVM = selectEdgeVM(task, deviceId);
        } else {
            // Mobile device - should use task's own mobile device
            List<MobileVM> vmArray = SimManager.getInstance().getMobileServerManager().getVmList(task.getMobileDeviceId());
            if (!vmArray.isEmpty()) {
                selectedVM = vmArray.get(0);
            }
        }

        return selectedVM;
    }

    /**
     * ML-optimized cloud VM selection
     */
    private Vm selectCloudVM(Task task) {
        Vm selectedVM = null;
        double bestScore = Double.MAX_VALUE;

        List<Host> hostList = SimManager.getInstance().getCloudServerManager().getDatacenter().getHostList();
        for (int hostIndex = 0; hostIndex < hostList.size(); hostIndex++) {
            List<CloudVM> vmArray = SimManager.getInstance().getCloudServerManager().getVmList(hostIndex);
            for (CloudVM vm : vmArray) {
                double requiredCapacity = ((CpuUtilizationModel_Custom)task.getUtilizationModelCpu()).predictUtilization(vm.getVmType());
                double availableCapacity = 100.0 - vm.getCloudletScheduler().getTotalUtilizationOfCpu(CloudSim.clock());

                if (requiredCapacity <= availableCapacity) {
                    // Calculate fitness score for this VM
                    double vmScore = calculateVMFitness(vm, task, availableCapacity);

                    if (vmScore < bestScore) {
                        selectedVM = vm;
                        bestScore = vmScore;
                    }
                }
            }
        }

        return selectedVM;
    }

    /**
     * ML-optimized edge VM selection
     */
    private Vm selectEdgeVM(Task task, int deviceId) {
        Vm selectedVM = null;
        double bestScore = Double.MAX_VALUE;

        // Use specific host if provided, otherwise check all hosts
        if (deviceId == SimSettings.GENERIC_EDGE_DEVICE_ID) {
            for (int hostIndex = 0; hostIndex < numberOfHost; hostIndex++) {
                Vm vm = selectBestVMForHost(task, hostIndex);
                if (vm != null) {
                    double score = calculateVMFitness((EdgeVM)vm, task,
                            100.0 - ((EdgeVM)vm).getCloudletScheduler().getTotalUtilizationOfCpu(CloudSim.clock()));
                    if (score < bestScore) {
                        selectedVM = vm;
                        bestScore = score;
                    }
                }
            }
        } else {
            // Select from specific host
            selectedVM = selectBestVMForHost(task, deviceId);
        }

        return selectedVM;
    }

    /**
     * Select best VM from a specific host
     */
    private Vm selectBestVMForHost(Task task, int hostIndex) {
        Vm selectedVM = null;
        double bestScore = Double.MAX_VALUE;

        List<EdgeVM> vmArray = SimManager.getInstance().getEdgeServerManager().getVmList(hostIndex);
        for (EdgeVM vm : vmArray) {
            double requiredCapacity = ((CpuUtilizationModel_Custom)task.getUtilizationModelCpu()).predictUtilization(vm.getVmType());
            double availableCapacity = 100.0 - vm.getCloudletScheduler().getTotalUtilizationOfCpu(CloudSim.clock());

            if (requiredCapacity <= availableCapacity) {
                // Calculate fitness score for this VM
                double vmScore = calculateVMFitness(vm, task, availableCapacity);

                if (vmScore < bestScore) {
                    selectedVM = vm;
                    bestScore = vmScore;
                }
            }
        }

        return selectedVM;
    }

    /**
     * Calculate VM fitness score for task (lower is better)
     */
    private double calculateVMFitness(Vm vm, Task task, double availableCapacity) {
        // VM type fitness - certain VMs may be better for certain task types
        double vmTypeFitness = getVMTypeFitness(vm.getMips(), task.getTaskType());

        // Capacity fitness - prefer VMs with enough but not too much capacity
        double capacityFitness = getCapacityFitness(availableCapacity,
                ((CpuUtilizationModel_Custom)task.getUtilizationModelCpu()).predictUtilization(SimSettings.VM_TYPES.EDGE_VM));

        // Historical performance of this VM type for this task type
        double historyFitness = getHistoricalFitness(vm, task.getTaskType());

        // Combined score (lower is better)
        return (vmTypeFitness * 0.3) + (capacityFitness * 0.5) + (historyFitness * 0.2);
    }

    /**
     * Get VM type fitness for task type
     */
    private double getVMTypeFitness(double vmMips, int taskType) {
        // Task types might have different computational needs
        double taskCompute = SimSettings.getInstance().getTaskLookUpTable()[taskType][5]; // MI

        // Normalize fitness (lower is better)
        return Math.abs(1.0 - (vmMips / (taskCompute * 2.0)));
    }

    /**
     * Get capacity fitness
     */
    private double getCapacityFitness(double availableCapacity, double requiredCapacity) {
        // Normalize required capacity to 0-100 range
        double normalizedRequired = Math.min(requiredCapacity, 100.0);

        // Prefer VMs with 20-30% more capacity than required (avoid over/under provisioning)
        double idealCapacity = normalizedRequired * 1.3;

        // Compute fitness (lower is better)
        return Math.abs(availableCapacity - idealCapacity) / 100.0;
    }

    /**
     * Get historical fitness
     */
    private double getHistoricalFitness(Vm vm, int taskType) {
        // Find this VM's host
        int hostId = vm.getHost().getId();

        if (!taskHistory.containsKey(hostId)) {
            return 0.5; // Default if no history
        }

        // Get history for this host
        List<TaskExecution> executions = taskHistory.get(hostId);

        // Filter for this task type
        List<TaskExecution> typeExecutions = executions.stream()
                .filter(e -> e.taskType == taskType)
                .toList();

        if (typeExecutions.isEmpty()) {
            return 0.5; // Default if no history for this type
        }

        // Calculate success rate
        long successCount = typeExecutions.stream().filter(e -> e.successful).count();
        double successRate = (double)successCount / typeExecutions.size();

        // Calculate average completion time relative to deadline
        double avgTimePerformance = typeExecutions.stream()
                .mapToDouble(e -> e.completionTime / Math.max(1.0, e.deadline))
                .average().orElse(1.0);

        // Combine metrics (lower is better)
        return (1.0 - successRate) * 0.7 + (Math.min(avgTimePerformance, 2.0) / 2.0) * 0.3;
    }

    /**
     * Get network metrics
     */
    private NetworkMetrics getNetworkMetrics(Task task) {
        // Dummy task for probing network conditions
        Task dummyTask = new Task(0, 0, 0, 0, 128, 128,
                new UtilizationModelFull(), new UtilizationModelFull(),
                new UtilizationModelFull());

        // Measure WAN delay/bandwidth
        double wanDelay = SimManager.getInstance().getNetworkModel().getUploadDelay(
                task.getMobileDeviceId(), SimSettings.CLOUD_DATACENTER_ID, dummyTask);
        double wanBW = (wanDelay == 0) ? 0 : (1 / wanDelay); // Mbps

        // Measure MAN delay
        double manDelay = SimManager.getInstance().getNetworkModel().getUploadDelay(
                SimSettings.GENERIC_EDGE_DEVICE_ID, SimSettings.GENERIC_EDGE_DEVICE_ID, dummyTask);

        return new NetworkMetrics(wanBW, manDelay);
    }

    /**
     * Find best edge hosts for task offloading
     * @return int array containing [nearestEdgeHostIndex, nearestEdgeUtilization, bestRemoteEdgeHostIndex]
     */
    private int[] findBestEdgeHosts(Task task) {
        int nearestEdgeHostIndex = 0;
        double nearestEdgeUtilization = 100; // Start with max value

        int bestRemoteEdgeHostIndex = 0;
        double bestRemoteEdgeUtilization = 100; // Start with max value

        for (int hostIndex = 0; hostIndex < numberOfHost; hostIndex++) {
            List<EdgeVM> vmArray = SimManager.getInstance().getEdgeServerManager().getVmList(hostIndex);

            if (vmArray.isEmpty()) {
                continue;
            }

            double totalUtilization = 0;
            for (EdgeVM vm : vmArray) {
                totalUtilization += vm.getCloudletScheduler().getTotalUtilizationOfCpu(CloudSim.clock());
            }

            double avgUtilization = totalUtilization / vmArray.size();

            EdgeHost host = (EdgeHost)(vmArray.get(0).getHost()); // All VMs have the same host
            if (host.getLocation().getServingWlanId() == task.getSubmittedLocation().getServingWlanId()) {
                nearestEdgeUtilization = avgUtilization;
                nearestEdgeHostIndex = hostIndex;
            } else if (avgUtilization < bestRemoteEdgeUtilization) {
                bestRemoteEdgeHostIndex = hostIndex;
                bestRemoteEdgeUtilization = avgUtilization;
            }
        }

        // Return values as array
        return new int[]{nearestEdgeHostIndex, (int)Math.round(nearestEdgeUtilization), bestRemoteEdgeHostIndex};
    }

    /**
     * Calculate MAN delay for agent
     */
    public double getManDelayForAgent() {
        double delay = 0;
        double mu = 0;
        double lambda = 0;
        double bandwidth = 1300 * 1024; // Kbps

        if (totalSizeOfActiveManTasks == 0) {
            mu = bandwidth;
        } else {
            mu = bandwidth / (totalSizeOfActiveManTasks * 8);
        }

        lambda = activeManTaskCount;

        if (lambda >= mu) {
            return 0;
        } else {
            delay = 1 / (mu - lambda);
            return delay;
        }
    }

    @Override
    public void startEntity() {
        // Initialize number of hosts
        numberOfHost = SimManager.getInstance().getEdgeServerManager().getDatacenterList().size();
    }

    @Override
    public void processEvent(SimEvent ev) {
        switch (ev.getTag()) {
//            case PERIODIC_METRIC_UPDATE:
//                updateMetrics();
//
//                // Schedule next update
//                CloudSim.send(getEntityId(), getEntityId(), METRIC_UPDATE_INTERVAL,
//                        SimSettings.PERIODIC_METRIC_UPDATE, null);
//                break;
//
//            case LEARNING_UPDATE:
//                updateLearningModels();
//
//                // Schedule next learning
//                CloudSim.send(getEntityId(), getEntityId(), LEARNING_INTERVAL,
//                        SimSettings.LEARNING_UPDATE, null);
//                break;
//
//            case MODEL_UPDATE:
//                retrainModels();
//
//                // Schedule next model update
//                CloudSim.send(getEntityId(), getEntityId(), MODEL_UPDATE_INTERVAL,
//                        SimSettings.MODEL_UPDATE, null);
//                break;

            case TASK_COMPLETED:
                Task completedTask = (Task) ev.getData();
                recordTaskCompletion(completedTask, true);
                updateReinforcementLearning(completedTask, true);
                break;

            case TASK_FAILED:
                Task failedTask = (Task) ev.getData();
                recordTaskCompletion(failedTask, false);
                updateReinforcementLearning(failedTask, false);
                break;

            default:
                break;
        }
    }

    /**
     * Update system metrics
     */
    private void updateMetrics() {
        // Update device stats
        for (int hostIndex = 0; hostIndex < numberOfHost; hostIndex++) {
            List<EdgeVM> vmArray = SimManager.getInstance().getEdgeServerManager().getVmList(hostIndex);

            if (!vmArray.isEmpty()) {
                double totalUtilization = 0;
                for (EdgeVM vm : vmArray) {
                    totalUtilization += vm.getCloudletScheduler().getTotalUtilizationOfCpu(CloudSim.clock());
                }

                double avgUtilization = totalUtilization / vmArray.size();

                DeviceStats stats = deviceStats.getOrDefault(hostIndex, new DeviceStats());
                stats.updateUtilization(avgUtilization);
                deviceStats.put(hostIndex, stats);
            }
        }

        // Update cloud stats
        double cloudUtilization = getCloudUtilization();
        DeviceStats cloudStats = deviceStats.getOrDefault(SimSettings.CLOUD_DATACENTER_ID, new DeviceStats());
        cloudStats.updateUtilization(cloudUtilization);
        deviceStats.put(SimSettings.CLOUD_DATACENTER_ID, cloudStats);
    }

    /**
     * Get cloud utilization
     */
    private double getCloudUtilization() {
        double totalUtilization = 0;
        int vmCount = 0;

        List<Host> hostList = SimManager.getInstance().getCloudServerManager().getDatacenter().getHostList();
        for (Host host : hostList) {
            List<CloudVM> vmArray = SimManager.getInstance().getCloudServerManager().getVmList(host.getId());
            for (CloudVM vm : vmArray) {
                totalUtilization += vm.getCloudletScheduler().getTotalUtilizationOfCpu(CloudSim.clock());
                vmCount++;
            }
        }

        return vmCount > 0 ? totalUtilization / vmCount : 0;
    }

    /**
     * Update ML models based on recent data
     */
    private void updateLearningModels() {
        // Update feature importance based on recent task performance
        updateFeatureImportance();

        // Update device reliability based on task history
        updateDeviceReliability();

        // Update prediction models with recent data
        for (PredictionModel model : predictionModels) {
            model.updateModel();
        }
    }

    /**
     * Update feature importance based on performance
     */
    private void updateFeatureImportance() {
        // Simple implementation - in a real system this would use more sophisticated methods
        Map<String, Double> featureCorrelation = new HashMap<>();

        // Initialize with existing importance
        featureCorrelation.putAll(featureImportance);

        // Analyze completed tasks
        for (Map.Entry<Integer, List<TaskExecution>> entry : taskHistory.entrySet()) {
            List<TaskExecution> executions = entry.getValue();

            if (executions.size() < 10) continue; // Need sufficient data

            // Get recent executions
            int startIdx = Math.max(0, executions.size() - 50);
            List<TaskExecution> recent = executions.subList(startIdx, executions.size());

            // Analyze feature correlations with success
            analyzeFeatureCorrelations(recent, featureCorrelation);
        }

        // Update feature importance (with smoothing)
        for (String feature : featureImportance.keySet()) {
            double currentValue = featureImportance.get(feature);
            double newValue = featureCorrelation.get(feature);

            // Smooth update (80% old, 20% new)
            featureImportance.put(feature, currentValue * 0.8 + newValue * 0.2);
        }

        // Normalize to ensure sum = 1.0
        double sum = featureImportance.values().stream().mapToDouble(Double::doubleValue).sum();
        if (sum > 0) {
            for (String feature : featureImportance.keySet()) {
                featureImportance.put(feature, featureImportance.get(feature) / sum);
            }
        }
    }

    /**
     * Analyze feature correlations with task success
     */
    private void analyzeFeatureCorrelations(List<TaskExecution> executions, Map<String, Double> correlations) {
        // Simple correlation analysis
        // In a real system, this would use more sophisticated ML techniques

        // Calculate average feature values for successful vs failed tasks
        Map<String, Double> successAvg = new HashMap<>();
        Map<String, Double> failureAvg = new HashMap<>();
        int successCount = 0;
        int failureCount = 0;

        for (TaskExecution execution : executions) {
            Map<String, Double> targetMap;
            if (execution.successful) {
                targetMap = successAvg;
                successCount++;
            } else {
                targetMap = failureAvg;
                failureCount++;
            }

            // Find matching task features
            TaskFeatures features = taskFeatureStore.get(execution.taskId);
            if (features == null) continue;

            // Sum feature values
            addFeatureValues(targetMap, features);
        }

        // Calculate averages
        for (String feature : successAvg.keySet()) {
            if (successCount > 0) {
                successAvg.put(feature, successAvg.get(feature) / successCount);
            }
        }

        for (String feature : failureAvg.keySet()) {
            if (failureCount > 0) {
                failureAvg.put(feature, failureAvg.get(feature) / failureCount);
            }
        }

        // Calculate correlation as difference between successful and failed averages
        for (String feature : correlations.keySet()) {
            double successVal = successAvg.getOrDefault(feature, 0.0);
            double failureVal = failureAvg.getOrDefault(feature, 0.0);

            // Calculate absolute difference and normalize
            double diff = Math.abs(successVal - failureVal);
            correlations.put(feature, diff);
        }
    }

    /**
     * Add feature values to map
     */
    private void addFeatureValues(Map<String, Double> map, TaskFeatures features) {
        map.put("taskType", map.getOrDefault("taskType", 0.0) + features.taskType);
        map.put("taskSize", map.getOrDefault("taskSize", 0.0) + features.taskSize);
        map.put("inputDataSize", map.getOrDefault("inputDataSize", 0.0) + features.inputDataSize);
        map.put("outputDataSize", map.getOrDefault("outputDataSize", 0.0) + features.outputDataSize);
        map.put("delayRequirement", map.getOrDefault("delayRequirement", 0.0) + features.delayRequirement);
        map.put("wanBandwidth", map.getOrDefault("wanBandwidth", 0.0) + features.wanBandwidth);
        map.put("edgeUtilization", map.getOrDefault("edgeUtilization", 0.0) + features.edgeUtilization);
    }

    /**
     * Update device reliability based on task history
     */
    private void updateDeviceReliability() {
        for (Map.Entry<Integer, List<TaskExecution>> entry : taskHistory.entrySet()) {
            int deviceId = entry.getKey();
            List<TaskExecution> executions = entry.getValue();

            if (executions.isEmpty()) continue;

            // Consider only recent history
            int startIdx = Math.max(0, executions.size() - 50);
            List<TaskExecution> recent = executions.subList(startIdx, executions.size());

            // Calculate success rate
            long successCount = recent.stream().filter(e -> e.successful).count();
            double reliability = recent.isEmpty() ? 0.95 : (double)successCount / recent.size();

            // Update with smoothing
            double currentReliability = deviceReliability.getOrDefault(deviceId, 0.95);
            deviceReliability.put(deviceId, currentReliability * 0.8 + reliability * 0.2);
        }
    }

    /**
     * Retrain prediction models with all available data
     */
    private void retrainModels() {
        SimLogger.printLine("Retraining ML models at time " + CloudSim.clock());

        // Collect all training data
        List<Map<String, Double>> featureVectors = new ArrayList<>();
        List<Map<PredictionTarget, Double>> targetValues = new ArrayList<>();

        // Extract features and targets from task history
        for (Map.Entry<Integer, List<TaskExecution>> entry : taskHistory.entrySet()) {
            for (TaskExecution execution : entry.getValue()) {
                // Find matching task features
                TaskFeatures features = taskFeatureStore.get(execution.taskId);
                if (features == null) continue;

                // Create feature vector
                Map<String, Double> featureVector = createFeatureVector(features);
                featureVectors.add(featureVector);

                // Create target values
                Map<PredictionTarget, Double> targets = createTargetValues(execution);
                targetValues.add(targets);
            }
        }

        // Retrain each model
        for (PredictionModel model : predictionModels) {
            model.retrain(featureVectors, targetValues);
        }
    }

    /**
     * Create feature vector for model training
     */
    private Map<String, Double> createFeatureVector(TaskFeatures features) {
        Map<String, Double> vector = new HashMap<>();

        // Normalize features
        vector.put("taskType", (double)features.taskType / SimSettings.getInstance().getTaskLookUpTable().length);
        vector.put("taskSize", Math.min(features.taskSize / 20000.0, 1.0));
        vector.put("inputDataSize", Math.min(features.inputDataSize / 1500.0, 1.0));
        vector.put("outputDataSize", Math.min(features.outputDataSize / 1500.0, 1.0));
        vector.put("delayRequirement", features.delayRequirement > 0 ?
                Math.min(10.0 / features.delayRequirement, 1.0) : 0.5);
        vector.put("wanBandwidth", Math.min(features.wanBandwidth / 10.0, 1.0));
        vector.put("edgeUtilization", features.edgeUtilization / 100.0);

        return vector;
    }

    /**
     * Create target values for model training
     */
    private Map<PredictionTarget, Double> createTargetValues(TaskExecution execution) {
        Map<PredictionTarget, Double> targets = new HashMap<>();

        // Set values based on execution device
        if (execution.deviceId == SimSettings.CLOUD_DATACENTER_ID) {
            // This was executed on cloud
            targets.put(PredictionTarget.CLOUD_COMPLETION_TIME, execution.completionTime);
            targets.put(PredictionTarget.CLOUD_ENERGY, execution.energyConsumption);
        } else if (execution.deviceId < numberOfHost) {
            // This was executed on edge
            targets.put(PredictionTarget.EDGE_COMPLETION_TIME, execution.completionTime);
            targets.put(PredictionTarget.EDGE_ENERGY, execution.energyConsumption);
        } else {
            // This was executed on mobile
            targets.put(PredictionTarget.MOBILE_COMPLETION_TIME, execution.completionTime);
            targets.put(PredictionTarget.MOBILE_ENERGY, execution.energyConsumption);
        }

        return targets;
    }

    /**
     * Record task completion
     */
    private void recordTaskCompletion(Task task, boolean success) {
        int deviceId = task.getAssociatedHostId();
        double completionTime = CloudSim.clock() - task.getSubmissionTime();
        double deadline = SimSettings.getInstance().getTaskLookUpTable()[task.getTaskType()][12];

        // Energy consumption estimate (simplified)
        double energy = estimateEnergy(task, deviceId);

        // Create task execution record
        TaskExecution execution = new TaskExecution(
                task.getCloudletId(),
                task.getTaskType(),
                deviceId,
                task.getSubmissionTime(),
                completionTime,
                deadline,
                energy,
                success
        );

        // Add to history
        if (!taskHistory.containsKey(deviceId)) {
            taskHistory.put(deviceId, new ArrayList<>());
        }
        taskHistory.get(deviceId).add(execution);

        // Update completion time history
        if (!lastNTaskCompletionTimes.containsKey(deviceId)) {
            lastNTaskCompletionTimes.put(deviceId, new ArrayList<>());
        }

        List<Double> times = lastNTaskCompletionTimes.get(deviceId);
        times.add(completionTime);

        // Keep history bounded
        if (times.size() > 100) {
            times.remove(0);
        }

        // Log completion
        SimLogger.getInstance().taskExecuted(task.getCloudletId());
    }

    /**
     * Estimate energy consumption (simplified model)
     */
    private double estimateEnergy(Task task, int deviceId) {
        double taskSize = task.getCloudletLength();
        double dataSize = task.getCloudletFileSize() + task.getCloudletOutputSize();

        if (deviceId == SimSettings.CLOUD_DATACENTER_ID) {
            // Cloud energy model
            return 0.1 + (taskSize * 0.0001) + (dataSize * 0.001);
        } else if (deviceId < numberOfHost) {
            // Edge energy model
            return 0.2 + (taskSize * 0.0002) + (dataSize * 0.0002);
        } else {
            // Mobile energy model
            return 0.5 + (taskSize * 0.0005);
        }
    }

    /**
     * Update reinforcement learning based on task result
     */
    private void updateReinforcementLearning(Task task, boolean success) {
        // Get task features
        TaskFeatures features = taskFeatureStore.get(task.getCloudletId());
        if (features == null) return;

        // Get state
        String state = getStateRepresentation(task,
                new NetworkMetrics(features.wanBandwidth, 0),
                features.edgeUtilization);

        // Get action taken
        String action;
        if (features.selectedDevice == SimSettings.CLOUD_DATACENTER_ID) {
            action = "cloud";
        } else if (features.selectedDevice < numberOfHost) {
            action = "edge";
        } else {
            action = "mobile";
        }

        // Calculate reward
        double reward = calculateReward(task, features.selectedDevice, success);

        // Update Q-value using Q-learning formula: Q(s,a) += α * (r + γ * max(Q(s',a')) - Q(s,a))
        Map<String, Double> actionValues = qTable.getOrDefault(state, new HashMap<>());
        if (actionValues.isEmpty()) {
            actionValues.put("cloud", 0.0);
            actionValues.put("edge", 0.0);
            actionValues.put("mobile", 0.0);
            qTable.put(state, actionValues);
        }

        double oldQValue = actionValues.getOrDefault(action, 0.0);
        double maxNextQValue = 0.0; // Simplified: assume terminal state

        // Q-learning update
        double newQValue = oldQValue + learningRate * (reward + discountFactor * maxNextQValue - oldQValue);
        actionValues.put(action, newQValue);
    }

    /**
     * Calculate reward for reinforcement learning
     */
    private double calculateReward(Task task, int deviceId, boolean success) {
        if (!success) {
            return -10.0; // Penalty for failure
        }

        double completionTime = CloudSim.clock() - task.getSubmissionTime();
        double deadline = SimSettings.getInstance().getTaskLookUpTable()[task.getTaskType()][12];

        // Calculate time-based reward component
        double timeReward;
        if (deadline > 0 && completionTime <= deadline) {
            // Met deadline - higher reward for faster completion
            timeReward = 5.0 * (1.0 - (completionTime / deadline));
        } else if (deadline > 0) {
            // Missed deadline - penalty based on how much it was missed
            timeReward = -5.0 * Math.min(2.0, (completionTime / deadline) - 1.0);
        } else {
            // No deadline - reward based on completion time
            timeReward = 2.0 - Math.min(2.0, completionTime / 5.0);
        }

        // Energy reward component
        double energyConsumption = estimateEnergy(task, deviceId);
        double energyReward = -energyConsumption; // Negative reward for energy used

        // Combined reward
        return timeReward + energyReward;
    }
//    @Override
//    public Vm getVmToOffload(Task task, int deviceId) {
//        Vm selectedVM = null;
//
//        if(deviceId == SimSettings.CLOUD_DATACENTER_ID){
//            //Select VM on cloud devices via Least Loaded algorithm!
//            double selectedVmCapacity = 0; //start with min value
//            List<Host> list = SimManager.getInstance().getCloudServerManager().getDatacenter().getHostList();
//            for (int hostIndex=0; hostIndex < list.size(); hostIndex++) {
//                List<CloudVM> vmArray = SimManager.getInstance().getCloudServerManager().getVmList(hostIndex);
//                for(int vmIndex=0; vmIndex<vmArray.size(); vmIndex++){
//                    double requiredCapacity = ((CpuUtilizationModel_Custom)task.getUtilizationModelCpu()).predictUtilization(vmArray.get(vmIndex).getVmType());
//                    double targetVmCapacity = (double)100 - vmArray.get(vmIndex).getCloudletScheduler().getTotalUtilizationOfCpu(CloudSim.clock());
//                    if(requiredCapacity <= targetVmCapacity && targetVmCapacity > selectedVmCapacity){
//                        selectedVM = vmArray.get(vmIndex);
//                        selectedVmCapacity = targetVmCapacity;
//                    }
//                }
//            }
//        }
//        else if(deviceId == SimSettings.GENERIC_EDGE_DEVICE_ID){
//            //Select VM on edge devices via Least Loaded algorithm!
//            double selectedVmCapacity = 0; //start with min value
//            for(int hostIndex=0; hostIndex<numberOfHost; hostIndex++){
//                List<EdgeVM> vmArray = SimManager.getInstance().getEdgeServerManager().getVmList(hostIndex);
//                for(int vmIndex=0; vmIndex<vmArray.size(); vmIndex++){
//                    double requiredCapacity = ((CpuUtilizationModel_Custom)task.getUtilizationModelCpu()).predictUtilization(vmArray.get(vmIndex).getVmType());
//                    double targetVmCapacity = (double)100 - vmArray.get(vmIndex).getCloudletScheduler().getTotalUtilizationOfCpu(CloudSim.clock());
//                    if(requiredCapacity <= targetVmCapacity && targetVmCapacity > selectedVmCapacity){
//                        selectedVM = vmArray.get(vmIndex);
//                        selectedVmCapacity = targetVmCapacity;
//                    }
//                }
//            }
//        }
//        else{
//            //if the host is specifically defined!
//            List<EdgeVM> vmArray = SimManager.getInstance().getEdgeServerManager().getVmList(deviceId);
//
//            //Select VM on edge devices via Least Loaded algorithm!
//            double selectedVmCapacity = 0; //start with min value
//            for(int vmIndex=0; vmIndex<vmArray.size(); vmIndex++){
//                double requiredCapacity = ((CpuUtilizationModel_Custom)task.getUtilizationModelCpu()).predictUtilization(vmArray.get(vmIndex).getVmType());
//                double targetVmCapacity = (double)100 - vmArray.get(vmIndex).getCloudletScheduler().getTotalUtilizationOfCpu(CloudSim.clock());
//                if(requiredCapacity <= targetVmCapacity && targetVmCapacity > selectedVmCapacity){
//                    selectedVM = vmArray.get(vmIndex);
//                    selectedVmCapacity = targetVmCapacity;
//                }
//            }
//        }
//        return selectedVM;
//    }

//    public double getManDelayForAgent(){
//        double delay = 0;
//        double mu = 0;
//        double lambda = 0;
//        double bandwidth = 1300*1024; //Kbps , C
//
//        if (totalSizeOfActiveManTasks == 0){
//            mu = bandwidth;
//        }else{
//            mu = bandwidth / (totalSizeOfActiveManTasks * 8);
//        }
//
//        lambda = activeManTaskCount;
//
//
//        if (lambda >= mu){
//            return 0;
//        }else{
//            delay = 1 / (mu - lambda);
//            return delay;
//        }
//    }
    /**
     * Prediction target enum
     */
    private enum PredictionTarget {
        CLOUD_COMPLETION_TIME,
        EDGE_COMPLETION_TIME,
        MOBILE_COMPLETION_TIME,
        CLOUD_ENERGY,
        EDGE_ENERGY,
        MOBILE_ENERGY
    }

    /**
     * Simple prediction model
     * In a real system this would be a proper ML model like Linear Regression, Random Forest, etc.
     */
    private static class PredictionModel {
        private final PredictionTarget target;
        private final Map<String, Double> weights = new HashMap<>();
        private double bias = 0.0;

        public PredictionModel(PredictionTarget target) {
            this.target = target;

            // Initialize with reasonable defaults
            weights.put("taskType", 0.1);
            weights.put("taskSize", 0.5);
            weights.put("inputDataSize", 0.1);
            weights.put("outputDataSize", 0.1);
            weights.put("delayRequirement", 0.1);
            weights.put("wanBandwidth", 0.1);
            weights.put("edgeUtilization", 0.1);

            // Different defaults based on target
            switch (target) {
                case CLOUD_COMPLETION_TIME:
                    weights.put("wanBandwidth", 0.3); // WAN bandwidth more important for cloud
                    break;
                case EDGE_COMPLETION_TIME:
                    weights.put("edgeUtilization", 0.3); // Edge utilization more important for edge
                    break;
                case MOBILE_COMPLETION_TIME:
                    weights.put("taskSize", 0.7); // Task size more important for mobile
                    break;
                case CLOUD_ENERGY:
                    weights.put("inputDataSize", 0.3); // Data size impacts network energy
                    weights.put("outputDataSize", 0.3);
                    break;
                case EDGE_ENERGY:
                    weights.put("taskSize", 0.4); // Task size impacts processing energy
                    break;
                case MOBILE_ENERGY:
                    weights.put("taskSize", 0.6); // Task size heavily impacts mobile energy
                    break;
            }

            // Set appropriate bias
            switch (target) {
                case CLOUD_COMPLETION_TIME:
                    bias = 0.5; // Base network delay
                    break;
                case EDGE_COMPLETION_TIME:
                    bias = 0.2; // Lower base delay
                    break;
                case MOBILE_COMPLETION_TIME:
                    bias = 0.1; // Lowest base delay
                    break;
                case CLOUD_ENERGY:
                    bias = 0.1; // Base energy cost
                    break;
                case EDGE_ENERGY:
                    bias = 0.2; // Medium energy cost
                    break;
                case MOBILE_ENERGY:
                    bias = 0.5; // Highest energy cost
                    break;
            }
        }

        public PredictionTarget getTarget() {
            return target;
        }

        /**
         * Predict value based on features
         */
        public double predict(Map<String, Double> features) {
            double prediction = bias;

            // Linear combination of features * weights
            for (Map.Entry<String, Double> feature : features.entrySet()) {
                if (weights.containsKey(feature.getKey())) {
                    prediction += feature.getValue() * weights.get(feature.getKey());
                }
            }

            // Scale based on target type
            switch (target) {
                case CLOUD_COMPLETION_TIME:
                case EDGE_COMPLETION_TIME:
                case MOBILE_COMPLETION_TIME:
                    return Math.max(0.1, prediction * 10.0); // Scale to reasonable completion time (seconds)
                case CLOUD_ENERGY:
                case EDGE_ENERGY:
                case MOBILE_ENERGY:
                    return Math.max(0.05, prediction * 5.0); // Scale to reasonable energy consumption
                default:
                    return prediction;
            }
        }

        /**
         * Update model with new data
         */
        public void updateModel() {
            // In a real system, this would do incremental model updates
            // For simplicity, just adjust weights slightly based on random values
            // This simulates model drift and adaptation
            Random rand = new Random();

            for (String feature : weights.keySet()) {
                // Small random adjustment to weights
                double currentWeight = weights.get(feature);
                double adjustment = (rand.nextDouble() - 0.5) * 0.02; // ±1% change
                weights.put(feature, currentWeight * (1.0 + adjustment));
            }

            // Small adjustment to bias
            bias *= (1.0 + (rand.nextDouble() - 0.5) * 0.01); // ±0.5% change
        }

        /**
         * Retrain model with all data
         */
        public void retrain(List<Map<String, Double>> featureVectors,
                            List<Map<PredictionTarget, Double>> targetValues) {
            // In a real system, this would be proper model training
            // For simulation, just adjust weights based on observed correlations

            // Find matching feature-target pairs
            List<Pair<Map<String, Double>, Double>> trainingData = new ArrayList<>();

            for (int i = 0; i < featureVectors.size(); i++) {
                if (i >= targetValues.size()) break;

                Map<String, Double> features = featureVectors.get(i);
                Map<PredictionTarget, Double> targets = targetValues.get(i);

                if (targets.containsKey(target)) {
                    trainingData.add(new Pair<>(features, targets.get(target)));
                }
            }

            if (trainingData.isEmpty()) {
                return; // No data for this target
            }

            // Simple gradient descent for linear regression
            double learningRate = 0.01;

            for (int epoch = 0; epoch < 10; epoch++) { // 10 epochs of training
                for (Pair<Map<String, Double>, Double> sample : trainingData) {
                    // Make prediction
                    double prediction = predict(sample.first);

                    // Calculate error
                    double error = prediction - sample.second;

                    // Update bias
                    bias -= learningRate * error;

                    // Update weights
                    for (Map.Entry<String, Double> feature : sample.first.entrySet()) {
                        if (weights.containsKey(feature.getKey())) {
                            double currentWeight = weights.get(feature.getKey());
                            double update = learningRate * error * feature.getValue();
                            weights.put(feature.getKey(), currentWeight - update);
                        }
                    }
                }
            }
        }
    }

    /**
     * Simple pair class for holding two values
     */
    private static class Pair<T, U> {
        public final T first;
        public final U second;

        public Pair(T first, U second) {
            this.first = first;
            this.second = second;
        }
    }
    /**
     * Helper class for task execution record
     */
    private static class TaskExecution {
        public final int taskId;
        public final int taskType;
        public final int deviceId;
        public final double submissionTime;
        public final double completionTime;
        public final double deadline;
        public final double energyConsumption;
        public final boolean successful;

        public TaskExecution(int taskId, int taskType, int deviceId,
                             double submissionTime, double completionTime,
                             double deadline, double energyConsumption, boolean successful) {
            this.taskId = taskId;
            this.taskType = taskType;
            this.deviceId = deviceId;
            this.submissionTime = submissionTime;
            this.completionTime = completionTime;
            this.deadline = deadline;
            this.energyConsumption = energyConsumption;
            this.successful = successful;
        }
    }

    /**
     * Helper class for task features
     */
    private static class TaskFeatures {
        public final int taskType;
        public final double taskSize;
        public final double inputDataSize;
        public final double outputDataSize;
        public final double delayRequirement;
        public final double wanBandwidth;
        public final double manDelay;
        public final double edgeUtilization;
        public final int selectedDevice;
        public final double decisionTime;

        public TaskFeatures(int taskType, double taskSize, double inputDataSize,
                            double outputDataSize, double delayRequirement,
                            double wanBandwidth, double manDelay, double edgeUtilization,
                            int selectedDevice, double decisionTime) {
            this.taskType = taskType;
            this.taskSize = taskSize;
            this.inputDataSize = inputDataSize;
            this.outputDataSize = outputDataSize;
            this.delayRequirement = delayRequirement;
            this.wanBandwidth = wanBandwidth;
            this.manDelay = manDelay;
            this.edgeUtilization = edgeUtilization;
            this.selectedDevice = selectedDevice;
            this.decisionTime = decisionTime;
        }
    }

    /**
     * Helper class for device statistics
     */
    private static class DeviceStats {
        private double avgUtilization = 0;
        private double peakUtilization = 0;
        private final List<Double> utilizationHistory = new ArrayList<>();
        private static final int MAX_HISTORY = 50;

        public void updateUtilization(double utilization) {
            avgUtilization = utilization;
            peakUtilization = Math.max(peakUtilization, utilization);

            utilizationHistory.add(utilization);
            if (utilizationHistory.size() > MAX_HISTORY) {
                utilizationHistory.remove(0);
            }
        }

        public double getAvgUtilization() {
            if (utilizationHistory.isEmpty()) {
                return 0;
            }

            double sum = 0;
            for (Double util : utilizationHistory) {
                sum += util;
            }

            return sum / utilizationHistory.size();
        }
    }

    /**
     * Helper class for network metrics
     */
    private static class NetworkMetrics {
        public final double wanBandwidth;
        public final double manDelay;

        public NetworkMetrics(double wanBandwidth, double manDelay) {
            this.wanBandwidth = wanBandwidth;
            this.manDelay = manDelay;
        }
    }
    /**
     * Callback interface for host finding
     */
    private interface HostIndexCallback {
        void onHostsFound(int[] indices);
    }


    @Override
    public void shutdownEntity() {

    }
}
