package edu.boun.edgecloudsim.applications.uarc;

import edu.boun.edgecloudsim.cloud_server.CloudVM;
import edu.boun.edgecloudsim.core.SimManager;
import edu.boun.edgecloudsim.core.SimSettings;
import edu.boun.edgecloudsim.edge_client.CpuUtilizationModel_Custom;
import edu.boun.edgecloudsim.edge_client.Task;
import edu.boun.edgecloudsim.edge_client.mobile_processing_unit.MobileVM;
import edu.boun.edgecloudsim.edge_orchestrator.EdgeOrchestrator;
import edu.boun.edgecloudsim.edge_server.EdgeHost;
import edu.boun.edgecloudsim.edge_server.EdgeVM;
import edu.boun.edgecloudsim.utils.Location;
import edu.boun.edgecloudsim.utils.SimLogger;
import org.cloudbus.cloudsim.Host;
import org.cloudbus.cloudsim.UtilizationModelFull;
import org.cloudbus.cloudsim.Vm;
import org.cloudbus.cloudsim.core.CloudSim;
import org.cloudbus.cloudsim.core.SimEvent;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

import static edu.boun.edgecloudsim.applications.uarc.AdaptiveMLEdgeOrchestrator.TASK_COMPLETED;
import static edu.boun.edgecloudsim.applications.uarc.AdaptiveMLEdgeOrchestrator.TASK_FAILED;

/**
 * Enhanced Edge Orchestrator using Fuzzy Q-Learning for adaptive decision making
 * This approach combines fuzzy logic with reinforcement learning to handle the
 * continuous state space of edge computing environments
 */
public class FuzzyQLEdgeOrchestrator extends EdgeOrchestrator {
    // Constants
    public static final double MAX_DATA_SIZE = 2500;
    public static final int METRIC_UPDATE_INTERVAL = 1; // seconds
    public static final int LEARNING_INTERVAL = 30; // seconds
    public static final int POLICY_UPDATE_INTERVAL = 300; // seconds

    // Fuzzy Q-Learning parameters
    private double alpha = 0.1; // Learning rate
    private double gamma = 0.9; // Discount factor
    private double epsilon = 0.1; // Exploration rate
    private double epsilonDecayRate = 0.9999; // Exploration decay rate
    private double minEpsilon = 0.01; // Minimum exploration rate

    // Metrics
    private double activeManTaskCount = 0;
    private double activeWanTaskCount = 0;
    private double totalSizeOfActiveManTasks = 0;
    private int numberOfHost; // used by load balancer
    private int totalTasks = 0;
    private int failedTasks = 0;
    private double edgeFailureRate = 0.0;
    private Map<Integer, Integer> taskTypeCounter = new HashMap<>();

    // Monitoring data structures
    private Map<Integer, DeviceStats> deviceStats = new ConcurrentHashMap<>();
    private Map<Integer, List<TaskExecution>> taskHistory = new ConcurrentHashMap<>();
    private Map<String, Double> averageQValues = new HashMap<>();

    // Fuzzy Q-Learning data structures
    private Map<String, Map<String, double[]>> fuzzyQTable = new HashMap<>();
    private Map<String, Map<String, Integer>> fuzzyRuleVisits = new HashMap<>();
    private List<FuzzyRule> fuzzyRules = new ArrayList<>();
    private Map<Integer, TaskMetadata> taskMetadata = new HashMap<>();

    // Device reliability tracking
    private Map<Integer, Double> deviceReliability = new HashMap<>();
    private Map<Integer, Double> deviceAvailability = new HashMap<>();

    // Capacity prediction
    private Map<Integer, Double> capacityTrend = new HashMap<>();
    private double[] edgeFailureHistory = new double[10]; // Last 10 intervals
    private int historyIndex = 0;
    private MultiLayerNetwork network = null;
    private final int EPOCH_SIZE = 75000;
    private double numberOfWlanOffloadedTask = 0;
    private double numberOfManOffloadedTask = 0;
    private double numberOfWanOffloadedTask = 0;
    /**
     * Constructor
     */
    public FuzzyQLEdgeOrchestrator(String orchestratorPolicy, String simScenario) {
        super(orchestratorPolicy, simScenario);
    }

    /**
     * Initialize the orchestrator
     */
    @Override
    public void initialize() {
        // Initialize device statistics
        initializeDeviceStats();

        // Initialize fuzzy rules
        initializeFuzzyRules();

        // Initialize fuzzy Q-table
        initializeFuzzyQTable();

        // Schedule periodic events
        //scheduleEvents();

        SimLogger.printLine("FuzzyQLEdgeOrchestrator initialized with fuzzy Q-learning based orchestration");
    }

    /**
     * Initialize device statistics
     */
    private void initializeDeviceStats() {
        // Get the number of edge hosts
        numberOfHost = SimSettings.getInstance().getNumOfEdgeHosts();

        // Initialize stats for all edge devices
        for (int i = 0; i < numberOfHost; i++) {
            deviceStats.put(i, new DeviceStats());
            taskHistory.put(i, new ArrayList<>());
            deviceReliability.put(i, 0.95); // Initial reliability estimate
            deviceAvailability.put(i, 0.95); // Initial availability estimate
            capacityTrend.put(i, 0.0); // Initial capacity trend
        }

        // Initialize stats for cloud
        deviceStats.put(SimSettings.CLOUD_DATACENTER_ID, new DeviceStats());
        taskHistory.put(SimSettings.CLOUD_DATACENTER_ID, new ArrayList<>());
        deviceReliability.put(SimSettings.CLOUD_DATACENTER_ID, 0.99); // Initial reliability estimate
        deviceAvailability.put(SimSettings.CLOUD_DATACENTER_ID, 0.99); // Initial availability estimate
        capacityTrend.put(SimSettings.CLOUD_DATACENTER_ID, 0.0); // Initial capacity trend

        // Initialize edge failure history
        Arrays.fill(edgeFailureHistory, 0.0);
    }

    /**
     * Initialize fuzzy rules for Q-learning
     */
    private void initializeFuzzyRules() {
        // Task size rules
        fuzzyRules.add(new FuzzyRule("task_size", "small", 0, 5000));
        fuzzyRules.add(new FuzzyRule("task_size", "medium", 4000, 15000));
        fuzzyRules.add(new FuzzyRule("task_size", "large", 12000, 30000));

        // WAN bandwidth rules
        fuzzyRules.add(new FuzzyRule("wan_bandwidth", "low", 0, 3));
        fuzzyRules.add(new FuzzyRule("wan_bandwidth", "medium", 2, 7));
        fuzzyRules.add(new FuzzyRule("wan_bandwidth", "high", 6, 15));

        // Edge utilization rules
        fuzzyRules.add(new FuzzyRule("edge_utilization", "low", 0, 40));
        fuzzyRules.add(new FuzzyRule("edge_utilization", "medium", 30, 70));
        fuzzyRules.add(new FuzzyRule("edge_utilization", "high", 60, 100));

        // Delay sensitivity rules
        fuzzyRules.add(new FuzzyRule("delay_sensitivity", "low", 0, 0.3));
        fuzzyRules.add(new FuzzyRule("delay_sensitivity", "medium", 0.2, 0.7));
        fuzzyRules.add(new FuzzyRule("delay_sensitivity", "high", 0.6, 1.0));

        // Data size rules
        fuzzyRules.add(new FuzzyRule("data_size", "small", 0, 500));
        fuzzyRules.add(new FuzzyRule("data_size", "medium", 400, 1500));
        fuzzyRules.add(new FuzzyRule("data_size", "large", 1200, 3000));

        // Edge capacity rules
        fuzzyRules.add(new FuzzyRule("edge_capacity", "low", 0, 30));
        fuzzyRules.add(new FuzzyRule("edge_capacity", "medium", 20, 70));
        fuzzyRules.add(new FuzzyRule("edge_capacity", "high", 60, 100));

        // Edge failure rate rules
        fuzzyRules.add(new FuzzyRule("edge_failure_rate", "low", 0, 0.05));
        fuzzyRules.add(new FuzzyRule("edge_failure_rate", "medium", 0.03, 0.15));
        fuzzyRules.add(new FuzzyRule("edge_failure_rate", "high", 0.1, 0.5));
    }

    /**
     * Initialize fuzzy Q-table
     */
    private void initializeFuzzyQTable() {
        // Define state combinations we care about
        List<String[]> stateSpaceCombinations = new ArrayList<>();

        // Task size, edge utilization, edge capacity
        stateSpaceCombinations.add(new String[]{"task_size", "edge_utilization", "edge_capacity"});

        // Task size, wan bandwidth, delay sensitivity
        stateSpaceCombinations.add(new String[]{"task_size", "wan_bandwidth", "delay_sensitivity"});

        // Edge utilization, edge failure rate, data size
        stateSpaceCombinations.add(new String[]{"edge_utilization", "edge_failure_rate", "data_size"});

        // Generate fuzzy state combinations
        for (String[] stateCombo : stateSpaceCombinations) {
            generateFuzzyStates(stateCombo);
        }
    }

    /**
     * Generate fuzzy states for the given state variables
     */
    private void generateFuzzyStates(String[] stateVariables) {
        List<List<String>> termCombinations = new ArrayList<>();

        // For each state variable, get its possible linguistic terms
        for (String variable : stateVariables) {
            List<String> terms = new ArrayList<>();
            for (FuzzyRule rule : fuzzyRules) {
                if (rule.getVariable().equals(variable) && !terms.contains(rule.getTerm())) {
                    terms.add(rule.getTerm());
                }
            }
            termCombinations.add(terms);
        }

        // Generate all combinations of terms
        List<List<String>> allCombinations = generateCombinations(termCombinations, 0, new ArrayList<>());

        // Create Q-table entries for each combination
        for (List<String> combo : allCombinations) {
            String stateKey = createStateKey(stateVariables, combo);
            Map<String, double[]> actions = new HashMap<>();
            Map<String, Integer> visits = new HashMap<>();

            // Initialize Q-values for each action (cloud, edge, mobile)
            actions.put("cloud", new double[]{0.0});
            actions.put("edge", new double[]{0.0});
            actions.put("mobile", new double[]{0.0});

            // Initialize visit counts
            visits.put("cloud", 0);
            visits.put("edge", 0);
            visits.put("mobile", 0);

            fuzzyQTable.put(stateKey, actions);
            fuzzyRuleVisits.put(stateKey, visits);
        }
    }

    /**
     * Generate all combinations of terms
     */
    private List<List<String>> generateCombinations(List<List<String>> lists, int index, List<String> current) {
        List<List<String>> result = new ArrayList<>();

        if (index == lists.size()) {
            result.add(new ArrayList<>(current));
            return result;
        }

        for (String term : lists.get(index)) {
            current.add(term);
            result.addAll(generateCombinations(lists, index + 1, current));
            current.remove(current.size() - 1);
        }

        return result;
    }

    /**
     * Create a state key from variables and terms
     */
    private String createStateKey(String[] variables, List<String> terms) {
        StringBuilder key = new StringBuilder();
        for (int i = 0; i < variables.length; i++) {
            if (i > 0) {
                key.append("_");
            }
            key.append(variables[i]).append("-").append(terms.get(i));
        }
        return key.toString();
    }

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
            // Track task type statistics
            trackTaskType(task.getTaskType());

            // Find best edge hosts
            int[] hostIndices = findBestEdgeHosts(task);
            int nearestEdgeHostIndex = hostIndices[0];
            double nearestEdgeUtilization = hostIndices[1];
            int bestRemoteEdgeHostIndex = hostIndices[2];
            Task dummyTask = new Task(0, 0, 0, 0, 128, 128, new UtilizationModelFull(), new UtilizationModelFull(), new UtilizationModelFull());

            double wanDelay = SimManager.getInstance().getNetworkModel().getUploadDelay(task.getMobileDeviceId(),
                    SimSettings.CLOUD_DATACENTER_ID, dummyTask /* 1 Mbit */);
            // Get network metrics
            NetworkMetrics networkMetrics = getNetworkMetrics(task);
            double wanBW = (wanDelay == 0) ? 0 : (1 / wanDelay); /* Mbps */
            // Get system metrics
            double edgeUtilization = SimManager.getInstance().getEdgeServerManager().getAvgUtilization();
            double edgeCapacity = estimateEdgeCapacity(nearestEdgeHostIndex, task);

            // Store task metadata for learning
            storeTaskMetadata(task, networkMetrics, edgeUtilization, edgeCapacity, nearestEdgeHostIndex);
            double utilization = edgeUtilization;
            // Make decision based on policy
            switch (policy) {
                case "NETWORK_BASED" -> {
                    if(wanBW > 6)
                        result = SimSettings.CLOUD_DATACENTER_ID;
                    else
                        result = SimSettings.GENERIC_EDGE_DEVICE_ID;
                }
                case "UTILIZATION_BASED" -> {

                    if(utilization > 80)
                        result = SimSettings.CLOUD_DATACENTER_ID;
                    else
                        result = SimSettings.GENERIC_EDGE_DEVICE_ID;
                }
                case "HYBRID" ->{
                    if(wanBW > 6 && utilization > 80)
                        result = SimSettings.CLOUD_DATACENTER_ID;
                    else
                        result = SimSettings.GENERIC_EDGE_DEVICE_ID;
                }
                case "FUZZY_Q_LEARNING" -> {
                    // Use fuzzy Q-learning to decide
                    result = fuzzyQLearningDecision(task, networkMetrics, nearestEdgeHostIndex,
                            edgeUtilization, edgeCapacity);
                    //SimLogger.printLine("Result: "+result);
                }
                case "ADAPTIVE_FUZZY" -> {
                    // Use adaptive fuzzy system with dynamic rule adjustments
                    result = adaptiveFuzzyDecision(task, networkMetrics, nearestEdgeHostIndex,
                            edgeUtilization, edgeCapacity);
                }
                case "DDEP" ->{
                    RLState rlState = getTaskFeatures(task);
                    INDArray output = network.output(rlState.getState());
                    result = output.argMax().getInt();
                    if (result == 14){
                        numberOfWanOffloadedTask++;
                    }
                    else if(task.getSubmittedLocation().getServingWlanId() == result){
                        numberOfWlanOffloadedTask++;
                    }
                    else{
                        numberOfManOffloadedTask++;
                    }
                }
                default -> {
                    SimLogger.printLine("Unknown edge orchestrator policy! Using default Fuzzy Q-Learning policy.");
                    result = SimSettings.GENERIC_EDGE_DEVICE_ID;
                }
            }
        }
        else {
            SimLogger.printLine("Unknown simulation scenario! Terminating simulation...");
            System.exit(0);
        }

        return result;
    }

    private RLState getTaskFeatures(Task task) {
        Task dummyTask = new Task(0, 0, 0, 0, 128, 128, new UtilizationModelFull(), new UtilizationModelFull(), new UtilizationModelFull());
        UarcNetworkModel networkModel = (UarcNetworkModel) SimManager.getInstance().getNetworkModel();
        RLState currentState = new RLState();
        ArrayList<Double> edgeCapacities = new ArrayList<>();

        int numberOfHost = SimSettings.getInstance().getNumOfEdgeHosts();

        double wanDelay = networkModel.getUploadDelay(task.getMobileDeviceId(),
                SimSettings.CLOUD_DATACENTER_ID, dummyTask /* 1 Mbit */);

        double wanBW = (wanDelay == 0) ? 0 : (1 / wanDelay); /* Mbps */

        currentState.setWanBw(wanBW/20.21873);


        double manDelayF = networkModel.getUploadDelayForTraining(SimSettings.GENERIC_EDGE_DEVICE_ID,
                SimSettings.GENERIC_EDGE_DEVICE_ID, dummyTask );

        double manBW = (manDelayF == 0) ? 0 : (1 / manDelayF);


        double manDelay = getManDelayForAgent();
        currentState.setManDelay(manDelay);

        double taskRequiredCapacity = ((CpuUtilizationModel_Custom)task.getUtilizationModelCpu()).predictUtilization(SimSettings.VM_TYPES.EDGE_VM);
        currentState.setTaskReqCapacity(taskRequiredCapacity/800);

        int wlanID = task.getSubmittedLocation().getServingWlanId();
        currentState.setWlanID((double)wlanID / (numberOfHost - 1));

        int nearestEdgeHostId = 0;


        for(int hostIndex=0; hostIndex<numberOfHost; hostIndex++){
            List<EdgeVM> vmArray = SimManager.getInstance().getEdgeServerManager().getVmList(hostIndex);
            EdgeHost host = (EdgeHost)(vmArray.get(0).getHost()); //all VMs have the same host

            double totalUtilizationForEdgeServer=0;
            for(int vmIndex=0; vmIndex<vmArray.size(); vmIndex++){
                totalUtilizationForEdgeServer += vmArray.get(vmIndex).getCloudletScheduler().getTotalUtilizationOfCpu(CloudSim.clock());
            }

            double totalCapacity = 100 * vmArray.size();
            double averageCapacity = (totalCapacity - totalUtilizationForEdgeServer)  / vmArray.size();
            double normalizedCapacity = averageCapacity / 100;

            if (normalizedCapacity < 0){
                normalizedCapacity = 0;
            }
            edgeCapacities.add(normalizedCapacity);

            if (host.getLocation().getServingWlanId() == task.getSubmittedLocation().getServingWlanId()){
                nearestEdgeHostId = hostIndex;
            }

        }

        currentState.setAvailVmInEdge(edgeCapacities);
        currentState.setNearestEdgeHostId((double)nearestEdgeHostId / numberOfHost);

        double delay_sensitivity = SimSettings.getInstance().getTaskLookUpTable()[task.getTaskType()][12];

        currentState.setDelaySensitivity(delay_sensitivity);

        currentState.setNumberOfWlanOffloadedTask(numberOfWlanOffloadedTask/ EPOCH_SIZE);
        currentState.setNumberOfManOffloadedTask(numberOfManOffloadedTask/ EPOCH_SIZE);
        currentState.setNumberOfWanOffloadedTask(numberOfWanOffloadedTask/ EPOCH_SIZE);
        currentState.setActiveManTaskCount(activeManTaskCount/25);
        currentState.setActiveWanTaskCount(activeWanTaskCount/25);
        return  currentState;
    }


    /**
     * Track task type statistics
     */
    private void trackTaskType(int taskType) {
        taskTypeCounter.put(taskType, taskTypeCounter.getOrDefault(taskType, 0) + 1);
    }

    /**
     * Store task metadata for learning
     */
    private void storeTaskMetadata(Task task, NetworkMetrics networkMetrics,
                                   double edgeUtilization, double edgeCapacity, int edgeHostIndex) {
        TaskMetadata metadata = new TaskMetadata(
                CloudSim.clock(),                    // Current time
                task.getTaskType(),                  // Task type
                task.getCloudletLength(),            // Task size
                task.getCloudletFileSize(),          // Input data size
                task.getCloudletOutputSize(),        // Output data size
                SimSettings.getInstance().getTaskLookUpTable()[task.getTaskType()][12], // Delay sensitivity
                networkMetrics.wanBandwidth,         // WAN bandwidth
                networkMetrics.manDelay,             // MAN delay
                edgeUtilization,                     // Edge utilization
                edgeCapacity,                        // Edge capacity
                edgeFailureRate,                     // Current edge failure rate
                edgeHostIndex                        // Target edge host
        );

        taskMetadata.put(task.getCloudletId(), metadata);
    }

    /**
     * Main fuzzy Q-learning decision algorithm. Uses Exploitation method
     */
    private int fuzzyQLearningDecision(Task task, NetworkMetrics networkMetrics,
                                       int nearestEdgeHostIndex, double edgeUtilization, double edgeCapacity) {

        double taskSize = task.getCloudletLength();
        double dataSize = task.getCloudletFileSize() + task.getCloudletOutputSize();
        double delaySensitivity = SimSettings.getInstance().getTaskLookUpTable()[task.getTaskType()][12];
        double normalizedDelaySensitivity = delaySensitivity > 0 ?
                Math.min(1.0, 1.0 / delaySensitivity) : 0.1;

        // Handle capacity constraints explicitly first
        boolean edgeHasCapacity = hasEdgeCapacity(task, nearestEdgeHostIndex);
        if (!edgeHasCapacity) {
            return SimSettings.CLOUD_DATACENTER_ID;
        }

        // Determine fuzzy state
        Map<String, Map<String, Double>> fuzzyInputs = new HashMap<>();
        fuzzyInputs.put("task_size", getFuzzyMemberships("task_size", taskSize));
        fuzzyInputs.put("wan_bandwidth", getFuzzyMemberships("wan_bandwidth", networkMetrics.wanBandwidth));
        fuzzyInputs.put("edge_utilization", getFuzzyMemberships("edge_utilization", edgeUtilization));
        fuzzyInputs.put("delay_sensitivity", getFuzzyMemberships("delay_sensitivity", normalizedDelaySensitivity));
        fuzzyInputs.put("data_size", getFuzzyMemberships("data_size", dataSize));
        fuzzyInputs.put("edge_capacity", getFuzzyMemberships("edge_capacity", edgeCapacity));
        fuzzyInputs.put("edge_failure_rate", getFuzzyMemberships("edge_failure_rate", edgeFailureRate));

        // Exploitation: choose best action based on fuzzy Q-values
        String[] relevantStates = {
                // State combination 1: Task size, edge utilization, edge capacity
                "task_size-" + getMaxFuzzyTerm(fuzzyInputs.get("task_size")) +
                        "_edge_utilization-" + getMaxFuzzyTerm(fuzzyInputs.get("edge_utilization")) +
                        "_edge_capacity-" + getMaxFuzzyTerm(fuzzyInputs.get("edge_capacity")),

                // State combination 2: Task size, wan bandwidth, delay sensitivity
                "task_size-" + getMaxFuzzyTerm(fuzzyInputs.get("task_size")) +
                        "_wan_bandwidth-" + getMaxFuzzyTerm(fuzzyInputs.get("wan_bandwidth")) +
                        "_delay_sensitivity-" + getMaxFuzzyTerm(fuzzyInputs.get("delay_sensitivity")),

                // State combination 3: Edge utilization, edge failure rate, data size
                "edge_utilization-" + getMaxFuzzyTerm(fuzzyInputs.get("edge_utilization")) +
                        "_edge_failure_rate-" + getMaxFuzzyTerm(fuzzyInputs.get("edge_failure_rate")) +
                        "_data_size-" + getMaxFuzzyTerm(fuzzyInputs.get("data_size"))
        };


//            // Calculate weighted Q-values based on fuzzy rules
            double cloudQValue = 0, edgeQValue = 0, mobileQValue = 0;
            double totalWeight = 0;

            for (String state : relevantStates) {
                if (fuzzyQTable.containsKey(state)) {
                    Map<String, double[]> actions = fuzzyQTable.get(state);
                    int cloudVisits = fuzzyRuleVisits.getOrDefault(state, new HashMap<>()).getOrDefault("cloud", 0);
                    int edgeVisits = fuzzyRuleVisits.getOrDefault(state, new HashMap<>()).getOrDefault("edge", 0);
                    int mobileVisits = fuzzyRuleVisits.getOrDefault(state, new HashMap<>()).getOrDefault("mobile", 0);

                    // Apply UCB1 exploration bonus to less-visited actions
                    double explorationBonus = Math.sqrt(2 * Math.log(totalTasks + 1));
                    double cloudBonus = cloudVisits > 0 ? explorationBonus / Math.sqrt(cloudVisits) : 10.0;
                    double edgeBonus = edgeVisits > 0 ? explorationBonus / Math.sqrt(edgeVisits) : 10.0;
                    double mobileBonus = mobileVisits > 0 ? explorationBonus / Math.sqrt(mobileVisits) : 10.0;

                    // Adjusted Q-values with exploration bonus
                    cloudQValue += actions.get("cloud")[0] + cloudBonus;
                    edgeQValue += actions.get("edge")[0] + edgeBonus;
                    mobileQValue += actions.get("mobile")[0] + mobileBonus;
                    totalWeight++;
                }
            }
//
            // Normalize if we found any relevant states
            if (totalWeight > 0) {
                cloudQValue /= totalWeight;
                edgeQValue /= totalWeight;
                mobileQValue /= totalWeight;
            }

            // For monitoring purposes
            recordAverageQValues(cloudQValue, edgeQValue, mobileQValue);

            // Choose action with highest Q-value
            if (edgeHasCapacity && edgeQValue >= cloudQValue && edgeQValue >= mobileQValue) {
                //SimLogger.printLine("Edge has capacity: "+edgeHasCapacity+" Edge host id: "+nearestEdgeHostIndex);
                return nearestEdgeHostIndex;
            } else if (mobileQValue >= cloudQValue) {
                //SimLogger.printLine("Edge has capacity: "+edgeHasCapacity+" Mobile host id: "+task.getMobileDeviceId());
                return task.getMobileDeviceId();
            } else {
                return SimSettings.CLOUD_DATACENTER_ID;
            }

    }

    /**
     * Record average Q-values for monitoring
     */
    private void recordAverageQValues(double cloudQValue, double edgeQValue, double mobileQValue) {
        // Simple exponential moving average (EMA)
        double alpha = 0.1; // Smoothing factor

        averageQValues.put("cloud", averageQValues.getOrDefault("cloud", 0.0) * (1 - alpha) + cloudQValue * alpha);
        averageQValues.put("edge", averageQValues.getOrDefault("edge", 0.0) * (1 - alpha) + edgeQValue * alpha);
        averageQValues.put("mobile", averageQValues.getOrDefault("mobile", 0.0) * (1 - alpha) + mobileQValue * alpha);
    }

    /**
     * Adaptive fuzzy decision with dynamic rule adjustments
     */
    private int adaptiveFuzzyDecision(Task task, NetworkMetrics networkMetrics,
                                      int nearestEdgeHostIndex, double edgeUtilization, double edgeCapacity) {
        // Start with basic fuzzy Q-learning decision
        int initialDecision = fuzzyQLearningDecision(task, networkMetrics, nearestEdgeHostIndex,
                edgeUtilization, edgeCapacity);

        // If edge is chosen but we're seeing high failure rates, override for safety
        if (initialDecision == nearestEdgeHostIndex) {
            // Check edge failure trend
            double failureTrend = calculateEdgeFailureTrend();

            if (failureTrend > 0.1) { // Increasing failure trend
                // As failures increase, become more conservative
                if (edgeFailureRate > 0.05 || edgeCapacity < 30 || edgeUtilization > 80) {
                    return SimSettings.CLOUD_DATACENTER_ID; // Redirect to cloud
                }
            }
        }
        return initialDecision;
    }

    /**
     * Calculate edge failure trend (positive means increasing failures)
     */
    private double calculateEdgeFailureTrend() {
        if (historyIndex < 5) return 0; // Not enough history

        // Calculate average of first half vs second half of history
        double firstHalfAvg = 0;
        double secondHalfAvg = 0;

        for (int i = 0; i < 5; i++) {
            firstHalfAvg += edgeFailureHistory[i];
        }
        for (int i = 5; i < 10; i++) {
            secondHalfAvg += edgeFailureHistory[i];
        }

        firstHalfAvg /= 5;
        secondHalfAvg /= 5;

        // Return difference - positive means increasing failure rate
        return secondHalfAvg - firstHalfAvg;
    }

    /**
     * Capacity-aware decision making to reduce VM capacity failures
     */
    private int capacityAwareDecision(Task task, NetworkMetrics networkMetrics,
                                      int nearestEdgeHostIndex, double edgeUtilization, double edgeCapacity) {
        // Extract task requirements
        double taskSize = task.getCloudletLength();
        double requiredCapacity = ((CpuUtilizationModel_Custom)task.getUtilizationModelCpu()).predictUtilization(SimSettings.VM_TYPES.EDGE_VM);

        // Safety threshold for capacity
        double safetyMargin = 1.5; // 50% extra capacity needed as safety margin
        double adjustedRequiredCapacity = requiredCapacity * safetyMargin;

        // If capacity is tight, prefer cloud
        if (edgeCapacity < adjustedRequiredCapacity || edgeFailureRate > 0.05) {
            return SimSettings.CLOUD_DATACENTER_ID;
        }

        // For large tasks that need a lot of capacity, prefer cloud
        if (taskSize > 10000 && edgeUtilization > 60) {
            return SimSettings.CLOUD_DATACENTER_ID;
        }

        // For smaller tasks with good network and available capacity, use edge
        if (taskSize < 5000 && edgeCapacity > adjustedRequiredCapacity * 2 &&
                networkMetrics.manDelay < 0.5) {
            return nearestEdgeHostIndex;
        }

        // Default to fuzzy Q-learning for other cases
        return fuzzyQLearningDecision(task, networkMetrics, nearestEdgeHostIndex,
                edgeUtilization, edgeCapacity);
    }
//
//    /**
//     * Hybrid approach combining fuzzy logic with traditional Q-learning
//     */
//    private int hybridFuzzyRLDecision(Task task, NetworkMetrics networkMetrics,
//                                      int nearestEdgeHostIndex, double edgeUtilization, double edgeCapacity) {
//        // Extract key features
//        double taskSize = task.getCloudletLength();
//        double dataSize = task.getCloudletFileSize() + task.getCloudletOutputSize();
//        double delaySensitivity = SimSettings.getInstance().getTaskLookUpTable()[task.getTaskType()][12];
//
//        // Create deterministic rules for extreme cases
//
//        // Rule 1: Critical task with tight delay requirements goes to nearest resource
//        if (delaySensitivity < 1.0 && dataSize < 500) {
//            if (edgeCapacity > 30 && edgeFailureRate < 0.1) {
//                return nearestEdgeHostIndex;
//            } else {
//                return SimSettings.CLOUD_DATACENTER_ID;
//            }
//        }
//
//        // Rule 2: Large computation with loose delay requirements goes to cloud
//        if (taskSize > 15000 && (delaySensitivity > 5.0 || delaySensitivity == 0)) {
//            return SimSettings.CLOUD_DATACENTER_ID;
//        }
//
//        // Rule 3: Edge is overloaded or showing high failure rate
//        if (edgeUtilization > 80 || edgeFailureRate > 0.1 || edgeCapacity < 20) {
//            return SimSettings.CLOUD_DATACENTER_ID;
//        }
//
//        // For other cases, use fuzzy Q-learning
//        return fuzzyQLearningDecision(task, networkMetrics, nearestEdgeHostIndex,
//                edgeUtilization, edgeCapacity);
//    }

//    /**
//     * UARC implementation with fuzzy Q-learning integration
//     */
//    private int uarcFuzzyDecision(Task task, List<MobileVM> mobileVMList, NetworkMetrics networkMetrics,
//                                  int nearestEdgeHostIndex, double edgeUtilization, double edgeCapacity) {
//        // Extract task requirements
//        double taskSize = task.getCloudletLength();
//        double inputDataSize = task.getCloudletFileSize();
//        double outputDataSize = task.getCloudletOutputSize();
//        double delaySensitivity = SimSettings.getInstance().getTaskLookUpTable()[task.getTaskType()][12];
//
//        // Calculate utility scores
//        double mobileScore = calculateMobileUtility(mobileVMList, task, delaySensitivity);
//        double edgeScore = calculateEdgeUtility(task, nearestEdgeHostIndex, edgeUtilization,
//                edgeCapacity, delaySensitivity);
//        double cloudScore = calculateCloudUtility(task, networkMetrics.wanBandwidth, delaySensitivity);
//
//        // Fuzzy Q-learning influence
//        double[] qInfluence = getFuzzyQLInfluence(task, networkMetrics, nearestEdgeHostIndex,
//                edgeUtilization, edgeCapacity);
//
//        // Combine UARC utility with Q-learning influence
//        double combinedMobileScore = mobileScore * (1.0 - 0.3) + qInfluence[2] * 0.3;
//        double combinedEdgeScore = edgeScore * (1.0 - 0.3) + qInfluence[1] * 0.3;
//        double combinedCloudScore = cloudScore * (1.0 - 0.3) + qInfluence[0] * 0.3;
//
//        // Select best option
//        if (combinedMobileScore >= combinedEdgeScore && combinedMobileScore >= combinedCloudScore &&
//                mobileVMList != null && !mobileVMList.isEmpty()) {
//            return task.getMobileDeviceId();
//        } else if (combinedEdgeScore >= combinedCloudScore && hasEdgeCapacity(task, nearestEdgeHostIndex)) {
//            return nearestEdgeHostIndex;
//        } else {
//            return SimSettings.CLOUD_DATACENTER_ID;
//        }
//    }

    /**
     * Get fuzzy Q-learning influence on decision
     */
    private double[] getFuzzyQLInfluence(Task task, NetworkMetrics networkMetrics,
                                         int nearestEdgeHostIndex, double edgeUtilization, double edgeCapacity) {
        // Extract key features
        double taskSize = task.getCloudletLength();
        double dataSize = task.getCloudletFileSize() + task.getCloudletOutputSize();
        double delaySensitivity = SimSettings.getInstance().getTaskLookUpTable()[task.getTaskType()][12];
        double normalizedDelaySensitivity = delaySensitivity > 0 ?
                Math.min(1.0, 1.0 / delaySensitivity) : 0.1;

        // Get fuzzy memberships
        Map<String, Double> taskSizeMemberships = getFuzzyMemberships("task_size", taskSize);
        Map<String, Double> wanBwMemberships = getFuzzyMemberships("wan_bandwidth", networkMetrics.wanBandwidth);
        Map<String, Double> edgeUtilMemberships = getFuzzyMemberships("edge_utilization", edgeUtilization);
        Map<String, Double> delaySensMemberships = getFuzzyMemberships("delay_sensitivity", normalizedDelaySensitivity);
        Map<String, Double> dataSizeMemberships = getFuzzyMemberships("data_size", dataSize);
        Map<String, Double> edgeCapMemberships = getFuzzyMemberships("edge_capacity", edgeCapacity);
        Map<String, Double> failRateMemberships = getFuzzyMemberships("edge_failure_rate", edgeFailureRate);

        // Create relevant state keys
        String[] relevantStates = {
                // Task size, edge utilization, edge capacity
                "task_size-" + getMaxFuzzyTerm(taskSizeMemberships) +
                        "_edge_utilization-" + getMaxFuzzyTerm(edgeUtilMemberships) +
                        "_edge_capacity-" + getMaxFuzzyTerm(edgeCapMemberships),

                // Task size, wan bandwidth, delay sensitivity
                "task_size-" + getMaxFuzzyTerm(taskSizeMemberships) +
                        "_wan_bandwidth-" + getMaxFuzzyTerm(wanBwMemberships) +
                        "_delay_sensitivity-" + getMaxFuzzyTerm(delaySensMemberships),

                // Edge utilization, edge failure rate, data size
                "edge_utilization-" + getMaxFuzzyTerm(edgeUtilMemberships) +
                        "_edge_failure_rate-" + getMaxFuzzyTerm(failRateMemberships) +
                        "_data_size-" + getMaxFuzzyTerm(dataSizeMemberships)
        };

        // Calculate Q-value influence
        double cloudQValue = 0, edgeQValue = 0, mobileQValue = 0;
        double totalWeight = 0;

        for (String state : relevantStates) {
            if (fuzzyQTable.containsKey(state)) {
                Map<String, double[]> actions = fuzzyQTable.get(state);
                cloudQValue += actions.get("cloud")[0];
                edgeQValue += actions.get("edge")[0];
                mobileQValue += actions.get("mobile")[0];
                totalWeight++;
            }
        }

        // Normalize
        if (totalWeight > 0) {
            cloudQValue /= totalWeight;
            edgeQValue /= totalWeight;
            mobileQValue /= totalWeight;
        }

        // Normalize to 0-1 range for influence
        double maxQ = Math.max(cloudQValue, Math.max(edgeQValue, mobileQValue));
        double minQ = Math.min(cloudQValue, Math.min(edgeQValue, mobileQValue));
        double range = maxQ - minQ;

        if (range > 0) {
            cloudQValue = (cloudQValue - minQ) / range;
            edgeQValue = (edgeQValue - minQ) / range;
            mobileQValue = (mobileQValue - minQ) / range;
        } else {
            // Equal values
            cloudQValue = edgeQValue = mobileQValue = 0.33;
        }

        return new double[]{cloudQValue, edgeQValue, mobileQValue};
    }

    /**
     * Estimate edge capacity
     */
    private double estimateEdgeCapacity(int hostIndex, Task task) {
        double maxAvailableCapacity = 0;

        List<EdgeVM> vmArray = SimManager.getInstance().getEdgeServerManager().getVmList(hostIndex);
        for (EdgeVM vm : vmArray) {
            double availableCapacity = 100.0 - vm.getCloudletScheduler().getTotalUtilizationOfCpu(CloudSim.clock());
            maxAvailableCapacity = Math.max(maxAvailableCapacity, availableCapacity);
        }

        return maxAvailableCapacity;
    }

    /**
     * Check if edge has capacity for task
     */
    private boolean hasEdgeCapacity(Task task, int hostIndex) {
        List<EdgeVM> vmArray = SimManager.getInstance().getEdgeServerManager().getVmList(hostIndex);
        double requiredCapacity = ((CpuUtilizationModel_Custom)task.getUtilizationModelCpu()).predictUtilization(SimSettings.VM_TYPES.EDGE_VM);

        // Add safety margin to avoid being too close to capacity limit
        double requiredWithMargin = requiredCapacity * 1.1; // 20% safety margin

        for (EdgeVM vm : vmArray) {
            double availableCapacity = 100.0 - vm.getCloudletScheduler().getTotalUtilizationOfCpu(CloudSim.clock());
            if (requiredWithMargin <= availableCapacity) {
                return true;
            }
        }

        return false;
    }

    /**
     * Get fuzzy membership values for a variable and value
     */
    private Map<String, Double> getFuzzyMemberships(String variable, double value) {
        Map<String, Double> memberships = new HashMap<>();

        for (FuzzyRule rule : fuzzyRules) {
            if (rule.getVariable().equals(variable)) {
                double membership = rule.getMembership(value);
                if (membership > 0) {
                    memberships.put(rule.getTerm(), membership);
                }
            }
        }

        return memberships;
    }

    /**
     * Get term with maximum membership
     */
    private String getMaxFuzzyTerm(Map<String, Double> memberships) {
        String maxTerm = "";
        double maxValue = 0;

        for (Map.Entry<String, Double> entry : memberships.entrySet()) {
            if (entry.getValue() > maxValue) {
                maxValue = entry.getValue();
                maxTerm = entry.getKey();
            }
        }

        return maxTerm;
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
     * Find best edge hosts
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

        return new int[]{nearestEdgeHostIndex, (int)Math.round(nearestEdgeUtilization), bestRemoteEdgeHostIndex};
    }

    /**
     * Get VM to offload the task to
     */
    @Override
    public Vm getVmToOffload(Task task, int deviceId) {
        Vm selectedVM = null;

        if (deviceId == SimSettings.CLOUD_DATACENTER_ID) {
            // Cloud VM selection
            selectedVM = selectCloudVM(task);
        } else if (deviceId == SimSettings.GENERIC_EDGE_DEVICE_ID || deviceId < numberOfHost) {
            //SimLogger.printLine("Device id: "+deviceId+" genereic Id: "+SimSettings.GENERIC_EDGE_DEVICE_ID+" number of host: "+numberOfHost);
            // Edge VM selection (either generic or specific host)
            selectedVM = selectEdgeVM(task, deviceId);
        } else {
            // Mobile device - use task's own mobile device
            List<MobileVM> vmArray = SimManager.getInstance().getMobileServerManager().getVmList(task.getMobileDeviceId());
            if (!vmArray.isEmpty()) {
                selectedVM = vmArray.get(0);
            }
        }

        return selectedVM;
    }

    /**
     * Select VM on cloud
     */
    private Vm selectCloudVM(Task task) {
        Vm selectedVM = null;
        double bestFitness = Double.MAX_VALUE; // Lower is better

        List<Host> hostList = SimManager.getInstance().getCloudServerManager().getDatacenter().getHostList();
        for (int hostIndex = 0; hostIndex < hostList.size(); hostIndex++) {
            List<CloudVM> vmArray = SimManager.getInstance().getCloudServerManager().getVmList(hostIndex);
            for (CloudVM vm : vmArray) {
                double requiredCapacity = ((CpuUtilizationModel_Custom)task.getUtilizationModelCpu()).predictUtilization(vm.getVmType());
                double availableCapacity = 100.0 - vm.getCloudletScheduler().getTotalUtilizationOfCpu(CloudSim.clock());

                if (requiredCapacity <= availableCapacity) {
                    // Calculate fitness score for this VM
                    double fitness = calculateVMFitness(vm, task, requiredCapacity, availableCapacity);

                    if (fitness < bestFitness) {
                        selectedVM = vm;
                        bestFitness = fitness;
                    }
                }
            }
        }

        return selectedVM;
    }

    /**
     * Select VM on edge
     */
    private Vm selectEdgeVM(Task task, int deviceId) {
        if (deviceId == SimSettings.GENERIC_EDGE_DEVICE_ID) {
            // Find best VM across all edge hosts
            return selectBestEdgeVM(task);
        } else {
            // Select from specific host
            return selectVMFromSpecificHost(task, deviceId);
        }
    }

    /**
     * Select best VM across all edge hosts
     */
    private Vm selectBestEdgeVM(Task task) {
        Vm selectedVM = null;
        double bestFitness = Double.MAX_VALUE; // Lower is better

        for (int hostIndex = 0; hostIndex < numberOfHost; hostIndex++) {
            List<EdgeVM> vmArray = SimManager.getInstance().getEdgeServerManager().getVmList(hostIndex);
            for (EdgeVM vm : vmArray) {
                double requiredCapacity = ((CpuUtilizationModel_Custom)task.getUtilizationModelCpu()).predictUtilization(vm.getVmType());
                double availableCapacity = 100.0 - vm.getCloudletScheduler().getTotalUtilizationOfCpu(CloudSim.clock());

                if (requiredCapacity <= availableCapacity) {
                    // Calculate fitness score for this VM
                    double fitness = calculateVMFitness(vm, task, requiredCapacity, availableCapacity);

                    // Add proximity factor
                    EdgeHost host = (EdgeHost)vm.getHost();
                    double distance = calculateDistance(task.getSubmittedLocation(), host.getLocation());
                    double proximityFactor = distance / 1000.0; // Normalize

                    // Add proximity to fitness (lower is better)
                    fitness += proximityFactor * 0.3;

                    if (fitness < bestFitness) {
                        selectedVM = vm;
                        bestFitness = fitness;
                    }
                }
            }
        }

        return selectedVM;
    }

    /**
     * Select VM from a specific host
     */
    private Vm selectVMFromSpecificHost(Task task, int hostIndex) {
        Vm selectedVM = null;
        double bestFitness = Double.MAX_VALUE; // Lower is better

        List<EdgeVM> vmArray = SimManager.getInstance().getEdgeServerManager().getVmList(hostIndex);
        for (EdgeVM vm : vmArray) {
            double requiredCapacity = ((CpuUtilizationModel_Custom)task.getUtilizationModelCpu()).predictUtilization(vm.getVmType());
            double availableCapacity = 100.0 - vm.getCloudletScheduler().getTotalUtilizationOfCpu(CloudSim.clock());

            if (requiredCapacity <= availableCapacity) {
                // Calculate fitness score for this VM
                double fitness = calculateVMFitness(vm, task, requiredCapacity, availableCapacity);

                if (fitness < bestFitness) {
                    selectedVM = vm;
                    bestFitness = fitness;
                }
            }
        }
        assert selectedVM != null;
        //SimLogger.printLine("Selected Id: "+selectedVM.getId());
        return selectedVM;
    }

    /**
     * Calculate VM fitness score for task assignment (lower is better)
     */
    private double calculateVMFitness(Vm vm, Task task, double requiredCapacity, double availableCapacity) {
        // Calculate capacity fitness (prefer VMs with enough but not too much capacity)
        double capacityFitness;

        // Prefer VMs with 30-50% more capacity than required
        double idealExtraCapacity = requiredCapacity * 0.4; // 40% extra capacity
        double idealCapacity = requiredCapacity + idealExtraCapacity;

        if (availableCapacity < requiredCapacity) {
            capacityFitness = 10.0; // Very bad fit
        } else {
            // How close is available capacity to ideal capacity
            capacityFitness = Math.abs(availableCapacity - idealCapacity) / 100.0;
        }

        // Calculate VM type fitness
        double vmTypeFitness = 0.5; // Default value

        // VM types might be better for certain task types
        int taskType = task.getTaskType();
        int vmType = 0;

        if (vm instanceof EdgeVM) {
            vmType = vm.getId();
        } else if (vm instanceof CloudVM) {
            vmType = vm.getId();
        }

        // Simple rule: some VM types are better for certain task types
        if ((taskType % 3 == 0 && vmType == 0) ||
                (taskType % 3 == 1 && vmType == 1) ||
                (taskType % 3 == 2 && vmType == 2)) {
            vmTypeFitness = 0.2; // Good match
        }

        // Calculate historical performance fitness
        double historyFitness = 0.5; // Default value

        // Better fitness for VMs with good historical performance
        int hostId = vm.getHost().getId();
        if (taskHistory.containsKey(hostId) && !taskHistory.get(hostId).isEmpty()) {
            List<TaskExecution> executions = taskHistory.get(hostId);

            // Consider only executions of the same task type
            List<TaskExecution> typeExecutions = new ArrayList<>();
            for (TaskExecution execution : executions) {
                if (execution.taskType == taskType) {
                    typeExecutions.add(execution);
                }
            }

            if (!typeExecutions.isEmpty()) {
                // Calculate success rate
                int successCount = 0;
                for (TaskExecution execution : typeExecutions) {
                    if (execution.successful) {
                        successCount++;
                    }
                }

                double successRate = (double)successCount / typeExecutions.size();
                historyFitness = 1.0 - successRate; // Lower is better
            }
        }

        // Combined fitness (weighted sum)
        return (capacityFitness * 0.5) + (vmTypeFitness * 0.3) + (historyFitness * 0.2);
    }

    /**
     * Calculate distance between locations
     */
    private double calculateDistance(Location loc1, Location loc2) {
        double x1 = loc1.getXPos();
        double y1 = loc1.getYPos();
        double x2 = loc2.getXPos();
        double y2 = loc2.getYPos();

        return Math.sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1));
    }

    @Override
    public void startEntity() {
        // Initialize number of hosts
        numberOfHost = SimManager.getInstance().getEdgeServerManager().getDatacenterList().size();
    }

    @Override
    public void processEvent(SimEvent ev) {
        switch (ev.getTag()) {
//            case SimSettings.PERIODIC_METRIC_UPDATE:
//                updateMetrics();
//
//                // Schedule next update
//                CloudSim.send(getEntityId(), getEntityId(), METRIC_UPDATE_INTERVAL,
//                        SimSettings.PERIODIC_METRIC_UPDATE, null);
//                break;
//
//            case SimSettings.LEARNING_UPDATE:
//                updateFuzzyQLearning();
//
//                // Schedule next learning
//                CloudSim.send(getEntityId(), getEntityId(), LEARNING_INTERVAL,
//                        SimSettings.LEARNING_UPDATE, null);
//                break;
//
//            case SimSettings.POLICY_UPDATE:
//                updatePolicy();
//
//                // Schedule next policy update
//                CloudSim.send(getEntityId(), getEntityId(), POLICY_UPDATE_INTERVAL,
//                        SimSettings.POLICY_UPDATE, null);
//                break;

            case TASK_COMPLETED:
                Task completedTask = (Task) ev.getData();
                recordTaskCompletion(completedTask, true);
                updateQTableWithReward(completedTask, true);
                break;

            case TASK_FAILED:
                Task failedTask = (Task) ev.getData();
                recordTaskCompletion(failedTask, false);
                updateQTableWithReward(failedTask, false);
                break;

            default:
                break;
        }
    }
    /**
     * Update device reliability estimates
     */
    private void updateDeviceReliability() {
        for (Map.Entry<Integer, List<TaskExecution>> entry : taskHistory.entrySet()) {
            int deviceId = entry.getKey();
            List<TaskExecution> executions = entry.getValue();

            if (executions.isEmpty()) {
                continue;
            }

            // Consider only recent history
            int startIdx = Math.max(0, executions.size() - 100);
            List<TaskExecution> recent = executions.subList(startIdx, executions.size());

            // Calculate success rate
            int successCount = 0;
            for (TaskExecution execution : recent) {
                if (execution.successful) {
                    successCount++;
                }
            }

            double reliability = (double)successCount / recent.size();

            // Update with smoothing (80% old, 20% new)
            deviceReliability.put(deviceId, deviceReliability.get(deviceId) * 0.8 + reliability * 0.2);

            // Also calculate availability based on capacity constraints
            int capacityFailures = 0;
            for (TaskExecution execution : recent) {
                if (!execution.successful && execution.failureReason == FailureReason.CAPACITY) {
                    capacityFailures++;
                }
            }

            double availability = 1.0 - ((double)capacityFailures / recent.size());
            deviceAvailability.put(deviceId, deviceAvailability.get(deviceId) * 0.8 + availability * 0.2);
        }
    }

    /**
     * Record task completion
     */
    private void recordTaskCompletion(Task task, boolean success) {
        int deviceId = task.getAssociatedHostId();
        double completionTime = CloudSim.clock() - task.getSubmissionTime();

        // Determine failure reason
        FailureReason failureReason = FailureReason.UNKNOWN;
        if (!success) {
            // For simplicity, assume VM capacity is the main failure reason
            // In a real system, this would be determined from the failure details
            failureReason = FailureReason.CAPACITY;
        }

        // Create execution record
        TaskExecution execution = new TaskExecution(
                task.getCloudletId(),
                task.getTaskType(),
                deviceId,
                completionTime,
                success,
                failureReason
        );

        // Add to history
        if (!taskHistory.containsKey(deviceId)) {
            taskHistory.put(deviceId, new ArrayList<>());
        }
        taskHistory.get(deviceId).add(execution);

        // Keep history bounded
        if (taskHistory.get(deviceId).size() > 1000) {
            taskHistory.get(deviceId).remove(0);
        }

        // Update failure rate
        totalTasks++;
        if (!success) {
            failedTasks++;
            edgeFailureRate = (double)failedTasks / totalTasks;
        }

        // Periodically reset counters for adaptive behavior
        if (totalTasks > 1000) {
            // Keep recent history (last 500 tasks)
            totalTasks = 500;
            failedTasks = (int)(edgeFailureRate * 500);
        }
    }

    /**
     * Update Q-table with reward from task execution
     */
    private void updateQTableWithReward(Task task, boolean success) {
        // Get task metadata
        TaskMetadata metadata = taskMetadata.remove(task.getCloudletId());
        if (metadata == null) {
            return; // No metadata for this task
        }

        // Calculate reward
        double reward = calculateReward(task, metadata, success);

        // Get action taken
        String action;
        if (task.getAssociatedHostId() == SimSettings.CLOUD_DATACENTER_ID) {
            action = "cloud";
        } else if (task.getAssociatedHostId() < numberOfHost) {
            action = "edge";
        } else {
            action = "mobile";
        }

        // Get relevant states
        String[] relevantStates = getRelevantStates(metadata);

        // Update Q-values for relevant states
        for (String state : relevantStates) {
            if (fuzzyQTable.containsKey(state)) {
                Map<String, double[]> actionValues = fuzzyQTable.get(state);
                Map<String, Integer> visits = fuzzyRuleVisits.get(state);

                // Current Q-value
                double oldQ = actionValues.get(action)[0];

                // Maximum Q-value for next state (simplified: assume terminal state)
                double maxNextQ = 0.0;

                // Q-learning update: Q(s,a) = Q(s,a) + alpha * (r + gamma * maxQ(s',a') - Q(s,a))
                double newQ = oldQ + alpha * (reward + gamma * maxNextQ - oldQ);
                actionValues.get(action)[0] = newQ;

                // Update visit count
                int oldVisits = visits.getOrDefault(action, 0);
                visits.put(action, oldVisits + 1);
            }
        }
    }

    /**
     * Get relevant fuzzy states for a task
     */
    private String[] getRelevantStates(TaskMetadata metadata) {
        // Get fuzzy memberships
        Map<String, Double> taskSizeMemberships = getFuzzyMemberships("task_size", metadata.taskSize);
        Map<String, Double> wanBwMemberships = getFuzzyMemberships("wan_bandwidth", metadata.wanBandwidth);
        Map<String, Double> edgeUtilMemberships = getFuzzyMemberships("edge_utilization", metadata.edgeUtilization);
        Map<String, Double> delaySensMemberships = getFuzzyMemberships("delay_sensitivity",
                metadata.delaySensitivity > 0 ?
                        Math.min(1.0, 1.0 / metadata.delaySensitivity) : 0.1);
        Map<String, Double> dataSizeMemberships = getFuzzyMemberships("data_size",
                metadata.inputDataSize + metadata.outputDataSize);
        Map<String, Double> edgeCapMemberships = getFuzzyMemberships("edge_capacity", metadata.edgeCapacity);
        Map<String, Double> failRateMemberships = getFuzzyMemberships("edge_failure_rate", metadata.edgeFailureRate);

        // State combination 1: Task size, edge utilization, edge capacity
        String state1 = "task_size-" + getMaxFuzzyTerm(taskSizeMemberships) +
                "_edge_utilization-" + getMaxFuzzyTerm(edgeUtilMemberships) +
                "_edge_capacity-" + getMaxFuzzyTerm(edgeCapMemberships);

        // State combination 2: Task size, wan bandwidth, delay sensitivity
        String state2 = "task_size-" + getMaxFuzzyTerm(taskSizeMemberships) +
                "_wan_bandwidth-" + getMaxFuzzyTerm(wanBwMemberships) +
                "_delay_sensitivity-" + getMaxFuzzyTerm(delaySensMemberships);

        // State combination 3: Edge utilization, edge failure rate, data size
        String state3 = "edge_utilization-" + getMaxFuzzyTerm(edgeUtilMemberships) +
                "_edge_failure_rate-" + getMaxFuzzyTerm(failRateMemberships) +
                "_data_size-" + getMaxFuzzyTerm(dataSizeMemberships);

        return new String[]{state1, state2, state3};
    }

    /**
     * Calculate reward for task execution
     */
    private double calculateReward(Task task, TaskMetadata metadata, boolean success) {
        if (!success) {
            return -10.0; // Significant penalty for failure
        }

        double timeReward = getTimeReward(task, metadata);

        // Additional reward components
        double resourceReward = 0.0;

        // Resource efficiency reward/penalty
        if (task.getAssociatedHostId() == SimSettings.CLOUD_DATACENTER_ID) {
            // Cloud execution - small penalty for using cloud resources
            resourceReward = -1.0;
        } else if (task.getAssociatedHostId() < numberOfHost) {
            // Edge execution - reward depends on utilization
            if (metadata.edgeUtilization < 50) {
                resourceReward = 2.0; // Good use of underutilized edge
            } else if (metadata.edgeUtilization < 70) {
                resourceReward = 1.0; // Moderate use of edge
            } else {
                resourceReward = 0.0; // No reward for using highly utilized edge
            }
        } else {
            // Mobile execution - good for offloading system load
            resourceReward = 2.0;
        }

        // Combined reward
        return timeReward + resourceReward;
    }

    private static double getTimeReward(Task task, TaskMetadata metadata) {
        double completionTime = CloudSim.clock() - task.getSubmissionTime();
        double deadline = metadata.delaySensitivity;

        // Calculate time component of reward
        double timeReward;
        if (deadline > 0 && completionTime <= deadline) {
            // Met deadline - reward based on how much faster than deadline
            timeReward = 5.0 * (1.0 - (completionTime / deadline));
        } else if (deadline > 0) {
            // Missed deadline - penalty proportional to how much it was missed
            timeReward = -3.0 * Math.min(2.0, (completionTime / deadline) - 1.0);
        } else {
            // No specific deadline - use a reasonable default (10 seconds)
            timeReward = 5.0 * (1.0 - Math.min(1.0, completionTime / 10.0));
        }
        return timeReward;
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

    @Override
    public void shutdownEntity() {
        // Log final statistics
        SimLogger.printLine("FuzzyQLEdgeOrchestrator final statistics:");
        SimLogger.printLine("  Total tasks processed: " + totalTasks);
        SimLogger.printLine("  Edge failure rate: " + edgeFailureRate);

        // Log best fuzzy rules
        SimLogger.printLine("Top performing fuzzy rules:");
        Map<String, Double> bestRules = getTopPerformingRules(5);
        for (Map.Entry<String, Double> entry : bestRules.entrySet()) {
            SimLogger.printLine("  " + entry.getKey() + ": " + entry.getValue());
        }
    }

    /**
     * Get top performing fuzzy rules
     */
    private Map<String, Double> getTopPerformingRules(int limit) {
        Map<String, Double> rulePerformance = new HashMap<>();

        // Calculate performance for each rule
        for (Map.Entry<String, Map<String, double[]>> entry : fuzzyQTable.entrySet()) {
            String state = entry.getKey();
            Map<String, double[]> actions = entry.getValue();

            // Find best action for this state
            String bestAction = "";
            double bestQ = Double.NEGATIVE_INFINITY;

            for (Map.Entry<String, double[]> actionEntry : actions.entrySet()) {
                if (actionEntry.getValue()[0] > bestQ) {
                    bestQ = actionEntry.getValue()[0];
                    bestAction = actionEntry.getKey();
                }
            }

            // Record rule performance
            rulePerformance.put(state + " -> " + bestAction, bestQ);
        }

        // Sort rules by performance
        List<Map.Entry<String, Double>> sortedRules = new ArrayList<>(rulePerformance.entrySet());
        sortedRules.sort((a, b) -> Double.compare(b.getValue(), a.getValue()));

        // Take top N rules
        Map<String, Double> topRules = new LinkedHashMap<>();
        for (int i = 0; i < Math.min(limit, sortedRules.size()); i++) {
            Map.Entry<String, Double> entry = sortedRules.get(i);
            topRules.put(entry.getKey(), entry.getValue());
        }


        return topRules;
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
     * Helper class for task execution record
     */
    private static class TaskExecution {
        public final int taskId;
        public final int taskType;
        public final int deviceId;
        public final double completionTime;
        public final boolean successful;
        public final FailureReason failureReason;

        public TaskExecution(int taskId, int taskType, int deviceId, double completionTime,
                             boolean successful, FailureReason failureReason) {
            this.taskId = taskId;
            this.taskType = taskType;
            this.deviceId = deviceId;
            this.completionTime = completionTime;
            this.successful = successful;
            this.failureReason = failureReason;
        }
    }

    /**
     * Enum for task failure reasons
     */
    private enum FailureReason {
        CAPACITY,
        NETWORK,
        TIMEOUT,
        UNKNOWN
    }

    /**
     * Helper class for task metadata
     */
    private static class TaskMetadata {
        public final double decisionTime;
        public final int taskType;
        public final double taskSize;
        public final double inputDataSize;
        public final double outputDataSize;
        public final double delaySensitivity;
        public final double wanBandwidth;
        public final double manDelay;
        public final double edgeUtilization;
        public final double edgeCapacity;
        public final double edgeFailureRate;
        public final int targetEdgeHost;

        public TaskMetadata(double decisionTime, int taskType, double taskSize,
                            double inputDataSize, double outputDataSize, double delaySensitivity,
                            double wanBandwidth, double manDelay, double edgeUtilization,
                            double edgeCapacity, double edgeFailureRate, int targetEdgeHost) {
            this.decisionTime = decisionTime;
            this.taskType = taskType;
            this.taskSize = taskSize;
            this.inputDataSize = inputDataSize;
            this.outputDataSize = outputDataSize;
            this.delaySensitivity = delaySensitivity;
            this.wanBandwidth = wanBandwidth;
            this.manDelay = manDelay;
            this.edgeUtilization = edgeUtilization;
            this.edgeCapacity = edgeCapacity;
            this.edgeFailureRate = edgeFailureRate;
            this.targetEdgeHost = targetEdgeHost;
        }
    }

    /**
     * Class representing a fuzzy rule
     */
    private static class FuzzyRule {
        private final String variable;
        private final String term;
        private final double min;
        private final double max;

        public FuzzyRule(String variable, String term, double min, double max) {
            this.variable = variable;
            this.term = term;
            this.min = min;
            this.max = max;
        }

        public String getVariable() {
            return variable;
        }

        public String getTerm() {
            return term;
        }

        /**
         * Calculate membership value for a given input
         */
        public double getMembership(double value) {
            // Trapezoidal membership function
            double middle = (max - min) / 3.0;

            if (value < min || value > max) {
                return 0.0;
            } else if (value >= (min + middle) && value <= (max - middle)) {
                return 1.0;
            } else if (value < (min + middle)) {
                return (value - min) / middle;
            } else {
                return (max - value) / middle;
            }
        }
    }
    public double getManDelayForAgent(){
        double delay = 0;
        double mu = 0;
        double lambda = 0;
        double bandwidth = 1300*1024; //Kbps , C

        if (totalSizeOfActiveManTasks == 0){
            mu = bandwidth;
        }else{
            mu = bandwidth / (totalSizeOfActiveManTasks * 8);
        }

        lambda = activeManTaskCount;


        if (lambda >= mu){
            return 0;
        }else{
            delay = 1 / (mu - lambda);
            return delay;
        }
    }

}