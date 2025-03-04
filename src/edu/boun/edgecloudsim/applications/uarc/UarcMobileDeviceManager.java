package edu.boun.edgecloudsim.applications.uarc;

import edu.boun.edgecloudsim.core.SimManager;
import edu.boun.edgecloudsim.core.SimSettings;
import edu.boun.edgecloudsim.edge_client.CpuUtilizationModel_Custom;
import edu.boun.edgecloudsim.edge_client.MobileDeviceManager;
import edu.boun.edgecloudsim.edge_client.Task;
import edu.boun.edgecloudsim.network.NetworkModel;
import edu.boun.edgecloudsim.utils.Location;
import edu.boun.edgecloudsim.utils.TaskProperty;
import org.cloudbus.cloudsim.UtilizationModel;
import org.cloudbus.cloudsim.UtilizationModelFull;
import org.cloudbus.cloudsim.core.CloudSim;

import java.util.HashMap;

public class UarcMobileDeviceManager extends MobileDeviceManager {
    private static final int BASE = 100000; //start from base in order not to conflict cloudsim tag!

    private static final int UPDATE_MM1_QUEUE_MODEL = BASE + 1;
    private static final int REQUEST_RECEIVED_BY_CLOUD = BASE + 2;
    private static final int REQUEST_RECEIVED_BY_EDGE_DEVICE = BASE + 3;
    private static final int REQUEST_RECEIVED_BY_REMOTE_EDGE_DEVICE = BASE + 4;
    private static final int REQUEST_RECEIVED_BY_EDGE_DEVICE_TO_RELAY_NEIGHBOR = BASE + 5;
    private static final int RESPONSE_RECEIVED_BY_MOBILE_DEVICE = BASE + 6;
    private static final int RESPONSE_RECEIVED_BY_EDGE_DEVICE_TO_RELAY_MOBILE_DEVICE = BASE + 7;

    private static final double MM1_QUEUE_MODEL_UPDATE_INTEVAL = 5; //seconds

    private int taskIdCounter=0;

    private boolean dqnTraining = false;

    private final int EPISODE_SIZE = 75000;

    private double numberOfWlanOffloadedTask = 0;
    private double numberOfManOffloadedTask = 0;
    private double numberOfWanOffloadedTask = 0;
    private double activeManTaskCount = 0;
    private double activeWanTaskCount = 0;
    private double totalSizeOfActiveManTasks = 0;
    private double totalReward = 0;

    private HashMap<Integer, HashMap<Integer, Integer>> taskToStateActionPair = new HashMap<>();
    //private HashMap<Integer, MemoryItem> stateIDToMemoryItemPair = new HashMap<>();

    private static UarcMobileDeviceManager instance = null;
    public UarcMobileDeviceManager() throws Exception {
    }

    @Override
    public void initialize() {

    }

    @Override
    public UtilizationModel getCpuUtilizationModel() {
        return new CpuUtilizationModel_Custom();
    }

    //TODO
    @Override
    public void submitTask(TaskProperty edgeTask) {
        int vmType=0;
        int nextEvent=0;
        int nextDeviceForNetworkModel;
        SimSettings.NETWORK_DELAY_TYPES delayType;
        double delay=0;

        NetworkModel networkModel = SimManager.getInstance().getNetworkModel();
        Task task = createTask(edgeTask);

        Location edgeLocation = SimManager.getInstance().getMobilityModel()
                .getLocation(task.getMobileDeviceId(), CloudSim.clock());
        task.setSubmittedLocation(edgeLocation);
    }

    @Override
    public void startEntity() {
        super.startEntity();
        schedule(getId(), SimSettings.CLIENT_ACTIVITY_START_TIME +
                MM1_QUEUE_MODEL_UPDATE_INTEVAL, UPDATE_MM1_QUEUE_MODEL);
    }

    private Task createTask(TaskProperty edgeTask){
        UtilizationModel utilizationModel = new UtilizationModelFull(); /*UtilizationModelStochastic*/
        UtilizationModel utilizationModelCPU = getCpuUtilizationModel();

        Task task = new Task(edgeTask.getMobileDeviceId(), ++taskIdCounter,
                edgeTask.getLength(), edgeTask.getPesNumber(),
                edgeTask.getInputFileSize(), edgeTask.getOutputFileSize(),
                utilizationModelCPU, utilizationModel, utilizationModel);

        //set the owner of this task
        task.setUserId(this.getId());
        task.setTaskType(edgeTask.getTaskType());

        if (utilizationModelCPU instanceof CpuUtilizationModel_Custom) {
            ((CpuUtilizationModel_Custom)utilizationModelCPU).setTask(task);
        }

        return task;
    }
}
