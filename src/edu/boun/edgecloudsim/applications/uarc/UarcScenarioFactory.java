package edu.boun.edgecloudsim.applications.uarc;

import edu.boun.edgecloudsim.cloud_server.CloudServerManager;
import edu.boun.edgecloudsim.cloud_server.DefaultCloudServerManager;
import edu.boun.edgecloudsim.core.ScenarioFactory;
import edu.boun.edgecloudsim.edge_client.MobileDeviceManager;
import edu.boun.edgecloudsim.edge_client.mobile_processing_unit.DefaultMobileServerManager;
import edu.boun.edgecloudsim.edge_client.mobile_processing_unit.MobileServerManager;
import edu.boun.edgecloudsim.edge_orchestrator.EdgeOrchestrator;
import edu.boun.edgecloudsim.edge_server.DefaultEdgeServerManager;
import edu.boun.edgecloudsim.edge_server.EdgeServerManager;
import edu.boun.edgecloudsim.mobility.MobilityModel;
import edu.boun.edgecloudsim.mobility.NomadicMobility;
import edu.boun.edgecloudsim.network.NetworkModel;
import edu.boun.edgecloudsim.task_generator.IdleActiveLoadGenerator;
import edu.boun.edgecloudsim.task_generator.LoadGeneratorModel;

public class UarcScenarioFactory  implements ScenarioFactory {
    private int numOfMobileDevice;
    private double simulationTime;
    private String orchestratorPolicy;
    private String simScenario;

    UarcScenarioFactory(int _numOfMobileDevice,
                         double _simulationTime,
                         String _orchestratorPolicy,
                         String _simScenario){
        orchestratorPolicy = _orchestratorPolicy;
        numOfMobileDevice = _numOfMobileDevice;
        simulationTime = _simulationTime;
        simScenario = _simScenario;
    }

    /**
     * provides abstract Load Generator Model
     */
    @Override
    public LoadGeneratorModel getLoadGeneratorModel() {
        return new IdleActiveLoadGenerator(numOfMobileDevice, simulationTime, simScenario);
    }

    /**
     * provides abstract Edge Orchestrator
     */
    @Override
    public EdgeOrchestrator getEdgeOrchestrator() {
        return new AdaptiveEdgeOrchestrator(orchestratorPolicy, simScenario);
    }

    /**
     * provides abstract Mobility Model
     */
    @Override
    public MobilityModel getMobilityModel() {
        return new NomadicMobility(numOfMobileDevice, simulationTime);
    }

    /**
     * provides abstract Network Model
     */
    @Override
    public NetworkModel getNetworkModel() {
        return new UarcNetworkModel(numOfMobileDevice, simScenario);
    }

    /**
     * provides abstract Edge Server Model
     */
    @Override
    public EdgeServerManager getEdgeServerManager() {
        return new DefaultEdgeServerManager();
    }

    /**
     * provides abstract Cloud Server Model
     */
    @Override
    public CloudServerManager getCloudServerManager() {
        return new DefaultCloudServerManager();
    }

    /**
     * provides abstract Mobile Server Model
     */
    @Override
    public MobileServerManager getMobileServerManager() {
        return new DefaultMobileServerManager();
    }

    /**
     * provides abstract Mobile Device Manager Model
     */
    @Override
    public MobileDeviceManager getMobileDeviceManager() throws Exception {
        return new UarcMobileDeviceManager();
    }
}
