package edu.boun.edgecloudsim.applications.uarc;

import edu.boun.edgecloudsim.edge_client.Task;
import edu.boun.edgecloudsim.edge_orchestrator.EdgeOrchestrator;
import org.cloudbus.cloudsim.Vm;
import org.cloudbus.cloudsim.core.SimEvent;

public class AdaptiveEdgeOrchestrator extends EdgeOrchestrator {
    @Override
    public void initialize() {
        
    }

    @Override
    public int getDeviceToOffload(Task task) {
        return 0;
    }

    @Override
    public Vm getVmToOffload(Task task, int deviceId) {
        return null;
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
