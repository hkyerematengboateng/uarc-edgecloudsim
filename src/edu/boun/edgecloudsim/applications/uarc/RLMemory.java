package edu.boun.edgecloudsim.applications.uarc;

public class RLMemory {
    private RLState state;
    private RLState nextState;
    private double value;
    private int action;
    private boolean isDone;

    public RLMemory(RLState state, RLState nextState, double value, int action, boolean isDone) {
        this.state = state;
        this.nextState = nextState;
        this.value = value;
        this.action = action;
        this.isDone = isDone;
    }

    public RLState getState() {
        return state;
    }

    public void setState(RLState state) {
        this.state = state;
    }

    public RLState getNextState() {
        return nextState;
    }

    public void setNextState(RLState nextState) {
        this.nextState = nextState;
    }

    public double getValue() {
        return value;
    }

    public void setValue(double value) {
        this.value = value;
    }

    public int getAction() {
        return action;
    }

    public void setAction(int action) {
        this.action = action;
    }

    public boolean isDone() {
        return isDone;
    }

    public void setDone(boolean done) {
        isDone = done;
    }
}
