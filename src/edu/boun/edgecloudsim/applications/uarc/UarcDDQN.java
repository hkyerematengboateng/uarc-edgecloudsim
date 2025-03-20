package edu.boun.edgecloudsim.applications.uarc;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.UniformDistribution;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;

public class UarcDDQN {
    private final double DISCOUNT_FACTOR = 0.8;
    private double totalReward;
    private final double LEARNING_RATE = 0.0001;
    private double epsilon = 1;
    private final double MIN_EPSILON = 0.1;
    private final double EPSILON_FACTOR = 0.99;
    private final int MEMORY_SIZE = 1000000;
    private final int BATCH_SIZE = 4;
    private final double TAU = 0.01;
    private final int C_SYNC = 10;
    private ArrayList<RLMemory> memory; //it needs a new memoryItem structure
    private int numberOfActions;
    private int counterForEpsilon;

    private static UarcDDQN instance = null;

    private MultiLayerNetwork network;
    private MultiLayerNetwork targetNetwork;

    private double reward;
    private double avgQValue;
    private int actionCount;

    public UarcDDQN(int numberOfEdgeServers){
        this.numberOfActions = numberOfEdgeServers;
        this.counterForEpsilon = 0;
        totalReward = 0;
        memory = new ArrayList<RLMemory>();

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .updater(new Adam(LEARNING_RATE))
                .seed(1254)
                .miniBatch(false)
                .list()
                .layer(new DenseLayer.Builder()
                        .nIn(9 + numberOfEdgeServers)
                        .nOut(64)
                        .activation(Activation.RELU)
                        // random initialize weights with values between 0 and 1
                        .weightInit(new UniformDistribution(-1, 1))
                        .build())
                .layer(new DenseLayer.Builder()
                        .nIn(64)
                        .nOut(64)
                        .activation(Activation.RELU)
                        // random initialize weights with values between 0 and 1
                        .weightInit(new UniformDistribution(-1, 1))
                        .build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.MSE) //create hidden layer
                        .nIn(64)
                        .nOut(this.numberOfActions)
                        .activation(Activation.IDENTITY)
                        .weightInit(new UniformDistribution(-1, 1))
                        .build())
                .build();
        network = new MultiLayerNetwork(conf);
        targetNetwork = new MultiLayerNetwork(conf);
        network.init();
        targetNetwork.init();

        reward = 0;
        this.actionCount = 0;
        this.avgQValue = 0;
        syncTargetNetwork();
        instance = this;
    }
    // Implemented for smooth transition but never used.
    // Instead, syncTargetNetwork() is used
    public void updateTargetNetwork(){
        ArrayList<INDArray> qNetworkWeights = new ArrayList<>();
        //System.out.println("SIZE: "+ qNetwork.getLayers().length);
        int numberOfLayers = network.getLayers().length;
        for (int i = 0; i < numberOfLayers; i++){
            String param = i + "_W";
            INDArray weightsOfLayer = this.network.getParam(param);
            qNetworkWeights.add(weightsOfLayer);
        }

        HashMap<Integer, INDArray > weightMap = new HashMap<>();
        int layerCounter = 0;
        for (INDArray weights:qNetworkWeights){
            double [] theweigths = weights.ravel().toDoubleVector();
            int counter = 0;
            INDArray intendedValues = Nd4j.zeros(weights.shape());
            String paramForTarget = layerCounter + "_W";
            INDArray weightsOfTarget = this.targetNetwork.getParam(paramForTarget);
            double [] targetWeights = weightsOfTarget.ravel().toDoubleVector();
            //System.out.println("Target weights: " + targetWeights[0]);
            for (double weigth: theweigths){
                double theNewValue = weigth * TAU + targetWeights[counter] * (1-TAU);
                intendedValues.putScalar(counter, theNewValue);
                counter++;
            }
            weightMap.put(layerCounter, intendedValues);
            layerCounter++;
        }

        for ( Map.Entry<Integer, INDArray>  weightsFromMap: weightMap.entrySet()) {
            int key = weightsFromMap.getKey();
            String param = key + "_W";
            //System.out.println("Param: "+ param);
            this.targetNetwork.setParam(param, weightsFromMap.getValue());

            //INDArray w0_qNetwork = this.targetNetwork.getParam(param);
            //System.out.println("weights for " +param+ " targetNetwork: " + w0_qNetwork);
        }

    }

    private void syncTargetNetwork(){
        this.targetNetwork = this.network.clone();
    }

    public int DoAction(RLState state){
        this.actionCount++;
        Random rand = new Random();
        double randomNumber = rand.nextFloat();
        this.counterForEpsilon++;
        if (randomNumber <= this.epsilon){
            if (this.epsilon > MIN_EPSILON){
                this.epsilon = Math.max(this.epsilon * EPSILON_FACTOR, MIN_EPSILON);
            }
            return rand.nextInt(this.numberOfActions);
        }
        else{
            INDArray output = this.network.output(state.getState());
            //int action = output.argMax().getInt();
            return output.argMax().getInt();
        }
    }

    public void DDQN(RLState state, RLState nextState, double reward, int action, boolean isDone){

        RLMemory memoryItem = new RLMemory(state, nextState, reward, action, isDone);
        ArrayList<RLMemory> memoryItems = new ArrayList<>();
        if (this.memory.size() >= MEMORY_SIZE){
            this.memory.remove(0);
        }

        this.memory.add(memoryItem);

        if (this.memory.size() > BATCH_SIZE){
            memoryItems = getRandomMemoryItems();
        }
        else{
            memoryItems = this.memory;
        }

        for (RLMemory item:memoryItems) {
            INDArray target = this.network.output(item.getState().getState());
            if (item.isDone()){
                //System.out.println("Shape: " + target.shape());
                target.putScalar(item.getAction(), item.getValue());
            }
            else{
                int argmaxOfQNetworkForNextState = this.network.output(item.getNextState().getState()).argMax().getInt();
                INDArray targetNetworkOutput = this.targetNetwork.output(item.getNextState().getState());
                double targetValue = item.getValue() + this.DISCOUNT_FACTOR * targetNetworkOutput.getDouble(argmaxOfQNetworkForNextState);
                this.avgQValue += this.network.output(item.getNextState().getState()).max().getDouble();
                target.putScalar(item.getAction(), targetValue);
            }
            this.network.fit(item.getState().getState(), target);

        }
        if (this.counterForEpsilon % C_SYNC == 0){
            syncTargetNetwork();
        }

    }
    private ArrayList<RLMemory> getRandomMemoryItems(){
        ArrayList<RLMemory> memoryItems = new ArrayList<>();
        ArrayList<Integer> selectedNumbers = new ArrayList<>();
        Random rand = new Random();

        for (int i=0; i < this.BATCH_SIZE; i++){
            int randomNumber = rand.nextInt(this.memory.size());

            while (selectedNumbers.contains(randomNumber)){
                randomNumber = rand.nextInt(this.memory.size());
            }

            selectedNumbers.add(randomNumber);
            memoryItems.add(this.memory.get(randomNumber));
        }
        return memoryItems;
    }

    public void saveModel(String episodeNo, Double reward, Double avgQValue) throws IOException {
        String modelName = "D-DqnModel-";
        modelName = modelName + episodeNo + "-"+ reward + "-" + avgQValue;
        this.network.save(new File(modelName), false);
    }

    public static UarcDDQN getInstance(){
        return instance;
    }

    public void setReward(double reward){
        this.reward = reward;
    }

    public double getReward(){
        return this.reward;
    }

    public double getAvgQvalue(){
        if (this.actionCount > 0){
            return this.avgQValue / this.actionCount;
        }
        return this.avgQValue;
    }

    public double getQvalue(){
        return this.avgQValue;
    }

    public void resetQValue(){
        this.avgQValue = 0;
    }
}
