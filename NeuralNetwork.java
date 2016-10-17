import java.lang.Math;
import java.lang.Exception;

class NeuralNetwork{

    private int LayerCount;
    private int InputSize;
    private int[] LayerSize;

    private TransferFunction [] TFuncs;

    private double [][] LayerOutput;
    private double [][] LayerInput;

    private double [][] Biases;
    private double [][] Delta;
    private double [][] PreviousBiasDelta;

    private double [][][] Weights;
    private double [][][] PreviousWeightDelta;

    public NeuralNetwork(int [] lSize, TransferFunction [] t){

        LayerCount = lSize.length -1;
        InputSize = lSize[0];
        LayerSize = new int[LayerCount];

        for(int i=0;i<LayerCount;i++){
            LayerSize[i] = lSize[i+1];
        }

        TFuncs = new TransferFunction[LayerCount];

        for(int i=0;i<LayerCount;i++){
            TFuncs[i] = t[i+1];
        }

        Biases = new double[LayerCount][];
        PreviousBiasDelta = new double[LayerCount][];
        Delta = new double[LayerCount][];
        LayerOutput = new double[LayerCount][];
        LayerInput = new double[LayerCount][];

        Weights = new double[LayerCount][][];
        PreviousWeightDelta = new double[LayerCount][][];

        for(int i=0;i<LayerCount;i++){
            Biases[i] = new double[LayerSize[i]];
            PreviousBiasDelta[i] = new double[LayerSize[i]];
            Delta[i] = new double[LayerSize[i]];
            LayerOutput[i] = new double[LayerSize[i]];
            LayerInput[i] = new double[LayerSize[i]];

            Weights[i] = new double[i == 0 ? InputSize : LayerSize[i - 1]][];
            PreviousWeightDelta[i] = new double[i == 0 ? InputSize : LayerSize[i - 1]][];

            for(int j=0;j<(i == 0 ? InputSize: LayerSize[i -1]);j++){
                Weights[i][j] = new double[LayerSize[i]];
                PreviousWeightDelta[i][j] = new double[LayerSize[i]];
            }
        }

        for(int i=0;i<LayerCount;i++){
            for(int j=0;j<LayerSize[i];j++){
                
                Biases[i][j] = Guassian.GetRandomGuassian();
                PreviousBiasDelta[i][j] = 0.0;
                LayerOutput[i][j] = 0.0;
                LayerInput[i][j] = 0.0;
                Delta[i][j] = 0.0;
            }

            for(int j =0;j<(i == 0 ? InputSize : LayerSize[i - 1]);j++){

                for(int k=0;k<LayerSize[i];k++){

                    Weights[i][j][k] = Guassian.GetRandomGuassian(); //guassian
                    PreviousWeightDelta[i][j][k] = 0.0;
                }
            }
        }
    }

    public double [] Run(double [] input, double [] output){

        output = new double[LayerSize[LayerCount -1]];

        for(int i=0;i<LayerCount;i++){

            for(int j=0;j<LayerSize[i];j++){

                double sum = 0.0;

                for(int k =0;k<(i == 0 ? InputSize : LayerSize[i]); k++){

                    sum += Weights[i][k][j] * (i == 0 ? input[i] : LayerOutput[i - 1][k]);
                }

                sum += Biases[i][j];
                LayerInput[i][j] = sum;

                LayerOutput[i][j] = TransferFunctions.Evaluate(TFuncs[i],sum);
            }
        }

        for(int i=0;i<LayerSize[LayerCount - 1];i++){
            output[i] = LayerOutput[LayerCount -1][i];
        }

        return output;
    }

    public double TrainBP(double [] input, double [] desired, double trainingRate, double momentum){

        	if (desired.length != LayerSize[LayerCount - 1])
			{
				System.out.println("length is not correct of param desired");
                return 0.0;
			}

        double error = 0.0;
        double sum = 0.0;
        double weightDelta = 0.0;
        double biasDelta = 0.0;

        double [] output = new double[LayerSize[LayerCount -1]];

        output = Run(input,output);

        	for (int i = LayerCount - 1; i >= 0; i--) //we're going backwards
			{
				//output layer
				if (i == LayerCount - 1)
				{
					for (int j = 0; j < LayerSize[i]; j++)
					{
						Delta[i][j] = output[j] - desired[j]; //get the error value (e = o - d) error == output - the truth val
						error += Math.pow(Delta[i][j], 2);
						Delta[i][j] *= TransferFunctions.EvaluateDerivative(TFuncs[i], LayerInput[i][j]);
					}
				}
				else //hidden layer
				{
					for (int j = 0; j < LayerSize[i]; j++)
					{
						sum = 0.0;
						for (int k = 0; k < LayerSize[i + 1]; k++)
						{
							sum += Weights[i + 1][j][k]*Delta[i][k];
						}

						sum *= TransferFunctions.EvaluateDerivative(TFuncs[i], LayerInput[i][j]);
						Delta[i][j] = sum;
					}
				}
			}

			//update weights and biases
			for (int i = 0; i < LayerCount; i++)
			{
				for (int j = 0; j < (i == 0?InputSize:LayerSize[i-1]); j++)
				{
					for (int k = 0; k < LayerSize[i]; k++)
					{
						weightDelta = trainingRate*Delta[i][k]*(i == 0 ? input[j] : LayerOutput[i - 1][j]);
						Weights[i][j][k] -= weightDelta + momentum*PreviousWeightDelta[i][j][k];

						PreviousWeightDelta[i][j][k] = weightDelta;
					}

				}
			}

			for (int i = 0; i < LayerCount; i++)
			{
				for (int j = 0; j < LayerSize[i]; j++)
				{
					biasDelta = trainingRate*Delta[i][j];
					Biases[i][j] -= biasDelta + momentum * PreviousBiasDelta[i][j];

					PreviousBiasDelta[i][j] = biasDelta;
				}
			}

			return error;
    }
}