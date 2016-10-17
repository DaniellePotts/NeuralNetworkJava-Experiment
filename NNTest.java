import java.text.DecimalFormat;

class MainProgram{

    public static void main(String [] args)
    {
        int [] layerSizes = new int[3];
        layerSizes[0] = 1;
        layerSizes[1] = 2;
        layerSizes[2] = 1;
			TransferFunction[]tFuncs=new TransferFunction[3];
            tFuncs[0] = TransferFunction.None;
            tFuncs[1] = TransferFunction.RationalSigmoid;
            tFuncs[2] = TransferFunction.Linear;
			NeuralNetwork backProp = new NeuralNetwork(layerSizes,tFuncs);

			double [] input = new double[1];
            input[0] = 1.0;
			double [] desired = new double[1];
            desired[0] = 2.5;
			double [] output = new double[1];

			double error = 0.0;

            DecimalFormat df = new DecimalFormat();
            df.setMaximumFractionDigits(2);

			for (int i = 0; i < 1000; i++)
			{
				error = backProp.TrainBP(input, desired, 0.17, 0.1);

				output = backProp.Run(input, output);

				if (i%100 == 0)
				{
				    System.out.println("Iteration " + i + ":\n\tInput " + input[0] + " output " + output[0] + " Error " + df.format(error));
				}
			}
    }
}