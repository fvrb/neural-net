package ui;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;
import java.util.regex.*;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;

public class NeuralNet {
    private double[][] inputs;
    private double[] realVals;
    private List<double[][]> weights = new ArrayList<>();
    private List<double[]> biases = new ArrayList<>();
    private double[][] result;
    private String[] activFns;


    public double[][] getResult() {
        return result;
    }

    public static ArrayPair archData(String arch) {
        String[] parts = arch.split("-");
        Pattern pattern = Pattern.compile("^(\\d+)(\\D+)$");

        int[] neuronsInLayer = new int[parts.length + 1];
        String[] activFns = new String[parts.length];

        for (int i = 0; i < parts.length; i++) {
            Matcher matcher = pattern.matcher(parts[i]);

            if (matcher.matches()) {
                neuronsInLayer[i] = Integer.parseInt(matcher.group(1));
                activFns[i] = matcher.group(2);
            } else {
                throw new IllegalArgumentException("Invalid format: " + arch);
            }

        }
        neuronsInLayer[neuronsInLayer.length-1] = 1;

        return new ArrayPair(neuronsInLayer, activFns);
    }

    public static double s(double x) {
        return 1/(1+Math.exp(-x));
    }
    public static double relu(double x) {return Math.max(0., x);}
    public static double tanh(double x) {return Math.tanh(x);}
    public static double lrelu(double x) {return x > 0 ? x : 0.01*x;}

    public static RealMatrix activationFunc(RealMatrix matrix, Function<Double, Double> func) {

        RealMatrix res = new Array2DRowRealMatrix(matrix.getRowDimension(), matrix.getColumnDimension());

        for (int i = 0; i < matrix.getRowDimension(); i++) {
            for (int j = 0; j < matrix.getColumnDimension(); j++) {
                res.setEntry(i, j, func.apply(matrix.getEntry(i, j)));
            }
        }

        return res;
    }

    public static RealMatrix dodajSvimStupcima(RealMatrix matrix, RealMatrix vector) {
        RealMatrix res = new Array2DRowRealMatrix(matrix.getRowDimension(), matrix.getColumnDimension());
        for (int i = 0; i < matrix.getColumnDimension(); i++) {
            res.setColumnMatrix(i, matrix.getColumnMatrix(i).add(vector));
        }
        return res;
    }

    public void prolaz() {
        RealMatrix res = new Array2DRowRealMatrix(inputs);
        res = res.transpose();

        for (int i = 0; i < biases.size(); i++) {
            res = new Array2DRowRealMatrix(weights.get(i)).multiply(res);
            res = dodajSvimStupcima(res, new Array2DRowRealMatrix(biases.get(i)));

            if (i != biases.size()-1) {
                switch(activFns[i]) {
                    case "s":
                        res = activationFunc(res, NeuralNet::s);
                        break;
                    case "r":
                        res = activationFunc(res, NeuralNet::relu);
                        break;
                    case "t":
                        res = activationFunc(res, NeuralNet::tanh);
                        break;
                    case "l":
                        res = activationFunc(res, NeuralNet::lrelu);
                        break;
                }
            }
        }

        result = res.getData();
    }

    public double mse() {
        double sum = 0;
        for (int i = 0; i < realVals.length; i++) {
            sum += (realVals[i] - result[0][i])*(realVals[i] - result[0][i]);
        }
        return sum / realVals.length;
    }

    public NeuralNet(String arch, double[][] inputs, double[] realVals, double[] params) {
        this.realVals = realVals;
        this.inputs = inputs;

        ArrayPair ap = archData(arch);
        int[] neuronsInLayer = ap.getIntArr();
        activFns = ap.getChArr();

        int lastLayerSize = inputs[0].length;
        int pInd = 0;

        for (int i = 0; i < neuronsInLayer.length; i++) {
            double[][] w = new double[neuronsInLayer[i]][lastLayerSize];

            for (int j = 0; j < neuronsInLayer[i]; j++) {
                for (int k = 0; k < lastLayerSize; k++) {
                    w[j][k] = params[pInd++];
                }
            }
            weights.add(w);

            double[] b = new double[neuronsInLayer[i]];

            for (int j = 0; j < neuronsInLayer[i]; j++) {
                b[j] = params[pInd++];
            }

            biases.add(b);

            lastLayerSize = neuronsInLayer[i];
        }
    }
}
