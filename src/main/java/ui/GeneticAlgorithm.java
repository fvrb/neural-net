package ui;

import org.apache.commons.math3.random.RandomDataGenerator;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;

import static java.lang.System.exit;

public class GeneticAlgorithm {

	static double[][] trainData = null;
	static double[] trainRes = null;
	static double[][] testData = null;
	static double[] testRes = null;

	public static void main(String ... args) {
		String train = null;
		String test = null;
		String nn = null;
		Integer popsize = null;
		Integer elitism = null;
		Double p = null;
		Double K = null;
		Integer iter = null;

		for (int i = 0; i < args.length;) {
			switch(args[i]) {
				case "--train" -> {
					train = args[i+1];
					i = i+2;
				}
				case "--test" -> {
					test = args[i+1];
					i = i+2;
				}
				case "--nn" -> {
					nn = args[i+1];
					i = i+2;
				}
				case "--popsize" -> {
					popsize = Integer.parseInt(args[i+1]);
					i = i+2;
				}
				case "--elitism" -> {
					elitism = Integer.parseInt(args[i+1]);
					i = i+2;
				}
				case "--p" -> {
					p = Double.parseDouble(args[i+1]);
					i = i+2;
				}
				case "--K" -> {
					K = Double.parseDouble(args[i+1]);
					i = i+2;
				}
				case "--iter" -> {
					iter = Integer.parseInt(args[i+1]);
					i = i+2;
				}
				default -> {
					System.out.println("Wrong arguments");
					exit(0);
				}
			}
		}

		ucitajPodatke(train, test);

		List<double[]> pop = postaviPocPopulaciju(nn, popsize, trainData[0].length);

		Map<double[], Double> eval = evaluate(pop, nn, trainData, trainRes);
		List<Map.Entry<double[], Double>> sortedEval = new ArrayList<>(eval.entrySet());
		sortedEval.sort(Map.Entry.comparingByValue());
		Collections.reverse(sortedEval);

		for (int i = 1; i <= iter; i++) {
			List<double[]> newPop = new ArrayList<>();

			for (Map.Entry<double[], Double> elem : sortedEval) {
				if (newPop.size() == elitism)
					break;

				newPop.add(elem.getKey());
			}

			while (newPop.size() < popsize) {
				double[] r1 = select(sortedEval);
				double[] r2 = select(sortedEval);

				while (Arrays.equals(r1, r2)) {
					r2 = select(sortedEval);
				}


				double[] d = krizaj(r1, r2);
				d = mutiraj(d, p, K);
				newPop.add(d);
			}
			pop = newPop;

			eval = evaluate(pop, nn, trainData, trainRes);
			sortedEval = new ArrayList<>(eval.entrySet());
			sortedEval.sort(Map.Entry.comparingByValue());
			Collections.reverse(sortedEval);

			if (i % 100 == 0) {

				NeuralNet trainedNN = new NeuralNet(nn, testData, testRes, sortedEval.get(0).getKey());
				trainedNN.prolaz();
				double testErr = trainedNN.mse();

				System.out.format("@%d [Train error]: %f\t\t[Test error]: %f\n", i, 1/sortedEval.get(0).getValue(), testErr);
			}
		}


		NeuralNet resNN = new NeuralNet(nn, testData, testRes, sortedEval.get(0).getKey());
		resNN.prolaz();
		ispis(resNN.getResult(), test);
	}

	public static void ispis(double[][] results, String testFilename) {
		String filename = testFilename.substring(0, testFilename.indexOf('.')) + "_res.txt";

		try (BufferedWriter writer = new BufferedWriter(new FileWriter(filename))) {
			int columnsNum = testData[0].length;

			for (int i = 0; i < testData.length; i++) {
				StringBuilder sb = new StringBuilder();

				for (int j = 0; j < columnsNum; j++) {
					if (j != 0)
						sb.append(", ");

					sb.append(testData[i][j]);
				}

				sb.append(", ").append(results[0][i]);
				writer.write(sb.toString());
				writer.newLine();
			}

		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	public static double[] mutiraj(double[] d, double p, double K) {
		double[] res = new double[d.length];

		RandomDataGenerator rdg = new RandomDataGenerator();

		for (int i = 0; i < d.length; i++) {
			if (rdg.nextUniform(0, 1) <= p) {
				res[i] = d[i] + rdg.nextGaussian(0, K);
			} else {
				res[i] = d[i];
			}
		}

		return res;
	}

	public static double[] select(List<Map.Entry<double[], Double>> entries) {
		double sum = entries.stream().mapToDouble(e->e.getValue()).sum();
		double incr = 0;

		RandomDataGenerator rdg = new RandomDataGenerator();
		double randDbl = rdg.nextUniform(0, 1) * sum;

		for (Map.Entry<double[], Double> elem : entries) {
			incr += elem.getValue();
			if (incr >= randDbl) {
				return elem.getKey();
			}
		}

		return null;
	}

	public static Map<double[], Double> evaluate(List<double[]> pop, String arch, double[][] data, double[] realVals) {
		Map<double[], Double> res = new HashMap<>();

		for (double[] elem : pop) {
			NeuralNet nn = new NeuralNet(arch, data, realVals, elem);
			nn.prolaz();
			res.put(elem, 1/nn.mse());
		}

		return res;
	}

	public static List<double[]> postaviPocPopulaciju(String arch, int popsize, int inputSize) {
		List<double[]> p = new ArrayList<>();

		int[] neuronsInLayer = NeuralNet.archData(arch).getIntArr();

		int size = 0;
		int last = inputSize;

		for (int i = 0; i < neuronsInLayer.length; i++) {
			size += last * neuronsInLayer[i] + neuronsInLayer[i];
			last = neuronsInLayer[i];
		}

		RandomDataGenerator rdg = new RandomDataGenerator();

		for (int i = 0; i < popsize; i++) {
			double[] nn = new double[size];
			for (int j = 0; j < size; j++) {
				nn[j] = rdg.nextGaussian(0, 0.2);
			}
			p.add(nn);
		}

		return p;
	}

	public static double[] krizaj(double[] r1, double[] r2) {
		double[] res = new double[r1.length];
		for (int i = 0; i < r1.length; i++) {
			res[i] = (r1[i] + r2[i]) / 2;
		}
		return res;
	}

	public static void ucitajPodatke(String trainPath, String testPath) {
		Path path = Paths.get(trainPath);
		List<String> lines = null;
		try {
			lines = Files.readAllLines(path);
		} catch (IOException e) {
			e.printStackTrace();
		}

		// lines = lines.subList(1, lines.size());    koristeno ako su prisutni headeri

		double[][] matrix = null;
		double[] results = new double[lines.size()];

		int i = 0;

		for (String l : lines) {
			String dataSubstr =  l.substring(0, l.lastIndexOf(","));
			String resSubstr = l.substring(l.lastIndexOf(",")+1).trim();

			String[] datas = dataSubstr.split(",");
			for (int j = 0; j < datas.length; j++)
				datas[j] = datas[j].trim();

			if (matrix == null) {
				matrix = new double[lines.size()][datas.length];
			}

			for (int j = 0; j < datas.length; j++) {
				matrix[i][j] = Double.parseDouble(datas[j]);
			}
			results[i] = Double.parseDouble(resSubstr);
			i++;
		}

		trainData = matrix;
		trainRes = results;

		path = Paths.get(testPath);
		lines = null;
		try {
			lines = Files.readAllLines(path);
		} catch (IOException e) {
			e.printStackTrace();
		}

		// lines = lines.subList(1, lines.size());		koristi se za headere

		matrix = null;
		results = new double[lines.size()];

		i = 0;

		for (String l : lines) {
			String dataSubstr =  l.substring(0, l.lastIndexOf(","));
			String resSubstr = l.substring(l.lastIndexOf(",")+1).trim();

			String[] datas = dataSubstr.split(",");
			for (int j = 0; j < datas.length; j++)
				datas[j] = datas[j].trim();

			if (matrix == null) {
				matrix = new double[lines.size()][datas.length];
			}

			for (int j = 0; j < datas.length; j++) {
				matrix[i][j] = Double.parseDouble(datas[j]);
			}
			results[i] = Double.parseDouble(resSubstr);
			i++;
		}

		testData = matrix;
		testRes = results;
	}
}
