import org.vu.contest.ContestSubmission;
import org.vu.contest.ContestEvaluation;

import java.util.Random;
import java.util.Properties;
import java.util.Arrays;

public class player28 implements ContestSubmission
{
	Random rnd_;
	ContestEvaluation evaluation_;
    private int evaluations_limit_;
    protected static final double spaceBoundary = 5.0D;
    protected static final int nDim = 10;
    protected static final int nPop = 100;

	public player28()
	{
		rnd_ = new Random();
	}

	public void setSeed(long seed)
	{
		// Set seed of algortihms random process
		rnd_.setSeed(seed);
	}

	public void setEvaluation(ContestEvaluation evaluation)
	{
		// Set evaluation problem used in the run
		evaluation_ = evaluation;

		// Get evaluation properties
		Properties props = evaluation.getProperties();
        // Get evaluation limit
        evaluations_limit_ = Integer.parseInt(props.getProperty("Evaluations"));
		// Property keys depend on specific evaluation
		// E.g. double param = Double.parseDouble(props.getProperty("property_name"));
        boolean isMultimodal = Boolean.parseBoolean(props.getProperty("Multimodal"));
        boolean hasStructure = Boolean.parseBoolean(props.getProperty("Regular"));
        boolean isSeparable = Boolean.parseBoolean(props.getProperty("Separable"));

		// Do sth with property values, e.g. specify relevant settings of your algorithm
        if(isMultimodal){
            // Do sth
        }else{
            // Do sth else
        }
    }

    public double[] generateRandomGenotype()
    {
        double[] genotype = new double[nDim];
        for(int i = 0; i < genotype.length; i++) {
            genotype[i] = -spaceBoundary + 2 * spaceBoundary * rnd_.nextDouble();
        }

        return genotype;
    }

    public double[][] generateRandomPopulation()
    {
        double[][] population = new double[nPop][nDim];
        for(int i = 0; i < population.length; i++){
            population[i] = generateRandomGenotype();
        }
        return population;
    }

    public int evaluatePopulation(double[][] population, int evals)
    {
        for(double[] individuals : population){
            Double fitness = (double) evaluation_.evaluate(individuals);
            evals++;
            if(evals == evaluations_limit_){
                break;
            }
        }
        return evals;
    }

	public void run()
	{
        int evals = 0;

        double [][] population = generateRandomPopulation();

        while(evals<evaluations_limit_){
            evals = evaluatePopulation(population, evals);
            population = generateRandomPopulation();
        }
    }

    public double[] zeros(int length)
    {
        double[] result = new double[length];
        for(int i = 0; i < length; i++)
            result[i] = 0.0D;
        return result;
    }

    public double[][] zeros(int height, int width)
    {
        double[][] matrix = new double[height][width];
        for(int h = 0; h < height; h++)
            for(int w = 0; w < width; w++)
                matrix[h][w] = 0.0D;
        return matrix;

    }

    public double[] rand(int length, double boundary)
    {
        double[] vector = new double[length];
        for(int i = 0; i < length; i++)
            vector[i] = -boundary + 2 * boundary * rnd_.nextDouble();
        return vector;
    }

    public double[] norm(int length)
    {
        double[] result = new double[length];
        for(int i = 0; i < length; i++)
            result[i] = rnd_.nextGaussian();
        return result;
    }

    public double average(double[] vector)
    {
        double sum = 0;
        for(int i = 0; i < vector.length; i++)
            sum += vector[i];
        return sum / (double)vector.length;
    }

    public double[] mean(double[][] matrix)
    {
        int width = matrix[0].length;
        int height = matrix.length;
        double[] mean = new double[width];
        for(int w = 0; w < width; w++)
            for(int h = 0; h < height; h++)
                mean[w] += matrix[h][w];
        for(int w = 0; w < width; w++)
            mean[w] /= height;
        return mean;
    }

    public double[][] covariance(double[][] matrix)
    {
        int height = matrix.length;
        int width = matrix[0].length;
        double[][] covariance = zeros(width, width);
        double[] mean = mean(matrix);
        for(int x = 0; x < width; x++)
            for(int y = x; y < width; y++){
                double c = 0;
                for(int h = 0; h < height; h++)
                    c +=  (matrix[h][x] - mean[x]) * (matrix[h][y] - mean[y]);
                c /= height - 1;
                covariance[x][y] = c;
                covariance[y][x] = c;
            }

        return covariance;
    }

    public double[] plus(double[] left, double[] right)
    {
        double[] result = new double[left.length];
        for(int i = 0; i < left.length; i++)
            result[i] = left[i] + right[i];
        return result;
    }

    public double[][] plus(double[] vector, double[][] matrix)
    {
        for(int h = 0; h < matrix.length; h++)
            for(int w = 0; w < matrix[0].length; w++)
                matrix[h][w] += vector[w];
        return matrix;
    }

    public double[] mult(double[][] matrix, double[] vector)
    {
        int width = matrix[0].length;
        double[] result = zeros(width);
        for(int h = 0; h < matrix.length; h++)
            for(int w = 0; w < width; w++)
                result[w] += matrix[h][w] * vector[w];
        return result;
    }

    public double[] mult(double scalar, double[] vector)
    {
        for(int i = 0; i < vector.length; i++)
            vector[i] *= scalar;
        return vector;
    }

    public double[][] mult(double scalar, double[][] matrix)
    {
        for(int h = 0; h < matrix.length; h++)
            for(int w = 0; w < matrix[0].length; w++)
                matrix[h][w] *= scalar;
        return matrix;
    }

    public void print(double[] vector)
    {
        System.out.println(Arrays.toString(vector));
    }

    public void print(double[][] matrix)
    {
        System.out.println();
        for(int i = 0; i < matrix.length; i++){
            print(matrix[i]);
        }
        System.out.println();
    }
}
