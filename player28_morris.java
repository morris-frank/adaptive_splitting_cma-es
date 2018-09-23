import org.vu.contest.ContestSubmission;
import org.vu.contest.ContestEvaluation;

import java.util.Random;

import javax.swing.plaf.synth.SynthMenuBarUI;

import java.util.Properties;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.ArrayList;
import java.lang.Object;

public class player28_morris implements ContestSubmission
{
    Random rnd_;
	ContestEvaluation evaluation_;
    private int evaluations_limit_;
    protected static final double maxPos = 5.0D;
    protected static final int nDim = 10;
    protected static final int nInd = 100;
    protected static final int nTribes = 8;
    double birthrate = 1.0D;
    int evals;

    public class Individual implements Comparable<Individual>
    {
        public double[] position;
        public double gradient;
        public double fitness;
        public int age;

        public Individual()
        {
            position = new double[nDim];
            fitness = 0;
            age = 1;
            gradient = 0.0D;
        }

        public double fitness()
        {
            if(fitness == 0){
                evals++;
                fitness = (double) evaluation_.evaluate(position);
            }
            return fitness;
        }

        @Override
        public int compareTo(Individual other) {
            if(this.fitness < other.fitness) return 1;
            else if(other.fitness < this.fitness) return -1;
            return 0;
        }
    }

    public class Tribe
    {
        public int size;
        public double birthrate;
        public List<Individual> individuals;
        public double[][] covariance;
        public double[] mean;
        public int generation;

        public Tribe()
        {
            size = 0;
            generation = 1;
            mean = new double[nDim];
            individuals = new ArrayList<Individual>();
        }

        public void addRandom(int n)
        {
            for(int i = 0; i < n; i++){
                Individual individual = new Individual();
                individual.position = rand(nDim, maxPos);
                individual.fitness();
                individuals.add(individual);
            }
            size += n;
        }

        public double[][] positions()
        {
            double[][] positions = zeros(size, nDim);
            for(int i = 0; i < size; i++)
                positions[i] = individuals.get(i).position;
            return positions;
        }

        public double[] ages()
        {
            double[] ages = new double[size];
            for(int i = 0; i < size; i++)
                ages[i] = individuals.get(i).age;
            return ages;
        }

        public double[] fitness()
        {
            double[] fitness = new double[size];
            for(int i = 0; i < size; i++)
                fitness[i] = individuals.get(i).fitness();
            return fitness;
        }

        public void newYear()
        {
            for(int i = 0; i < individuals.size(); i++)
            individuals.get(i).age++;

        }

        public void nextGeneration()
        {
            generation++;
            System.out.println();
            System.out.println(generation);
            System.out.println(average(ages()));
            covariance = covariance(positions());
            List<Individual> offspring = sample(size/2);
            individuals.addAll(offspring);
            selection(size);
        }

        public double stepSize()
        {
            // todo make step size not static
            return 2;
        }

        public List<Individual> sample(int n)
        {
            List<Individual> offspring = new ArrayList<Individual>();
            for(int i = 0; i < n; i++){
                Individual baby = new Individual();
                baby.position = plus(mean, mult(stepSize(), mult(covariance, norm(nDim))));
                baby.fitness();
                offspring.add(baby);
            }
            return offspring;
        }

        public void selection(int mu)
        {
            // todo Make weights NOT static and resize the indiviual array
            Collections.sort(individuals);
            double[] mean = zeros(nDim);
            for(int i = 0; i < mu; i++){
                double weight = 1 / mu;
                for(int x = 0; x < nDim; x++)
                    mean[x] += weight * individuals.get(i).position[x];
            }
            while (individuals.size() > mu)
                individuals.remove(individuals.size() - 1);
            size = mu;
            this.mean = mean;
        }
    }

	public player28_morris()
	{
        rnd_ = new Random();
        evals = 0;
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

	public void run()
	{
        Tribe tribe = new Tribe();
        tribe.addRandom(100);
        while(true){
            tribe.nextGeneration();
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
