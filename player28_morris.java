import org.vu.contest.ContestSubmission;
import org.vu.contest.ContestEvaluation;

import java.lang.Object;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Properties;
import java.util.Random;
import java.util.ListIterator;
import package.population;

public class player28_morris implements ContestSubmission
{
    Random rnd_;
	ContestEvaluation evaluation_;
    private int evaluations_limit_;
    protected static final double maxPos = 5.0D;
    protected static final int nDim = 10;
    protected static final int nPopulations = 8;
    int evals;

    public class Individual implements Comparable<Individual>
    {
        public double[] position;
        public double fitness;
        public int age;

        public Individual()
        {
            position = new double[nDim];
            fitness = 0;
            age = 1;
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

    public class Matrix
    {
        public final int N;
        public final int Dim;
        public final double[][] data;

        public Matrix(int N, int Dim)
        {
            this.N = N;
            this.Dim = Dim;
            data = new double[N][Dim];
        }

        public Matrix(double[][] data)
        {
            N = data.length;
            Dim = data[0].length;
            this.data = new double[N][Dim];
            for (int i = 0; i < N; i++)
                for (int d = 0; d < Dim; d++)
                        this.data[i][d] = data[i][d];
        }

        private Matrix(Matrix other)
        {
            this(other.data);
        }

        public Matrix plus(Vector vector)
        {
            Matrix result = new Matrix(this);
            for(int i = 0; i < N; i++)
                for(int d = 0; d < Dim; d++)
                    result.data[i][d] += vector.data[d];
            return result;
        }

        public Matrix times(double scalar)
        {
            Matrix result = new Matrix(this);
            for(int i = 0; i < N; i++)
                for(int d = 0; d < Dim; d++)
                    result.data[i][d] *= scalar;
            return result;
        }

        public Vector times(Vector vector)
        {
            Vector result = new Vector(Dim);
            for(int i = 0; i < N; i++)
                for(int d = 0; d < Dim; d++)
                    result.data[d] += data[i][d] * vector.data[d];
            return result;
        }

        public Vector mean()
        {
            Vector mean = new Vector(Dim);
            for(int n = 0; n < N; n++)
                for(int d = 0; d < Dim; d++)
                    mean.data[d] += data[n][d];
            for(int d = 0; d < Dim; d++)
                mean.data[d] /= N;
            return mean;
        }

        public double max()
        {
            double max = 0;
            for (int i = 0; i < N; i++)
                for (int d = 0; d < Dim; d++)
                    if(data[i][d] > max)
                        max = data[i][d];
            return max;
        }

        public Matrix covariance()
        {
            Matrix covariance = new Matrix(Dim, Dim);
            Vector mean = mean();
            for(int i = 0; i < N; i++)
                for(int d = i; d < Dim; d++){
                    double c = 0;
                    for(int h = 0; h < N; h++)
                        c +=  (data[h][i] - mean.data[i]) * (data[h][d] - mean.data[d]);
                    c /= N - 1;
                    covariance.data[i][d] = c;
                    covariance.data[d][i] = c;
                }
            return covariance;
        }

        public Matrix cholesky()
        {
            Matrix L = new Matrix(N, N);

            for (int i = 0; i < N; i++){
                for (int j = 0; j <= i; j++){
                    double sum = 0.0;
                    for (int k = 0; k < j; k++)
                        sum += L.data[i][k] * L.data[j][k];
                    if (i == j) L.data[i][i] = Math.sqrt(data[i][i] - sum);
                    else        L.data[i][j] = 1.0 / L.data[j][j] * (data[i][j] - sum);
                }
                if (L.data[i][i] <= 0)
                    throw new RuntimeException("Matrix not positive definite");
            }
            return L;
        }

        public void print()
        {
            for(int i = 0; i < N; i++)
                System.out.println(Arrays.toString(data[i]));
            System.out.println();
        }
    }
    public class Vector
    {
        public final int length;
        public final double[] data;

        public Vector(int length)
        {
            this.length = length;
            data = new double[length];
        }

        public Vector(double[] data)
        {
            length = data.length;
            this.data = new double[length];
            for (int l = 0; l < length; l++)
                    this.data[l] = data[l];
        }

        private Vector(Vector other)
        {
            this(other.data);
        }

        public double mean()
        {
            double sum = 0;
            for(int i = 0; i < length; i++)
                sum += data[i];
            return sum / (double)length;
        }

        public Vector abs()
        {
            Vector result = new Vector(this);
            for (int i = 0; i < length; i++)
                if (result.data[i] < 0)
                    result.data[i] = -result.data[i];
            return result;
        }

        public double max()
        {
            double max = 0;
            for (int i = 0; i < length; i++)
                if (data[i] > max)
                    max = data[i];
            return max;
        }

        public Vector plus(double scalar)
        {
            Vector result = new Vector(this);
            for (int i = 0; i < length; i++)
                result.data[i] += scalar;
            return result;
        }

        public Vector plus(Vector other)
        {
            Vector result = new Vector(this);
            for(int i = 0; i < length; i++)
                result.data[i] += other.data[i];
            return result;
        }

        public Vector minus(Vector other)
        {
            Vector result = new Vector(this);
            for(int i = 0; i < length; i++)
                result.data[i] -= other.data[i];
            return result;
        }

        public Vector times(double scalar)
        {
            Vector result = new Vector(this);
            for(int i = 0; i < length; i++)
                result.data[i] *= scalar;
            return result;
        }

        public void print()
        {
            for (int i = 0; i < length; i++){
                System.out.print(data[i]);
                System.out.print(" ");
            }
            System.out.println();
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
        int population_size = 100;
        double birthrate = 2D;

        Population population = new Population(birthrate);
        population.addRandom(population_size);
        population.selection(population_size);
        while(population.nextGeneration()){
            population.report();
        }
    }

    public Matrix sample(Vector mean, Matrix cov, int n)
    {
        Matrix result = new Matrix(n, cov.N);
        Matrix L = cov.cholesky();
        for (int i = 0; i < n; i++) {
            Vector NI = new Vector(randn(cov.N));
            result.data[i] = L.times(NI).plus(mean).data;
        }
        return result;
    }

    public double[] rand(int length, double boundary)
    {
        double[] result = new double[length];
        for(int i = 0; i < length; i++)
        result[i] = -boundary + 2 * boundary * rnd_.nextDouble();
        return result;
    }

    public double[] randn(int length)
    {
        double[] result = new double[length];
        for(int i = 0; i < length; i++)
            result[i] = rnd_.nextGaussian();
        return result;
    }
}