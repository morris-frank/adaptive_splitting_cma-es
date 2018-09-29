import org.vu.contest.ContestEvaluation;
import org.vu.contest.ContestSubmission;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.ListIterator;
import java.util.Properties;
import java.util.Random;

public class player28 implements ContestSubmission
{
    Random rnd_;
	ContestEvaluation evaluation_;
    private int evaluations_limit_;
    protected static final double maxPos = 5.0D;
    protected static final int nDim = 10;
    int evals;

    // Parameters
    int init_population_size;
    double init_birthrate;

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

        if(!isMultimodal && !hasStructure && !isSeparable){
            // BentCigar
            int maxeval = 10000;
            init_population_size = 100;
            init_birthrate = 2D;
        }else if(isMultimodal && hasStructure && !isSeparable){
            // Schaffers
            int maxeval = 100000;
            init_population_size = 100;
            init_birthrate = 2D;

        }else if(isMultimodal && !hasStructure && !isSeparable){
            // Katsuura
            int maxeval = 1000000;
            init_population_size = 100;
            init_birthrate = 7D;
        }
    }

	public void run()
	{
        evals = 0;
        Population population = new Population(init_birthrate, 0.9, 0.9, 0);
        population.addRandom(init_population_size, maxPos);
        population.selection(init_population_size);
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

    public class Population
    {
        public int size;
        public List<Individual> individuals;
        public Matrix covariance;
        public Vector mean;
        public int generation;
        public double[] weights;
        public double bump = 1.0D;

        //Parameters
        public double birthrate;
        public double lr_m;
        public double lr_c;
        public double max_age;

        public Population(double birthrate, double lr_m, double lr_c, double max_age)
        {
            this.birthrate = birthrate;
            this.lr_m = lr_m;
            this.lr_m = lr_m;
            this.max_age = max_age;
            size = 0;
            generation = 1;
            mean = new Vector(nDim);
            individuals = new ArrayList<Individual>();
            genWeights();
        }

        private void genWeights()
        {
            weights = new double[size];
            int sum = 0;
            for (int i = 0; i < size; i++) {
                weights[i] = size - i + 1;
                sum += weights[i];
            }
            for (int i = 0; i < size; i++)
                weights[i] /= sum;
        }

        public void addRandom(int n, double maxPos)
        {
            for(int i = 0; i < n; i++){
                Individual individual = new Individual();
                individual.position = rand(nDim, maxPos);
                individual.fitness();
                individuals.add(individual);
            }
            size += n;
            genWeights();
        }

        public Matrix positions()
        {
            Matrix positions = new Matrix(size, nDim);
            for(int i = 0; i < size; i++)
                positions.data[i] = individuals.get(i).position;
            return positions;
        }

        public Vector ages()
        {
            Vector ages = new Vector(size);
            for(int i = 0; i < size; i++)
                ages.data[i] = individuals.get(i).age;
            return ages;
        }

        public Vector fitness()
        {
            Vector fitness = new Vector(size);
            for(int i = 0; i < size; i++)
                fitness.data[i] = individuals.get(i).fitness();
            return fitness;
        }

        public void newYear()
        {
            generation++;
            for(int i = 0; i < individuals.size(); i++)
                individuals.get(i).age++;
        }

        public void report()
        {
            System.out.print(generation);
            System.out.print(" | AVG-Age: ");
            System.out.print(ages().mean());
            System.out.print(" | MAX-Fit: ");
            System.out.print(fitness().max());
            System.out.print(" | LR: ");
            System.out.print(bump);
            System.out.println();
        }

        public double sigma()
        {
            // todo make step size not static
            return 1 * bump;
        }

        public List<Individual> makeBabies(int n)
        {
            List<Individual> offspring = new ArrayList<Individual>();
            Matrix sampled_positions = sample(mean, covariance, n);
            for(int i = 0; i < n; i++){
                Individual baby = new Individual();
                baby.position = sampled_positions.data[i];
                baby.fitness();
                if(evals == evaluations_limit_) break;
                offspring.add(baby);
            }
            return offspring;
        }

        public void selection(int mu)
        {
            Collections.sort(individuals);
            while (individuals.size() > mu)
                individuals.remove(individuals.size() - 1);
            size = mu;

            Vector new_mean = new Vector(nDim);

            for(int d = 0; d < nDim; d++) {
                for(int i = 0; i < mu; i++)
                    new_mean.data[d] += weights[i] * (individuals.get(i).position[d] - mean.data[d]);
                new_mean.data[d] *= lr_m;
                new_mean.data[d] += mean.data[d];
            }

            if(mean.minus(new_mean).mean() < 10E-8) bump *= 1.5;
            else bump = 1;
            mean = new_mean;
        }

        public void killElderly(int maxAge)
        {
            if(maxAge > 0){
                ListIterator<Individual> indiIt = individuals.listIterator();
                while(indiIt.hasNext()) {
                    Individual individual = indiIt.next();
                    if(individual.age > maxAge) {
                        indiIt.remove();
                    }
                }
            }
        }

        public boolean nextGeneration()
        {
            if(generation == 1) {
                covariance = positions().covariance();
            } else {
                double d_C = 0.1;
                covariance = covariance.times(d_C).plus(positions().covariance().times(1.0D - d_C));
            }
            List<Individual> babies = makeBabies((int)(size * birthrate));
            individuals.addAll(babies);
            // killElderly(5);
            selection(size);
            newYear();
            return evals < evaluations_limit_;
        }
    }

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
}
