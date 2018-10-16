import org.vu.contest.ContestEvaluation;
import org.vu.contest.ContestSubmission;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.ListIterator;
import java.util.Properties;
import java.util.Random;

public class player28_alt implements ContestSubmission
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

	public player28_alt()
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
            init_birthrate = 0.5D;
        }else if(isMultimodal && hasStructure && !isSeparable){
            // Schaffers
            int maxeval = 100000;
            init_population_size = 100;
            init_birthrate = 2D;

        }else if(isMultimodal && !hasStructure && !isSeparable){
            // Katsuura
            int maxeval = 1000000;
            init_population_size = 100;
            init_birthrate = 5D;
        }
    }

	public void run()
	{
        evals = 0;
        Population population = new Population(init_birthrate, 0.9);
        population.addRandom(init_population_size, maxPos);
        population.select(init_population_size);
        while(population.nextGeneration()){
            // population.report();
        }
    }

    public altMatrix sample(Vector mean, altMatrix cov, int n)
    {
        altMatrix result = new altMatrix(n, cov.N);
        altMatrix L = cov.cholesky();
        for (int i = 0; i < n; i++) {
            Vector NI = new Vector(randn(cov.N));
            result.data[i] = L.times(NI).plus(mean).data;
        }
        return result;
    }

    public altMatrix sample(altMatrix cov, int n)
    {
        altMatrix result = new altMatrix(n, cov.N);
        altMatrix L = cov.cholesky();
        for(int i = 0; i < n; i++){
            Vector NI = new Vector(randn(cov.N));
            result.data[i] = L.times(NI).data;
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
        public altMatrix covariance;
        public Vector mean;
        public Vector meanPath;
        public int generation;
        public double[] weights;
        public double sigma;

        //Parameters
        public double birthrate;
        public double lr;

        public Population(double birthrate, double lr)
        {
            this.birthrate = birthrate;
            this.lr = lr;
            size = 0;
            sigma = 1;
            generation = 1;
            mean = new Vector(nDim);
            meanPath = new Vector(nDim);
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

        public altMatrix positions()
        {
            altMatrix positions = new altMatrix(size, nDim);
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
            System.out.format(">% 5d", generation);
            System.out.format(" | AVG-Age: %6.2e", ages().mean());
            System.out.format(" | MAX-Fit: %6.2e", fitness().max());
            System.out.format(" | SIGMA: %6.2e", sigma);
            System.out.format(" | MAX COV: %6.2e", covariance.max());
            System.out.println();
        }

        public void reproduce(int n)
        {
            altMatrix sampled_positions = sample(covariance, n).times(sigma).plus(mean);
            // altMatrix sampled_positions = sample(covariance, n).plus(mean);
            for(int i = 0; i < n; i++){
                Individual baby = new Individual();
                baby.position = sampled_positions.data[i];
                baby.fitness();
                if(evals == evaluations_limit_) break;
                individuals.add(baby);
            }
        }

        public void select(int mu)
        {
            // Sort by fitness
            Collections.sort(individuals);

            // only let fitesst babies survive
            while (individuals.size() > mu)
                individuals.remove(individuals.size() - 1);
            size = mu;

            updateMean();
            updateSigma();
        }

        public void updateMean()
        {
            double mueff = (double)size/4;
            double cc = 1 / ((Math.sqrt(nDim) + nDim)/2);

            Vector new_mean = new Vector(nDim);

            for(int d = 0; d < nDim; d++)
                for(int i = 0; i < size; i++)
                    new_mean.data[d] += weights[i] * (individuals.get(i).position[d] - mean.data[d]);

            new_mean = new_mean.times(lr).plus(mean);
            meanPath = new_mean.minus(mean).times(cc).plus(meanPath.times(1 - cc));
            mean = new_mean;
        }

        public void updateSigma()
        {
            double mueff = (double)size/4;
            double csigma = (mueff + 2)/(nDim + mueff + 5);
            double dsigma = 1 + csigma + Math.max(0, Math.sqrt((mueff - 1)/(nDim +1))-1);
            sigma *= Math.exp(csigma / dsigma / 1000 * (meanPath.norm()/Math.sqrt(nDim) - 1));
        }

        public void updateCovariance()
        {
            if(generation == 1)
                covariance = positions().covariance();
            else
                covariance = covariance.times(1 - lr).plus(positions().covariance().times(lr));
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
            updateCovariance();
            reproduce((int)(size * birthrate));
            // killElderly(100);
            select(size);
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
