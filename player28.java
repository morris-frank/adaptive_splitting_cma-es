import org.vu.contest.ContestEvaluation;
import org.vu.contest.ContestSubmission;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.ListIterator;
import java.util.Properties;
import java.util.Random;

import Jama.Matrix;
// import java.util.Vector;

public class player28 implements ContestSubmission
{
    Random rnd_;
	ContestEvaluation evaluation_;
    private int evaluations_limit_;
    protected static final double maxPos = 5.0D;
    protected static final int nDim = 10;
    int evals;
    int nextTribe;
    List<Population> tribes;

    // Parameters
    int init_population_size;
    double init_birthrate;

	public player28()
	{
        rnd_ = new Random();
        nextTribe = 0;
        tribes = new ArrayList<Population>();
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
        tribes.add(population);
        boolean somethinLeft = true;
        while(somethinLeft){
            int numTribes = tribes.size();
            for (int i = 0; i < numTribes; i++) {
                somethinLeft = tribes.get(i).nextGeneration();
                tribes.get(i).report();
            }
            System.out.println();
        }
    }

    public Matrix sample(Matrix covariance, int n)
    {
        Matrix X = new Matrix(n, covariance.getRowDimension());
        Matrix L = covariance.chol().getL();
        for(int i = 0; i < n; i++) {
            double[] pos = L.times(randn(covariance.getRowDimension()));
            for (int j = 0; j < covariance.getRowDimension(); j++) {
                X.set(i, j, pos[j]);
            }
        }
        return X;
    }

    public double mean(double[] v) {
        double X = 0;
        for (int i = 0; i < v.length; i++) {
            X += v[i];
        }
        X /= (double)v.length;
        return X;
    }

    public double max(double[] v) {
        double X = v[0];
        for (int i = 0; i < v.length; i++) {
            X = Math.max(X, v[i]);
        }
        return X;
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
        public double[] mean;
        public double[] meanPath;
        public int generation;
        public double[] weights;
        public double mu_weights;
        public double sigma;
        public int id;

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
            mean = new double[nDim];
            meanPath = new double[nDim];
            individuals = new ArrayList<Individual>();
            genWeights();
            id = nextTribe;
            nextTribe++;
        }

        private void genWeights()
        {
            weights = new double[size];
            int sum = 0;
            mu_weights = 0;
            for (int i = 0; i < size; i++) {
                weights[i] = size - i + 1;
                sum += weights[i];
            }
            for (int i = 0; i < size; i++){
                weights[i] /= sum;
                mu_weights += weights[i] * weights[i];
            }
            mu_weights = 1.0D/mu_weights;
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
            for(int i = 0; i < size; i++) {
                for (int j = 0; j < nDim; j++) {
                    positions.set(i, j, individuals.get(i).position[j]);
                }
            }
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
            generation++;
            for(int i = 0; i < individuals.size(); i++)
                individuals.get(i).age++;
        }

        public void report()
        {
            // System.out.format(">% 5d", generation);
            // System.out.println();
            // System.out.format(" | MAX-Fit: %6.2e", fitness().max());
            // System.out.format(" | MAX COV: %6.2e", covariance.max());
            System.out.format(" #%3d", id);
            System.out.format(" | AVG-Age: %6.2e", mean(ages()));
            System.out.format(" | MAX-Fit: %6.2e", max(fitness()));
            System.out.format(" | SIGMA: %6.2e", sigma);
            System.out.format(" | MAX-MP: %6.2e", max(meanPath));
            // System.out.format(" | %3d", individuals.size());
            System.out.println();
        }

        public void reproduce(int n)
        {
            Matrix sampled_positions = sample(covariance, n).timesEquals(sigma).plusEquals(mean);
            for(int i = 0; i < n; i++){
                Individual baby = new Individual();
                for (int j = 0; j < nDim; j++) {
                    baby.position[j] = sampled_positions.get(i, j);
                }
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
            // double mueff = (double)size/4;
            // double cc = 1 / ((Math.sqrt(nDim) + nDim)/2);
            double csigma = 4.0D / nDim;
            double dsigma = 1.0D;

            double[] new_mean = new double[nDim];

            for(int d = 0; d < nDim; d++)
                for(int i = 0; i < size; i++)
                    new_mean[d] += weights[i] * (individuals.get(i).position[d] - mean[d]);

            double mu_weights_squared = Math.sqrt(mu_weights);
            double invCSigma = Math.sqrt(1.0D - Math.pow(1.0D - csigma, 2));
            for (int d = 0; d < nDim; d++) {
                new_mean[d] += mean[d];
                meanPath[d] *= (1.0D - csigma);
                meanPath[d] += (new_mean[d] - mean[d]) / sigma * mu_weights_squared * invCSigma;
            }
            mean = new_mean;
        }

        public void updateSigma()
        {
            // double mueff = (double)size/4;
            // double csigma = (mueff + 2)/(nDim + mueff + 5);
            // double dsigma = 1 + csigma + Math.max(0, Math.sqrt((mueff - 1)/(nDim +1))-1);
            // sigma *= Math.exp(csigma / dsigma / 1000 * (meanPath.norm()/Math.sqrt(nDim) - 1));
            // sigma *= Math.exp(csigma / dsigma * (meanPath.norm()/Math.sqrt(nDim) - 1));
            // double csigma = 4.0D / nDim;
            // double dsigma = 1.0D;
            // sigma *= Math.exp(csigma / dsigma * ((meanPath.norm()/Math.sqrt(nDim)) - 1));
            sigma *= 0.999;
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
            ListIterator<Individual> indiIt = individuals.listIterator();
            while(indiIt.hasNext()) {
                Individual individual = indiIt.next();
                if(individual.age > maxAge) {
                    indiIt.remove();
                }
            }
        }

        // public void split()
        // {
        //     double maxEig = 8;

        //     Vector eigV = covariance.powerIteration();
        //     double eig = covariance.times(eigV).times(eigV)/eigV.times(eigV);

        //     if (eig > maxEig && generation > 4){
        //         Population newTribe = new Population(init_birthrate, 0.9);
        //         ListIterator<Individual> indiIt = individuals.listIterator();
        //         while(indiIt.hasNext()) {
        //             Individual individual = indiIt.next();
        //             Vector position = new Vector(individual.position);
        //             double dirStrength = position.minus(mean).times(eigV);
        //             if(dirStrength > 0 || (dirStrength == 0 && rnd_.nextBoolean())){
        //                 Individual defector = new Individual();
        //                 defector.position = individual.position;
        //                 defector.fitness = individual.fitness;
        //                 defector.age = individual.age;
        //                 newTribe.individuals.add(defector);
        //                 indiIt.remove();
        //                 newTribe.size++;
        //             }
        //         }
        //         newTribe.mean = newTribe.positions().mean();
        //         newTribe.covariance =  newTribe.positions().covariance();
        //         if(newTribe.size < 100){
        //             newTribe.reproduce(100 - newTribe.size);
        //         }
        //         newTribe.size = newTribe.individuals.size();
        //         newTribe.genWeights();
        //         newTribe.select(100);
        //         tribes.add(newTribe);
        //     }

        // }

        public boolean nextGeneration()
        {
            updateCovariance();
            // split();
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
