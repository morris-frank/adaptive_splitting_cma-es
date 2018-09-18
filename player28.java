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
}
