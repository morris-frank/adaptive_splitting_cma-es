import org.vu.contest.ContestSubmission;
import org.vu.contest.ContestEvaluation;

import java.util.Random;
import java.util.Properties;
import java.util.Arrays;
import java.lang.Object;

public class player28_morris implements ContestSubmission
{
    Random rnd_;
	ContestEvaluation evaluation_;
    private int evaluations_limit_;
    protected static final double maxPos = 5.0D;
    protected static final int nDim = 10;
    protected static final int nInd = 100;
    protected static final int nClans = 8;
    double birthrate = 0.5D;
    int evals;

    public class Individual implements Comparable<Individual>
    {
        public int age;
        public double[] position;
        public double gradient;
        public double fitness;
        public int clan;

        public Individual()
        {
            fitness = 0;
            age = 1;
            position = new double[nDim];
            gradient = 0.0D;
            // gradient = new  double[nDim];
            // for(int i = 0; i < nDim; i++){
            //     gradient[i] = 0.0D;
            // }
        }

        @Override
        public int compareTo(Individual other) {
            if(this.fitness < other.fitness)
                return -1;
            else if(other.fitness < this.fitness)
                return 1;
            return 0;
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
        Individual[] population = generateRandomPopulation();

        while(evals < evaluations_limit_){
            newYear(population);
            Individual[] children = nextGeneration(population);
            population = chooseSurvivors(population, children);
        }
    }

    public Individual generateRandomIndividual()
    {
        Individual individual = new Individual();
        individual.position = generateRandomPosition();
        checkFitness(individual);
        return individual;
    }

    public double[] generateRandomPosition()
    {
        double[] position = new double[nDim];
        for(int i = 0; i < nDim; i++){
            position[i] = -maxPos + 2 * maxPos * rnd_.nextDouble();
        }
        return position;
    }

    public Individual[] generateRandomPopulation()
    {
        Individual[] population = new Individual[nInd];
        for(int i = 0; i < nInd; i++){
            population[i] = generateRandomIndividual();
        }
        return population;
    }

    public void checkFitness(Individual[] population)
    {
        for(Individual individual : population){
            checkFitness(individual);
        }
    }

    public void checkFitness(Individual individual)
    {
        evals++;
        if (individual.fitness == 0){
            individual.fitness = (double) evaluation_.evaluate(individual.position);
        }
    }

    public Individual crossover(Individual mama, Individual papa)
    {
        Individual baby = new Individual();

        // The cut-off intersection index between the two chromosoms
        int handle = rnd_.nextInt(nDim) + 1;
        // Who shots first?
        boolean mamaFirst = rnd_.nextBoolean();

        for(int i = 0; i < nDim; i++){
            if (i < handle){
                baby.position[i] = mamaFirst ? mama.position[i]: papa.position[i];
            } else {
                baby.position[i] = mamaFirst ? papa.position[i]: mama.position[i];
            }
        }

        mutate(baby);
        checkFitness(baby);

        double mamasInfluence = (double)handle/(double)nDim;
        mamasInfluence = mamaFirst ? mamasInfluence : 1 - mamasInfluence;
        baby.gradient =  mamasInfluence * (baby.fitness - mama.fitness) + (1 - mamasInfluence) * (baby.fitness - papa.fitness);

        return baby;
    }

    public void newYear(Individual[] population){
        for(Individual individual : population){
            individual.age++;
        }
    }

    public Individual[] nextGeneration(Individual[] population)
    {
        int nChild = (int)((double)population.length * birthrate);
        if(nChild > (evaluations_limit_ - evals)){
            nChild = evaluations_limit_ - evals;
        }
        Individual[] children = new Individual[nChild];
        for(int i = 0; i < nChild; i++){
            Individual papa = population[rnd_.nextInt(population.length)];
            Individual mama = population[rnd_.nextInt(population.length)];
            children[i] = crossover(mama, papa);
        }
        return children;
    }

    public Individual[] chooseSurvivors(Individual[] parents, Individual[] children){

        Arrays.sort(parents);
        Arrays.sort(children);

        int p = parents.length - 1;
        for(int c = children.length - 1; c >= 0; c--){
            for(; p >= 0; p--){
                if(children[c].compareTo(parents[p]) > 0){
                    parents[p] = children[c];
                }
            }
        }

        return parents;
    }

    public void mutate(Individual individual)
    {
        mutate(individual, 0.4, 0.1);
    }

    public void mutate(Individual individual, double sigma, double p)
    {
        for(int i = 0; i < nDim; i++){
            if(rnd_.nextDouble() < p){
                individual.position[i] += rnd_.nextGaussian() * sigma;
            }
        }
    }
}
