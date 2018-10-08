import org.vu.contest.ContestEvaluation;
import org.vu.contest.ContestSubmission;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.ListIterator;
import java.util.Properties;
import java.util.Random;

import Jama.EigenvalueDecomposition;
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
    int lambda;
    int mu;
    boolean verbose;

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


        verbose = Boolean.parseBoolean(System.getProperty("verbose"));

        // BentCigar
        if(!isMultimodal && !hasStructure && !isSeparable){
            int maxeval = 10000;
            lambda = 11;
        // Schaffers
        }else if(isMultimodal && hasStructure && !isSeparable){
            int maxeval = 100000;
            lambda = 100;

        // Katsuura
        }else if(isMultimodal && !hasStructure && !isSeparable){
            int maxeval = 1000000;
            lambda = 200;
        }
        if(System.getProperty("lambda") != null){
            lambda = Integer.parseInt(System.getProperty("lambda"));
        }
        mu = lambda/2;
    }

	public void run()
	{
        evals = 0;

        Population population = new Population(lambda, mu);
        population.initRandom();
        tribes.add(population);

        boolean somethinLeft = true;
        while(somethinLeft){
            for (int i = 0; i < tribes.size(); i++)
                somethinLeft = tribes.get(i).nextGeneration(lambda, mu);
        }
    }

    public Matrix sample(Matrix V, Matrix D, int N)
    {
        Matrix X = new Matrix(N, V.getRowDimension());
        for (int n = 0; n < N; n++)
            X.setRow(n, sample(V, D));
        return X;
    }

    public double[] sample(Matrix V, Matrix D)
    {
        int d = V.getRowDimension();
        double[] X = new double[d];
        for (int i = 0; i < d; i++) {
            double s = rnd_.nextGaussian() * Math.sqrt(Math.max(0,D.get(i,i)));
            for (int j = 0; j < d; j++)
                X[j] += s * V.get(j,i);
        }
        return X;
    }

    public double norm(double[] v)
    {
        double x = 0;
        for (int i = 0; i < v.length; i++)
            x += v[i] * v[i];
        x = Math.sqrt(x);
        return x;
    }

    public double mean(double[] v)
    {
        double X = 0;
        for (int i = 0; i < v.length; i++) {
            X += v[i];
        }
        X /= (double)v.length;
        return X;
    }

    public double max(double[] v)
    {
        double X = v[0];
        for (int i = 0; i < v.length; i++) {
            X = Math.max(X, v[i]);
        }
        return X;
    }

    public double[] normalize(double[] v)
    {
        double[] x = new double[v.length];
        double sum = sum(v);
        for (int i = 0; i < v.length; i++)
            x[i] = v[i]/sum;
        return x;
    }

    public double sum(double[] v)
    {
        double x = 0;
        for (int i = 0; i < v.length; i++)
            x += v[i];
        return x;
    }

    public void print(double[] v)
    {
        for (int i = 0; i < v.length; i++)
            System.out.format(" %1.2f", v[i]);
        System.out.println();
    }

    public double[] rand(int length, double boundary)
    {
        double[] result = new double[length];
        for(int i = 0; i < length; i++)
        result[i] = -boundary + 2 * boundary * rnd_.nextDouble();
        return result;
    }

    public class Population
    {
        public List<Individual> individuals;

        public int lambda;
        public int mu;

        public Matrix C;
        public Matrix V;
        public Matrix D;
        public Matrix invSqrtC;

        public double[] mean;
        public int generation;

        public double[] weights;
        public double mueff;
        public double cc;
        public double cs;
        public double c1;
        public double cmu;
        public double damps;
        public double chiN;

        public double[] pc;
        public double[] ps;

        public double sigma;
        public int id;

        public Population(int lambda, int mu)
        {
            generation = 1;
            individuals = new ArrayList<Individual>();
            id = nextTribe;
            nextTribe++;

            this.lambda = lambda;
            this.mu = mu;

            sigma = 0.3 * maxPos;
            C = Matrix.identity(nDim, nDim);
            V = Matrix.identity(nDim, nDim);
            D = Matrix.identity(nDim, nDim);
            invSqrtC = Matrix.identity(nDim, nDim);
            mean = new double[nDim];
            pc = new double[nDim];
            ps = new double[nDim];
            genWeights();
            resetParameters();
        }

        private void genWeights()
        {
            weights = new double[mu];
            for (int i = 0; i < mu; i++)
                weights[i] = Math.log((double)mu + 0.5) - Math.log(i+1);
            weights = normalize(weights);
            mueff = Math.pow(sum(weights), 2) / Math.pow(norm(weights), 2);
        }

        public void resetParameters()
        {
            double N = (double)nDim;
            cc = (4. + mueff/N) / (N + 4. + 2.*mueff/N);
            cs = (mueff+2.) / (N+mueff+5.);
            c1 = 2. / (Math.pow(N+1.3, 2)+mueff);
            cmu = Math.min(1.-c1, 2. * (mueff-2.+1./mueff)/(Math.pow(N+2., 2)+mueff));
            // damps = 1. + 2 * Math.max(0, Math.sqrt((mueff-1.)/(N+1.))-1) + cs;
            damps = 2. * mueff/(double)lambda + 0.3 + cs;
            // chiN = Math.sqrt(N) * (1. - 1./(4.*N) + 1./(21.*N*N));
            chiN = 1. - 1./(4.*N) + 1./(21.*N*N);
        }

        public void initRandom()
        {
            individuals = new ArrayList<Individual>();
            for (int i = 0; i < lambda; i++) {
                Individual I = new Individual();
                I.position = rand(nDim, maxPos);
                individuals.add(I);
            }
            C = positions().covariance();
            mean = new double[nDim];
            updateCovariance();
        }

        public Matrix positions()
        {
            int size = individuals.size();
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
            int size = individuals.size();
            double[] ages = new double[size];
            for(int i = 0; i < size; i++)
                ages[i] = individuals.get(i).age;
            return ages;
        }

        public double[] fitness()
        {
            int size = individuals.size();
            double[] fitness = new double[size];
            for(int i = 0; i < size; i++)
                fitness[i] = individuals.get(i).fitness();
            return fitness;
        }

        public void mature()
        {
            generation++;
            for(int i = 0; i < individuals.size(); i++)
                individuals.get(i).age++;
        }

        public void report()
        {
            System.out.format(" #%d : %d", id, generation);
            // System.out.format(" | MAX-D: %6.2e", D.max());
            System.out.format(" | MAXFit: %.10f", max(fitness()));
            System.out.format(" | SIGMA: %.10f", sigma);
            System.out.println();
        }

        public void reproduction()
        {
            individuals = new ArrayList<Individual>();
            for(int i = 0; i < lambda; i++) {
                reproduce();
                if(evals == evaluations_limit_) break;
            }
        }

        public void reproduce()
        {
            Individual baby = new Individual();
            baby.position = sample(V, D); // N(0,C)
            for (int j = 0; j < nDim; j++)
                baby.position[j] = baby.position[j] * sigma + mean[j]; // N(0,C) * sigma + m
            baby.fitness();
            individuals.add(baby);
        }

        public void selection()
        {
            // Sort by fitness
            Collections.sort(individuals);

            // only let fitesst babies survive
            while (individuals.size() > mu)
                individuals.remove(individuals.size() - 1);
        }

        public void adapt()
        {
            double N = (double)nDim;

            double[] old_mean = mean;

            mean = new double[nDim];
            for (int i = 0; i < mu; i++)
                for (int j = 0; j < nDim; j++)
                    mean[j] += individuals.get(i).position[j] * weights[i];

            // Normalized mean difference
            double[] mean_diff = new double[nDim];
            for (int i = 0; i < nDim; i++)
                mean_diff[i] = (mean[i] - old_mean[i]) / sigma;

            double[] meanDiffC = invSqrtC.times(mean_diff);
            for (int i = 0; i < nDim; i++)
                ps[i] = (1.-cs)*ps[i] +  Math.sqrt(cs*(2.-cs)*mueff) * meanDiffC[i];

            // General step-size adaption
            double cn = cs/damps;
            double sum_squared_ps = Math.pow(norm(ps), 2.);
            sigma *= Math.exp(Math.min(1, cn * (sum_squared_ps / N - 1.)/2));

            // double hsig = (norm(ps) / Math.sqrt(1.-Math.pow(1.-cs, 2.*evals/lambda)) / chiN) < (1.4 + 2./(N+1.)) ? 1. : 0.;
            double hsig = (Math.pow(norm(ps),2.)/N  / (1. - Math.pow(1.-cs, 2.*(double)evals/(double)lambda))) < (2. + 4./(N+1.)) ? 1. : 0.;

            // Rank one update
            for (int i = 0; i < nDim; i++)
                pc[i] = (1.-cc)*pc[i] + hsig * Math.sqrt(cc*(2.-cc)*mueff) * mean_diff[i];
            double c1a = c1 * (1. - (1.-hsig*hsig) * cc * (2.-cc));
            C.timesEquals(1.-c1a-cmu * sum(weights));
            double[][] pcOuterProductValues = new double[nDim][nDim];
            for (int i = 0; i < nDim; i++)
                for (int j = 0; j < nDim; j++)
                    pcOuterProductValues[i][j] = c1 * pc[i] * pc[j];
            Matrix pcOuterProduct = new Matrix(pcOuterProductValues, nDim, nDim);
            C.plusEquals(pcOuterProduct);


            // Rank mu Update
            for (int n = 0; n < mu; n++) {
                double[] pos = individuals.get(n).position;
                double[][] posOuterProductValues = new double[nDim][nDim];
                for (int i = 0; i < nDim; i++)
                    for (int j = 0; j < nDim; j++)
                        posOuterProductValues[i][j] = weights[n] * cmu * (pos[i] - old_mean[i]) * (pos[j] - old_mean[j]);
                Matrix posOuterProduct = new Matrix(posOuterProductValues, nDim, nDim);
                C.plusEquals(posOuterProduct);
            }

            // Covariance adaption
            // C = positions().covariance();
            for (int i = 0; i < nDim; i++)
                for (int j = 0; j < i; j++)
                        C.set(i,j,C.get(j,i));

            // Decompose convariance matrix.
            EigenvalueDecomposition eig = new EigenvalueDecomposition(C);
            V = eig.getV();
            D = eig.getD();
            double[][] invSTD = new double[nDim][nDim];
            for (int i = 0; i < nDim; i++) {
                double e = D.get(i,i);
                if(e > 0){
                    invSTD[i][i] = 1./Math.sqrt(e);
                }else{
                    invSTD[i][i] = 0.;
                }
            }
            invSqrtC = new Matrix(invSTD, nDim, nDim);
            invSqrtC = V.times(invSqrtC.times(V.transpose()));
        }

        // public void split()
        // {
        //     if (generation < 10) return;

        //     // Finding biggest eigenvalue and index of eigenvector
        //     double maxE = 0;
        //     int maxEID = 0;
        //     for (int i = 0; i < nDim; i++)
        //         if (D.get(i,i) > maxE) {
        //             maxE = D.get(i,i);
        //             maxEID = i;
        //         }

        //     // Conditioning of the current covariance matrix
        //     double condition =  maxE / D.minNonZero();

        //     if (condition < 6) return;

        //     int lambda = individuals.size();
        //     Population newTribe = new Population();

        //     ListIterator<Individual> indiIt = individuals.listIterator();
        //     while(indiIt.hasNext()) {
        //         Individual I = indiIt.next();

        //         double strength = 0;
        //         for (int i = 0; i < nDim; i++)
        //             strength += (I.position[i] - mean[i]) * V.get(maxEID, i);

        //         if(strength > 0 || (strength == 0 && rnd_.nextBoolean())) {
        //             newTribe.individuals.add(I);
        //             indiIt.remove();
        //         }
        //     }

        //     newTribe.resetTo(lambda);
        //     tribes.add(newTribe);
        //     resetTo(lambda);
        // }

        // public void resetTo(int lambda)
        // {
        //     generation = 1;
        //     mean = positions().mean();
        //     updateCovariance();
        //     if(individuals.size() < lambda)
        //         reproduce(lambda - individuals.size());
        //     genWeights();
        // }

        public void updateCovariance()
        {
            for (int i = 0; i < nDim; i++)
                for (int j = 0; j < i; j++)
                        C.set(i,j,C.get(j,i));
            EigenvalueDecomposition eig = new EigenvalueDecomposition(C);
            V = eig.getV();
            D = eig.getD();
            double[][] invSTD = new double[nDim][nDim];
            for (int i = 0; i < nDim; i++)
                invSTD[i][i] = 1./Math.sqrt(Math.max(0,D.get(i,i)));
            invSqrtC = new Matrix(invSTD, nDim, nDim);
            invSqrtC = V.times(invSqrtC.times(V.transpose()));
        }

        public boolean nextGeneration(int lambda, int mu)
        {
            reproduction();
            selection();
            if (evals < evaluations_limit_)
                adapt();
            evaluation_.evaluate(mean);evals++;
            if (verbose)
                report();
            mature();
            return (sigma < 90.) && (D.max() > 0.) && (evals < evaluations_limit_);
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
