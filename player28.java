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
    double maxFitness;

    // Parameters
    int lambda;
    int mu;
    boolean verbose;

    player28_alt altRun;

	public player28()
	{
        rnd_ = new Random();
        tribes = new ArrayList<Population>();
        altRun = new player28_alt();
	}

	public void setSeed(long seed)
	{
		// Set seed of algortihms random process
        rnd_.setSeed(seed);
        altRun.setSeed(seed);
	}

	public void setEvaluation(ContestEvaluation evaluation)
	{
        altRun.setEvaluation(evaluation);
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
            // lambda = 75;
            lambda = 31;
        // Katsuura
        }else if(isMultimodal && !hasStructure && !isSeparable){
            int maxeval = 1000000;
            // lambda = 195;
            lambda = 100;
        }
        if(System.getProperty("lambda") != null){
            lambda = Integer.parseInt(System.getProperty("lambda"));
        }
        mu = lambda/2;
    }

    public double eval(double[] point)
    {
        evals++;
        double fitness = (double) evaluation_.evaluate(point);
        maxFitness = Math.max(fitness, maxFitness);
        return fitness;
    }

	public void run()
	{
        if(lambda == 31){
            altRun.run();
        }
        evals = 0;

        Population population = new Population(lambda, mu);
        tribes.add(population);

        boolean notFinished = true;
        while(notFinished){
            for(int i = 0; i < tribes.size(); i++){
                if (verbose)
                    tribes.get(i).report();
                int nTribes = tribes.size();
                tribes.get(i).reproduction();
                tribes.get(i).selection();
                if (evals < evaluations_limit_ - 1)
                    tribes.get(i).adapt();
                eval(tribes.get(i).mean);
                tribes.get(i).generation++;
                // tribes.get(i).split();
                tribes.get(i).restart();
                tribes.get(i).last_gen_maxfitness = max(tribes.get(i).fitness());
                if(maxFitness == 10.0 || evals == evaluations_limit_){
                    notFinished = false;
                    break;
                }
            }
            ArrayList<Integer> DyingTribes = new ArrayList<Integer>();
            for (int i = 0; i < tribes.size(); i++) {
                for (int j = 0; j < tribes.size(); j++) {
                    if(i==j || DyingTribes.contains(i) || DyingTribes.contains(j) || tribes.get(j).generation < 10) continue;
                    double dist_of_means = distance(tribes.get(i).mean, tribes.get(j).mean);
                    // if(tribes.get(i).D.max() > dist_of_means)
                    if(dist_of_means < 10e-8)
                        DyingTribes.add(j);
                }
            }
            for (int i = tribes.size()-1; i>=0; i--)
                if(DyingTribes.contains(i))
                    tribes.remove(i);
        }
    }

    public double norm(double[] v)
    {
        double x = 0;
        for (int i = 0; i < v.length; i++)
            x += v[i] * v[i];
        x = Math.sqrt(x);
        return x;
    }

    public double times(double[] v, double[] w)
    {
        double x = 0;
        for (int i = 0; i < v.length; i++)
            x += v[i] * w[i];
        return x;
    }

    public double distance(double[] v, double[] w)
    {
        double distance = 0;
        for (int i = 0; i < v.length; i++) {
            distance += Math.pow(w[i] - v[i], 2.);
        }
        distance = Math.sqrt(distance);
        return distance;
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

        public double[] mean;
        public double[] zmean;
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
        public int parent_id;
        public double last_gen_maxfitness;

        public Population(int lambda, int mu)
        {
            individuals = new ArrayList<Individual>();
            id = nextTribe;
            nextTribe++;

            this.lambda = lambda;
            this.mu = mu;

            init();
        }

        public void init()
        {
            generation = 0;
            sigma = 0.3 * maxPos;
            C = Matrix.identity(nDim, nDim);
            V = Matrix.identity(nDim, nDim);
            D = Matrix.identity(nDim, nDim);
            mean = new double[nDim];
            zmean = new double[nDim];
            pc = new double[nDim];
            ps = new double[nDim];
            resetParameters();
        }

        public void resetParameters()
        {
            weights = new double[mu];
            for (int i = 0; i < mu; i++)
                weights[i] = Math.log((double)mu + 0.5) - Math.log(i+1);
            weights = normalize(weights);
            mueff = Math.pow(sum(weights), 2) / Math.pow(norm(weights), 2);

            double N = (double)nDim;
            cc = (4. + mueff/N) / (N + 4. + 2.*mueff/N);
            // cs = (mueff+2.) / (N+mueff+5.);
            cs = 0.6;
            c1 = 2. / (Math.pow(N+1.3, 2)+mueff);
            cmu = Math.min(1.-c1, 2. * (mueff-2.+1./mueff)/(Math.pow(N+2., 2)+mueff));
            // damps = 1. + 2 * Math.max(0, Math.sqrt((mueff-1.)/(N+1.))-1) + cs;
            damps = 2. * mueff/(double)lambda + 0.3 + cs;
            chiN = Math.sqrt(N) * (1. - 1./(4.*N) + 1./(21.*N*N));
            // chiN = 1. - 1./(4.*N) + 1./(21.*N*N);
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

        public Matrix zpositions()
        {
            int size = individuals.size();
            Matrix zpositions = new Matrix(size, nDim);
            for(int i = 0; i < size; i++) {
                for (int j = 0; j < nDim; j++) {
                    zpositions.set(i, j, individuals.get(i).zposition[j]);
                }
            }
            return zpositions;

        }

        public double[] fitness()
        {
            int size = individuals.size();
            double[] fitness = new double[size];
            for(int i = 0; i < size; i++)
                fitness[i] = individuals.get(i).fitness();
            return fitness;
        }

        public void reproduction()
        {
            individuals = new ArrayList<Individual>();
            for(int i = 0; i < lambda; i++) {
                reproduce();
                if(evals == evaluations_limit_ - 1) break;
            }
        }

        public void reproduce()
        {
            Individual baby = new Individual();

            for (int i = 0; i < nDim; i++) {
                baby.zposition[i] = rnd_.nextGaussian(); // N(0,1)
                double s = baby.zposition[i] * Math.sqrt(Math.max(0,D.get(i,i))); // D·N(0,1)
                for (int j = 0; j < nDim; j++)
                    baby.position[j] += s * V.get(j,i); // V·D·N(0,1) = N(0,C)
            }
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

        public void calculateMean()
        {
            mean = new double[nDim];
            zmean = new double[nDim];
            for (int i = 0; i < individuals.size(); i++){
                for (int j = 0; j < nDim; j++){
                    mean[j] += individuals.get(i).position[j] * weights[i];
                    zmean[j] += individuals.get(i).zposition[j] * weights[i];
                }
            }
        }

        public void adapt()
        {
            double N = (double)nDim;

            double[] old_mean = mean.clone();
            calculateMean();

            double[] vzmean = V.times(zmean);
            for (int i = 0; i < nDim; i++){
                ps[i] = (1.-cs)*ps[i] +  Math.sqrt(cs*(2.-cs)*mueff) * vzmean[i];
            }

            // General step-size adaption
            // double sigmafac = Math.min(1., cn/2. * (Math.pow(norm(ps), 2.) / N - 1.)); // tutorial: eq.33
            double sigmafac = norm(ps) / (Math.sqrt(N) + 1./N) - 1; // tutorial: eq.32
            // double sigmafac = (norm(ps)/chiN - 1.); // From paper
            sigma *= Math.exp(Math.min(1., cs/damps * sigmafac));


            // double hsig = (norm(ps) / Math.sqrt(1.-Math.pow(1.-cs, 2.*evals/lambda)) / chiN) < (1.4 + 2./(N+1.)) ? 1. : 0.;
            double hsig = (Math.pow(norm(ps),2.)/N  / (1. - Math.pow(1.-cs, 2.*(double)evals/(double)lambda))) < (2. + 4./(N+1.)) ? 1. : 0.;

            // Covariance adaption
            // Rank one update

            // Update cov hist path
            double[] vdzmean = V.times(D.times(zmean));
            for (int i = 0; i < nDim; i++)
                pc[i] = (1.-cc)*pc[i] + hsig * Math.sqrt(cc*(2.-cc)*mueff) * vdzmean[i];

            double c1a = c1 * (1. - (1.-hsig*hsig) * cc * (2.-cc));
            C.timesEquals(1.-c1a-cmu * sum(weights)); // Regard old matrix

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

            for (int i = 0; i < nDim; i++)
                for (int j = 0; j < i; j++)
                        C.set(i,j,C.get(j,i));

            // Decompose convariance matrix.
            EigenvalueDecomposition eig = new EigenvalueDecomposition(C);
            V = eig.getV();
            D = eig.getD();
        }

        public void split()
        {
            if(tribes.size() > 3) return;
            if(generation < 10) return;

            double sigmaMean = 0;
            double sigmaMax = 0;
            int sigmaMaxI = 0;
            for (int i = 0; i < nDim; i++) {
                sigmaMean += D.get(i,i);
                if (D.get(i,i) > sigmaMax) {
                    sigmaMax = D.get(i,i);
                    sigmaMaxI = i;
                }
            }
            sigmaMean /= (double)nDim;
            double sigmaSTD = 0;
            for (int i = 0; i < nDim; i++)
                sigmaSTD += Math.pow(D.get(i,i) - sigmaMean, 2.);
            sigmaSTD = Math.sqrt(sigmaSTD/(double)nDim);

            // Finding biggest and second biggest eigenvalue and index of eigenvector
            double[] Vmax = new double[nDim];
            for (int i = 0; i < nDim; i++)
                Vmax[i] = V.get(sigmaMaxI, i);

            // Spiltting condition
            if ((sigmaMax - sigmaMean)/sigmaSTD < 2.5)
                return;

            Population other = new Population(lambda, mu);
            tribes.add(other);
            other.parent_id = this.id;
            other.sigma = sigma;
            generation = 0;
            int old_mu = this.mu;
            // double newSigma = sigmaMax/2. + 0.1 * sigmaMax; //Half but overlapping
            double newSigma = sigmaMean;

            // Split population according to dot-product
            ListIterator<Individual> indiviualIter = individuals.listIterator();
            while(indiviualIter.hasNext()) {
                Individual I = indiviualIter.next();

                double strength = 0;
                for (int i = 0; i < nDim; i++)
                    strength += (I.position[i] - mean[i]) * V.get(sigmaMaxI, i);

                if(strength > 0) {
                    other.individuals.add(I);
                    indiviualIter.remove();
                }else if(strength == 0){ //Item on splitting plane!
                    indiviualIter.remove();
                }
            }

            // Update mean and historic path in both populations
            Population[] p = {this, other};
            for (int i = 0; i < 1; i++) {
                p[i].mu = p[i].individuals.size();
                double[] oldMean = p[i].mean.clone();
                p[i].resetParameters();
                p[i].calculateMean();
                p[i].mu = old_mu;
                p[i].resetParameters();
                for (int j = 0; j < nDim; j++) {
                    p[i].ps[j] += p[i].mean[j] - oldMean[j];
                    p[i].pc[j] += p[i].mean[j] - oldMean[j];
                }
            }

            // Compute the the splitted covariance matrix
            D.set(sigmaMaxI, sigmaMaxI, sigmaMean);
            C = V.times(D.times(V.transpose()));
            other.C = this.C.copy();
            other.D = this.D.copy();
            other.V = this.V.copy();
        }

        public void restart()
        {
            double flatFit = individuals.get(0).fitness - individuals.get((int)(0.5 * (float)individuals.size())).fitness;
            double Conditioning = Math.abs(D.max() / D.minNonZero());
            if((sigma > 5 || flatFit < 1e-20 || Conditioning > 1e5)){
                init();
            }
        }

        public void report()
        {
            if (generation == 0) return;

            double sigmaSTD = 0;
            double sigmaMean = 0;
            double sigmaMax = 0;
            for (int i = 0; i < nDim; i++) {
                sigmaMean += D.get(i,i);
                sigmaMax = Math.max(sigmaMax, D.get(i,i));
            }
            sigmaMean /= (double)nDim;
            for (int i = 0; i < nDim; i++)
                sigmaSTD += Math.pow(D.get(i,i) - sigmaMean, 2.);
            sigmaSTD = Math.sqrt(sigmaSTD/(double)nDim);
            double N = (double)nDim;

            System.out.format("%d,", id);
            System.out.format("%d,", evals);

            System.out.format("%.10f,", individuals.get(0).fitness());
            System.out.format("%.10f,", sigma);
            System.out.format("%6.3e,", (sigmaMax - sigmaMean)/sigmaSTD);
            // System.out.format("%6.3e,", (norm(ps)));
            System.out.format("%6.3e,", C.max());
            System.out.format("%6.3e,", sigmaMean);
            for (int i = 0; i < nDim; i++){
                // System.out.format("%6.3e,", D.get(i,i));
                System.out.format("%6.3e,", ps[i]);
            }
            System.out.format("%6.3e", sigmaMean);
            System.out.println();
        }
    }

    public class Individual implements Comparable<Individual>
    {
        public double[] position;
        public double[] zposition;
        public double fitness;

        public Individual()
        {
            position = new double[nDim];
            zposition = new double[nDim];
            fitness = 0;
        }

        public double fitness()
        {
            if(fitness == 0){
                fitness = eval(position);
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
